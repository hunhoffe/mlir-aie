//===- ObjectFifoToConduit.cpp - ObjectFIFO → Conduit IR (Pass A) -*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Pass A of the Conduit lowering pipeline: lift aie.objectfifo.* ops into
// Conduit IR.
//
// Architecture:
//
//   aie.objectfifo.*  ──┐
//                       ├──► Conduit IR ──► aie.dma_bd / aie.lock / aie.buffer
//   air.channel.*     ──┘
//
//   (this file)              (ConduitToDMA.cpp)
//
// What this pass does
// -------------------
// 1. Scans the aie.device body for aie.objectfifo ops and builds a name→tiles
//    map (producer tile, consumer tiles, element type, depth).
//
// 2. For each aie.objectfifo:
//      emits  conduit.create {name, capacity=depth*numElems,
//                             producer_tile=[col,row],
//                             consumer_tiles=[col0,row0,...],
//                             element_type=<memref type>,
//                             depth=<depth>}
//      (typed attributes; no conduit.annotate ops are emitted)
//
// 3. For each aie.objectfifo.link:
//      determines mode (distribute: 1 src, N dsts; join: N srcs, 1 dst)
//      picks memtile as the relay tile (heuristic: consumer of first src)
//      emits conduit.link {srcs, dsts, mode, memtile, offsets}
//
// 4. For each aie.objectfifo.acquire (inside core bodies):
//      emits conduit.acquire {name, count, port="Produce"|"Consume"}
//              : !conduit.window<elemType>
//      The window SSA value is threaded into each conduit.subview_access.
//
// 5. For each aie.objectfifo.subview.access:
//      emits conduit.subview_access %window {index}
//              : !conduit.window<T> -> T
//      Uses of the aie subview result are replaced with the conduit result.
//      No memref.alloc placeholder is generated.
//
// 6. For each aie.objectfifo.release:
//      emits conduit.release {name, count, port="Produce"|"Consume"}
//
// All original ObjectFIFO ops are erased after rewriting.
//
// Limitations (documented honestly)
// ----------------------------------
// - The memtile heuristic in link rewriting is approximate.
// - capacity = depth * numElems uses 1 as numElems when the memref element
//   count cannot be statically determined from the type.
// - The pass currently operates on the whole module; nested device ops are
//   handled one level deep only.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace xilinx::conduit {

#define GEN_PASS_DEF_OBJECTFIFOTOCONDUIT
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helper: count static elements in a MemRefType (returns 1 if unknown)
// ---------------------------------------------------------------------------

static int64_t numElemsInMemref(mlir::Type ty) {
  auto mref = mlir::dyn_cast<mlir::MemRefType>(ty);
  if (!mref)
    return 1;
  int64_t count = 1;
  for (int64_t d : mref.getShape()) {
    if (mlir::ShapedType::isDynamic(d))
      return 1;
    count *= d;
  }
  return count;
}

// ---------------------------------------------------------------------------
// Struct: information gathered from aie.objectfifo ops
// ---------------------------------------------------------------------------

struct FifoInfo {
  llvm::SmallVector<int64_t> producerTileArr; // [col, row]
  llvm::SmallVector<int64_t> consumerTilesArr;     // non-shim: [col0,row0,...]
  llvm::SmallVector<int64_t> shimConsumerTilesArr; // shim (row==0): [col0,row0,...]
  int64_t depth = 1;
  int64_t numElems = 1;
  mlir::MemRefType elemType; // the actual element memref type
  // Cyclostatic (CSDF) access pattern.
  // Populated in Phase 1.5 by scanning acquire counts for each consumer of this
  // fifo.  If all acquires use the same count the pattern is absent (uniform SDF).
  // If acquires vary, this holds the sequence of counts in program order.
  llvm::SmallVector<int64_t> accessPattern;
  // Per-consumer depths, when the source ObjectFIFO uses an ArrayAttr
  // elemNumber (e.g., {2 : i32, 4 : i32} meaning producer depth=2,
  // consumer 0 depth=4).  Empty when all consumers share the uniform depth.
  llvm::SmallVector<int64_t> consumerDepths;
};

// ---------------------------------------------------------------------------
// Main pass
// ---------------------------------------------------------------------------

struct ObjectFifoToConduitPass
    : impl::ObjectFifoToConduitBase<ObjectFifoToConduitPass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<ConduitDialect>();
  }

  // -----------------------------------------------------------------------
  // Shared state across phases
  // -----------------------------------------------------------------------

  /// Set to true when an unrecoverable error is detected.  Checked at
  /// phase boundaries in runOnOperation() to prevent subsequent phases
  /// from executing on corrupted state.  (CL-2: same anti-pattern as
  /// the P0-A bug in ConduitToDMA.cpp — signalPassFailure() inside a
  /// walk lambda does NOT stop the pass; it only marks the result as
  /// failed after all phases complete.)
  bool passFailed = false;

  /// Name → fifo metadata, populated by collectFifoInfo().
  llvm::DenseMap<mlir::StringAttr, FifoInfo> fifoInfoMap;

  /// Names of objectfifos that use aie_stream routing and must be skipped
  /// by Pass A. These ops are left intact for the stateful transform to handle.
  llvm::DenseSet<mlir::StringAttr> aieStreamFifoNames;

  /// ObjectFifo create ops to erase after all rewrites complete.
  llvm::SmallVector<AIE::ObjectFifoCreateOp> fifosToErase;

  /// Subview, acquire, and release ops to erase after Phase 4.
  llvm::SmallVector<AIE::ObjectFifoSubviewAccessOp> subviewsToErase;
  llvm::SmallVector<AIE::ObjectFifoAcquireOp> acquiresToErase;
  llvm::SmallVector<AIE::ObjectFifoReleaseOp> releasesToErase;

  // -----------------------------------------------------------------------
  // Phase 1: collectFifoInfo
  // -----------------------------------------------------------------------
  //
  // Walk the module to build fifoInfoMap (name → tile/depth/type info) and
  // detect cyclostatic (CSDF) access patterns from acquire op counts.

  void collectFifoInfo(mlir::ModuleOp module, mlir::MLIRContext *ctx) {
    fifoInfoMap.clear();
    aieStreamFifoNames.clear();

    // Phase 1: collect FifoInfo for all aie.objectfifo ops.
    module.walk([&](AIE::ObjectFifoCreateOp op) {
      // aie_stream ObjectFIFOs route data through the Core AXI stream port
      // rather than DMA. The correct lowering emits aie.flow(Core:N → DMA:0)
      // and places buffers/locks on the consumer tile — a fundamentally
      // different code path that Pass A does not yet implement.
      // Skip these ops entirely and leave them intact for the stateful
      // transform (--aie-objectFifo-stateful-transform) to handle.
      if (op.getAieStream().has_value()) {
        op.emitRemark(
            "objectfifo-to-conduit: aie_stream ObjectFIFO not yet supported "
            "via Conduit path; skipping (op left intact for stateful transform)");
        aieStreamFifoNames.insert(op.getSymNameAttr());
        return;
      }

      FifoInfo info;
      // Producer tile
      auto prodTile =
          mlir::cast<AIE::TileOp>(op.getProducerTile().getDefiningOp());
      info.producerTileArr = {prodTile.getCol(), prodTile.getRow()};
      // Consumer tiles: separate shim tiles (row==0) from compute tiles.
      // Shim tiles are DMA endpoints with no local memory; Pass C handles
      // them via aie.shim_dma_allocation rather than aie.buffer + aie.lock.
      for (mlir::Value cons : op.getConsumerTiles()) {
        auto consTile = mlir::cast<AIE::TileOp>(cons.getDefiningOp());
        int64_t col = consTile.getCol();
        int64_t row = consTile.getRow();
        if (row == 0) {
          // Shim tile (row==0): no local memory, handled separately.
          info.shimConsumerTilesArr.push_back(col);
          info.shimConsumerTilesArr.push_back(row);
        } else {
          info.consumerTilesArr.push_back(col);
          info.consumerTilesArr.push_back(row);
        }
      }
      // Depth — producer depth is always index 0.
      info.depth = op.size(0);

      // P2-9: Per-consumer depths.  When elemNumber is an ArrayAttr,
      // indices 1..N are per-consumer depths (one per consumer tile).
      if (auto arrAttr = mlir::dyn_cast<mlir::ArrayAttr>(op.getElemNumber())) {
        // Index 0 = producer depth; indices 1..N = consumer depths.
        for (unsigned i = 1; i < arrAttr.size(); ++i) {
          auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(arrAttr[i]);
          if (intAttr)
            info.consumerDepths.push_back(intAttr.getInt());
        }
      }
      // Element type — must be a MemRefType for window semantics.
      auto objfifoTy = mlir::cast<AIE::AIEObjectFifoType>(op.getElemType());
      mlir::Type elemTy = objfifoTy.getElementType();
      if (auto mrefTy = mlir::dyn_cast<mlir::MemRefType>(elemTy)) {
        info.elemType = mrefTy;
        info.numElems = numElemsInMemref(mrefTy);
      } else {
        // Fallback: treat as single-element i32 memref
        info.elemType = mlir::MemRefType::get(
            {1}, mlir::IntegerType::get(ctx, 32));
        info.numElems = 1;
      }

      fifoInfoMap[op.getSymNameAttr()] = std::move(info);
    });

    // Phase 1.5: detect cyclostatic (CSDF) access patterns.
    //
    // For each objectfifo, collect the sequence of acquire counts from all
    // Consume-port acquire ops that reference it (in module walk order, which
    // approximates program order within a flat core body).  If the sequence
    // has more than one distinct value, it is a cyclostatic pattern and we
    // record it in fifoInfoMap[name].accessPattern.
    //
    // Limitation: acquires inside scf.for loops are visited multiple times by
    // module.walk, but each unique acquire op appears exactly once.  For the
    // corpus files (acquire 1 / acquire 2 / acquire 1 as three separate ops)
    // this gives the correct pattern [1, 2, 1].  Acquire ops inside loops that
    // all use the same count do not create a cyclostatic pattern.

    // Per-fifo: ordered list of (Consume) acquire counts seen in program order.
    //
    // Fix 4p: Note: consumeAcquireCounts accumulates counts across ALL cores
    // for each fifo name via module.walk. For multi-consumer fifos, this
    // merges patterns from different cores in DFS walk order, which does not
    // correspond to any single core's CSDF pattern.
    llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<int64_t>>
        consumeAcquireCounts;

    module.walk([&](AIE::ObjectFifoAcquireOp op) {
      // Only Consume-port acquires determine the consumer access pattern.
      if (op.getPort() != AIE::ObjectFifoPort::Consume)
        return;
      auto nameAttr = mlir::StringAttr::get(ctx, op.getObjFifoName().str());
      // Skip aie_stream fifos — not lowered by Pass A.
      if (aieStreamFifoNames.count(nameAttr))
        return;
      consumeAcquireCounts[nameAttr].push_back(op.acqNumber());
    });

    // For each fifo, if the counts are not all equal, record the pattern.
    for (auto &[nameAttr, counts] : consumeAcquireCounts) {
      if (counts.empty())
        continue;
      // Check uniformity.
      bool uniform = llvm::all_of(counts, [&](int64_t c) { return c == counts[0]; });
      if (!uniform) {
        auto it = fifoInfoMap.find(nameAttr);
        if (it != fifoInfoMap.end())
          it->second.accessPattern = counts;
      }
    }

    // P2-D: erase aie.objectfifo.register_process ops.
    // register_process is a code-generation macro that the dedicated pre-pass
    // (--aie-register-objectFifos) expands into standard acquire/release loops
    // before the stateful transform runs. When that pre-pass has already run,
    // these ops are already gone. When it has not run, silently erasing them
    // is safe: the op has zero presence in the stateful-transform corpus (all
    // 130 files in test/objectFifo-stateful-transform/ use raw acquire/release,
    // not register_process) and the stateful transform itself ignores them.
    // Users who need register_process expansion must run
    // --aie-register-objectFifos before --objectfifo-to-conduit.
    llvm::SmallVector<AIE::ObjectFifoRegisterProcessOp> regProcOps;
    module.walk([&](AIE::ObjectFifoRegisterProcessOp op) {
      regProcOps.push_back(op);
    });
    for (auto op : regProcOps)
      op.erase();
  }

  // -----------------------------------------------------------------------
  // Phase 2–4: transformFifos
  // -----------------------------------------------------------------------
  //
  // Emit conduit.create / conduit.link / conduit.acquire /
  // conduit.subview_access / conduit.release, replacing all ObjectFIFO ops.
  // Original ops are collected in erasure vectors for later cleanup.

  void transformFifos(mlir::ModuleOp module, mlir::OpBuilder &builder,
                      mlir::MLIRContext *ctx) {
    fifosToErase.clear();
    subviewsToErase.clear();
    acquiresToErase.clear();
    releasesToErase.clear();

    // Phase 2: rewrite each aie.objectfifo → conduit.create with typed attrs.
    // NOTE: do NOT erase the objectfifo op here — the AIE verifier requires
    // aie.objectfifo.acquire to reference a live objectfifo symbol.  Collect
    // for deferred erasure after Phase 4 completes.

    module.walk([&](AIE::ObjectFifoCreateOp op) {
      // Skip aie_stream fifos — left intact for stateful transform.
      if (aieStreamFifoNames.count(op.getSymNameAttr()))
        return;

      builder.setInsertionPoint(op);
      mlir::Location loc = op.getLoc();

      auto &info = fifoInfoMap[op.getSymNameAttr()];
      std::string name = op.getSymName().str();
      int64_t capacity = info.depth * info.numElems;

      // conduit.create with typed attributes — no conduit.annotate ops.
      // shim_consumer_tiles carries shim (row==0) consumer tiles separately;
      // they are DMA endpoints handled via shim_dma_allocation in Pass C.
      mlir::DenseI64ArrayAttr shimConsAttr;
      if (!info.shimConsumerTilesArr.empty())
        shimConsAttr = mlir::DenseI64ArrayAttr::get(ctx, info.shimConsumerTilesArr);

      // Emit access_pattern attribute for cyclostatic (CSDF) fifos.
      // When the consumer acquires varying counts per iteration, the pattern
      // is stored here so Pass C can generate the correct lock protocol.
      mlir::DenseI64ArrayAttr accessPatternAttr;
      if (!info.accessPattern.empty())
        accessPatternAttr =
            mlir::DenseI64ArrayAttr::get(ctx, info.accessPattern);

      // Extract repeat_count from the source objectfifo, if present.
      // Propagated unconditionally (any value including >1) so Pass C can set
      // DMAStartOp repeat_count accordingly.
      mlir::IntegerAttr repeatCountAttr;
      if (op.getRepeatCount().has_value()) {
        repeatCountAttr = mlir::IntegerAttr::get(
            mlir::IntegerType::get(ctx, 64),
            static_cast<int64_t>(op.getRepeatCount().value()));
      }

      // P2-9: Build consumer_depths attribute if per-consumer depths differ.
      mlir::DenseI64ArrayAttr consumerDepthsAttr;
      if (!info.consumerDepths.empty())
        consumerDepthsAttr =
            mlir::DenseI64ArrayAttr::get(ctx, info.consumerDepths);

      // Propagate disable_synchronization.
      mlir::BoolAttr disableSyncAttr;
      if (op.getDisableSynchronization())
        disableSyncAttr = mlir::BoolAttr::get(ctx, true);

      // Propagate iter_count.
      mlir::IntegerAttr iterCountAttr;
      if (op.getIterCount().has_value()) {
        iterCountAttr = mlir::IntegerAttr::get(
            mlir::IntegerType::get(ctx, 64),
            static_cast<int64_t>(op.getIterCount().value()));
      }

      // Propagate dimensionsToStream (producer side).
      // Stored as generic mlir::Attribute to avoid cross-dialect tablegen dep.
      mlir::Attribute prodDimsAttr;
      {
        auto dims = op.getDimensionsToStreamAttr();
        if (dims && !dims.getValue().empty())
          prodDimsAttr = dims;
      }

      // Propagate dimensionsFromStreamPerConsumer (per-consumer).
      // Check that at least one consumer has non-empty dims (the outer array
      // is always non-empty even for fifos with no dims — it has one [] per
      // consumer — so we must check the inner arrays).
      mlir::Attribute consDimsAttr;
      {
        auto dims = op.getDimensionsFromStreamPerConsumerAttr();
        bool hasNonEmptyDims = false;
        if (dims) {
          for (auto consArr : dims.getValue()) {
            if (!consArr.getValue().empty()) {
              hasNonEmptyDims = true;
              break;
            }
          }
        }
        if (hasNonEmptyDims)
          consDimsAttr = dims;
      }

      // Propagate via_DMA.
      // Auto-set via_DMA=true when:
      //   (a) dimensionsToStream or dimensionsFromStream are non-empty: the
      //       shared-memory path skips DMA BDs entirely, silently dropping N-D
      //       transforms. Forcing DMA ensures BDDimLayout attributes are applied
      //       at the hardware level.
      //   (b) repeat_count > 1: the BD chain is replayed N times by the DMA
      //       engine. Shared-memory has no BD replay mechanism — the hardware
      //       lock protocol would need the core to re-acquire N times, but with
      //       no consumer core body (the common repeat_count pattern) no one
      //       drives the lock. Forcing DMA ensures the BD chain is emitted and
      //       the repeat_count is applied via DMAStartOp.
      mlir::BoolAttr viaDMAAttr;
      bool hasRepeat = op.getRepeatCount().has_value() &&
                       op.getRepeatCount().value() > 1;
      if (op.getVia_DMA() || prodDimsAttr || consDimsAttr || hasRepeat)
        viaDMAAttr = mlir::BoolAttr::get(ctx, true);

      // Propagate via_cascade → routing_mode = "cascade".
      // Cascade has no hardware FIFO; depth must be 1.
      mlir::StringAttr routingModeAttr;
      if (op.getViaCascade()) {
        if (info.depth != 1) {
          op.emitError(
              "objectfifo-to-conduit: via_cascade=true requires depth=1 "
              "(cascade has no hardware FIFO buffering), got depth=")
              << info.depth;
          signalPassFailure();
          passFailed = true;
          return; // skip conduit.create for this fifo
        }
        // CSDF patterns require buffering that the cascade stream cannot
        // provide. Cascade is a single-register pass-through with no queue;
        // any rate other than 1:1 produces deadlock or data corruption.
        // Detect CSDF by checking whether Phase 1.5 observed varying acquire
        // counts (non-empty accessPattern means at least two distinct counts).
        if (!info.accessPattern.empty()) {
          op.emitError(
              "cascade conduit requires SDF rate (1,1); CSDF patterns "
              "require buffering which cascade cannot provide");
          signalPassFailure();
          passFailed = true;
          return; // skip conduit.create for this fifo
        }
        routingModeAttr = mlir::StringAttr::get(ctx, "cascade");
      }

      builder.create<Create>(
          loc,
          mlir::StringAttr::get(ctx, name),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), capacity),
          mlir::DenseI64ArrayAttr::get(ctx, info.producerTileArr),
          mlir::DenseI64ArrayAttr::get(ctx, info.consumerTilesArr),
          shimConsAttr,
          mlir::TypeAttr::get(info.elemType),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), info.depth),
          /*link_mode=*/mlir::StringAttr{},
          accessPatternAttr,
          routingModeAttr,
          /*producer_rates=*/mlir::DenseI64ArrayAttr{},
          /*consumer_rates=*/mlir::DenseI64ArrayAttr{},
          /*alloc_tile=*/mlir::DenseI64ArrayAttr{},
          repeatCountAttr,
          consumerDepthsAttr,
          disableSyncAttr,
          viaDMAAttr,
          iterCountAttr,
          prodDimsAttr,
          consDimsAttr);

      fifosToErase.push_back(op);
    });

    // Phase 3: rewrite aie.objectfifo.link → conduit.link.
    module.walk([&](AIE::ObjectFifoLinkOp op) {
      builder.setInsertionPoint(op);
      mlir::Location loc = op.getLoc();

      auto fifoIns = op.getFifoIns();
      auto fifoOuts = op.getFifoOuts();

      // Determine mode
      std::string mode;
      if (fifoIns.size() == 1 && fifoOuts.size() >= 1)
        mode = "distribute";
      else if (fifoIns.size() >= 1 && fifoOuts.size() == 1)
        mode = "join";
      else {
        // Fix 4h: N→M link (N>1 sources AND N>1 destinations) is not
        // supported. Emit an error rather than silently using "distribute".
        op.emitError(
            "objectfifo-to-conduit: N→M link (N>1 sources AND N>1 "
            "destinations) is not supported; use cascade mode when implemented");
        signalPassFailure();
        passFailed = true;
        return;
      }

      // Build src/dst name arrays
      llvm::SmallVector<mlir::Attribute> srcAttrs, dstAttrs;
      for (auto sym : fifoIns) {
        auto flat = mlir::cast<mlir::FlatSymbolRefAttr>(sym);
        srcAttrs.push_back(mlir::StringAttr::get(ctx, flat.getValue()));
      }
      for (auto sym : fifoOuts) {
        auto flat = mlir::cast<mlir::FlatSymbolRefAttr>(sym);
        dstAttrs.push_back(mlir::StringAttr::get(ctx, flat.getValue()));
      }

      // Relay tile detection: find the tile that sits between the source and
      // destination fifos in the link.  By definition, the relay tile is the
      // PRODUCER of the destination fifo (it receives data from upstream and
      // forwards it downstream).
      //
      // The old heuristic ("consumer of first src fifo") fails for broadcast
      // links where the source fifo has multiple consumers — it picks the
      // first consumer in array order, which may be a compute tile rather
      // than the actual MemTile relay.
      //
      // New logic: use the producer tile of the first dst fifo.  This is
      // always correct because in an objectfifo.link, each dst fifo's
      // producer IS the relay tile.  Falls back to consumer of first src
      // if no dst fifo info is available.
      std::string memtileStr = "unknown";
      bool found = false;

      // Primary: producer of the first dst fifo.
      if (!fifoOuts.empty()) {
        auto firstDstSym = mlir::cast<mlir::FlatSymbolRefAttr>(fifoOuts[0]);
        auto nameAttr = mlir::StringAttr::get(ctx, firstDstSym.getValue());
        auto it = fifoInfoMap.find(nameAttr);
        if (it != fifoInfoMap.end() &&
            it->second.producerTileArr.size() >= 2) {
          int64_t col = it->second.producerTileArr[0];
          int64_t row = it->second.producerTileArr[1];
          std::string s;
          llvm::raw_string_ostream os(s);
          os << "tile(" << col << "," << row << ")";
          memtileStr = os.str();
          found = true;
        }
      }

      // Fallback: consumer of first src fifo (original heuristic).
      if (!found && !fifoIns.empty()) {
        auto firstSym = mlir::cast<mlir::FlatSymbolRefAttr>(fifoIns[0]);
        auto nameAttr = mlir::StringAttr::get(ctx, firstSym.getValue());
        auto it = fifoInfoMap.find(nameAttr);
        if (it != fifoInfoMap.end() &&
            it->second.consumerTilesArr.size() >= 2) {
          int64_t col = it->second.consumerTilesArr[0];
          int64_t row = it->second.consumerTilesArr[1];
          std::string s;
          llvm::raw_string_ostream os(s);
          os << "tile(" << col << "," << row << ")";
          memtileStr = os.str();
        }
      }

      // Extract offsets from the link op.
      mlir::DenseI64ArrayAttr offsetsAttr;
      auto joinOffsets = op.getSrcOffsets();
      auto distOffsets = op.getDstOffsets();

      llvm::SmallVector<int64_t> offVec;
      if (mode == "join" && joinOffsets && !joinOffsets.empty()) {
        for (auto attr : joinOffsets)
          offVec.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
      } else if (mode == "distribute" && distOffsets && !distOffsets.empty()) {
        for (auto attr : distOffsets)
          offVec.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
      }
      if (!offVec.empty())
        offsetsAttr = mlir::DenseI64ArrayAttr::get(ctx, offVec);

      builder.create<Link>(
          loc, mlir::ArrayAttr::get(ctx, srcAttrs),
          mlir::ArrayAttr::get(ctx, dstAttrs),
          mlir::StringAttr::get(ctx, mode),
          mlir::StringAttr::get(ctx, memtileStr),
          offsetsAttr,
          /*lock_id=*/nullptr);

      op.erase();
    });

    // Phase 4: rewrite acquire/release/subview.access inside core bodies.
    //
    // SSA connectivity fix (Fix 1):
    //   conduit.acquire returns !conduit.window<T>.  This SSA value is passed
    //   directly to conduit.subview_access as the window operand.  All uses of
    //   the aie.objectfifo.subview.access result are replaced with the
    //   conduit.subview_access result.  No memref.alloc placeholder is needed.
    //
    // Port propagation fix (Fix 2):
    //   port="Produce"|"Consume" is read from the AIE op and forwarded.
    //
    // Use-after-erase fix:
    //   All subview.access ops are collected for deferred erasure BEFORE the
    //   acquire op is erased.  This prevents the acquire result SSA value from
    //   being invalidated while we still need it for the subview rewrite.

    // Collect cascade fifo names for Phase 4 dispatch.
    llvm::DenseSet<mlir::StringAttr> cascadeFifoNames;
    module.walk([&](AIE::ObjectFifoCreateOp op) {
      if (op.getViaCascade())
        cascadeFifoNames.insert(op.getSymNameAttr());
    });

    // Per-block window maps: block → (fifo name → window SSA value).
    // Populated as each block is visited; used for cross-block lookups.
    //
    // Domination invariant: conduit.release takes a !conduit.window<T> SSA
    // operand that must dominate the release site.  A flat module.walk over
    // all acquires followed by a flat walk over all releases breaks this
    // because the global `lastWindowForName` map gets clobbered by acquires
    // from later blocks.
    //
    // Fix: process every basic block independently.  Within each block the ops
    // are visited in program order, so the acquire always precedes its release
    // and the window SSA value is live at the release site.
    //
    // Cross-block dominance fix (C1):
    //   When a release is in block B but its acquire is in a dominating parent
    //   block A (e.g., release inside scf.if true region, acquire in the
    //   enclosing core entry block), the local blockWindowMap for B will be
    //   empty.  Before synthesizing a phantom acquire and emitting the C1
    //   warning, we walk up the region/block parent chain to see if any
    //   enclosing block already has a window value for this conduit name.
    //   If found, we reuse that dominating window SSA value directly — no
    //   phantom, no warning.  If not found (truly unreachable acquire), we
    //   fall back to the phantom + C1 warning as before.
    llvm::DenseMap<mlir::Block *, llvm::DenseMap<mlir::StringAttr, mlir::Value>>
        allBlockWindowMaps;

    // Helper: walk the region/block parent chain from `startBlock` upward,
    // returning the first window value found for `nameAttr`, or null if none.
    // This handles the common case where the release is inside a nested region
    // (scf.if, scf.for body) and the acquire is in an enclosing block.
    //
    // Fix 4j: Note: if acquire and release are in the same block but release
    // appears BEFORE acquire in program order, the local blockWindowMap lookup
    // returns null and the dominating-block walk finds nothing, emitting a
    // spurious C1 warning. Forward-declared-then-released patterns trigger
    // false positives.
    auto findWindowInDominatingBlock =
        [&](mlir::Block *startBlock,
            mlir::StringAttr nameAttr) -> mlir::Value {
      mlir::Block *cursor = startBlock;
      while (cursor) {
        // Check if this block's window map has an entry for the conduit.
        auto mapIt = allBlockWindowMaps.find(cursor);
        if (mapIt != allBlockWindowMaps.end()) {
          mlir::Value v = mapIt->second.lookup(nameAttr);
          if (v)
            return v;
        }
        // Walk up: the enclosing block is the block that contains
        // cursor's parent region's parent op.
        mlir::Operation *parentOp = cursor->getParentOp();
        if (!parentOp)
          break;
        cursor = parentOp->getBlock();
      }
      return {};
    };

    // Use PreOrder so parent blocks are visited before their nested child blocks.
    // This ensures that when the scf.if body block is visited, the enclosing
    // core entry block's window map is already populated — enabling the
    // parent-block walk in findWindowInDominatingBlock to succeed.
    module.walk<mlir::WalkOrder::PreOrder>([&](mlir::Block *block) {
      // Per-block window map: fifo name → window SSA value emitted by the
      // most recently processed acquire in this block.
      llvm::DenseMap<mlir::StringAttr, mlir::Value> &blockWindowMap =
          allBlockWindowMaps[block];

      // Sequential acquire pattern (P2-D: AIE2_delayed_release):
      //
      // ObjectFIFO acquire semantics: acquire(N) means "I need N total
      // elements right now", not "give me N more".  If you call acquire(2)
      // then acquire(1) then acquire(3) then release(3), the stateful
      // transform emits: AcquireGreaterEqual(2), [nothing], AcquireGreaterEqual(1),
      // [nothing], Release(3) — only the incremental delta is acquired each step.
      //
      // The Conduit IR window model is acquire-one-window-at-a-time.  To match
      // the oracle, Pass A collapses each release-group of consecutive acquires
      // into a SINGLE conduit.acquire with the MAXIMUM count in the group, then
      // reuses that one window for all subview_access rewrites within the group.
      //
      // Pre-scan: for each block, walk its ops in program order and group
      // consecutive acquires on the same (fifo, port) pair between releases.
      // Record (acquire_op → effective_max_count_for_group).  When an acquire's
      // count does not exceed the current group max, it is "subsumed" by the
      // first acquire in the group and will not emit a new conduit.acquire.
      //
      // Only applies to non-cascade fifos (cascade has no window semantics).
      //
      // Example:  acquire(2), acquire(1), acquire(3), acquire(1), release(3)
      //   group max = 3 (from acquire(3))
      //   first acquire in group → conduit.acquire{count=3}
      //   remaining acquires → suppressed; reuse group window
      //   release(3) → conduit.release %groupWin {count=3}  [M8: 3≤3 OK]

      // Key: (fifo-name-attr, port-enum-as-int64)
      using GroupKey = std::pair<mlir::StringAttr, int64_t>;
      // Map from acquire-op ptr to the effective max count for its group.
      llvm::DenseMap<mlir::Operation *, int64_t> acqGroupMax;
      // Map from acquire-op ptr to whether it is the first (non-suppressed)
      // acquire in its group — only the first emits conduit.acquire.
      llvm::DenseMap<mlir::Operation *, bool> acqIsGroupLeader;

      {
        // current max held and the group leader op for each (name, port).
        llvm::DenseMap<GroupKey, int64_t> heldMax;
        llvm::DenseMap<GroupKey, mlir::Operation *> groupLeader;

        for (mlir::Operation &rawOp : *block) {
          if (auto acqOp = mlir::dyn_cast<AIE::ObjectFifoAcquireOp>(rawOp)) {
            auto nameAttr =
                mlir::StringAttr::get(ctx, acqOp.getObjFifoName().str());
            // Skip cascade fifos — they have no window semantics.
            if (cascadeFifoNames.count(nameAttr))
              continue;
            int64_t portInt =
                (acqOp.getPort() == AIE::ObjectFifoPort::Produce) ? 0 : 1;
            GroupKey key = {nameAttr, portInt};
            int64_t newCount = acqOp.acqNumber();
            int64_t &held = heldMax[key];
            if (newCount > held) {
              // This acquire extends the group (or starts a new one if held==0).
              if (held == 0) {
                // New group: this op is the leader.
                groupLeader[key] = &rawOp;
                acqIsGroupLeader[&rawOp] = true;
              } else {
                // Extends existing group: update the current leader's effective
                // max so Pass C generates the right count.
                mlir::Operation *leader = groupLeader[key];
                acqGroupMax[leader] =
                    std::max(acqGroupMax.count(leader) ? acqGroupMax[leader]
                                                       : heldMax[key],
                             newCount);
                acqIsGroupLeader[&rawOp] = false;
              }
              held = newCount;
            } else {
              // Sub-max acquire: suppressed; reuse the current group leader.
              acqIsGroupLeader[&rawOp] = false;
            }
          } else if (auto relOp =
                         mlir::dyn_cast<AIE::ObjectFifoReleaseOp>(rawOp)) {
            auto nameAttr =
                mlir::StringAttr::get(ctx, relOp.getObjFifoName().str());
            if (cascadeFifoNames.count(nameAttr))
              continue;
            int64_t portInt =
                (relOp.getPort() == AIE::ObjectFifoPort::Produce) ? 0 : 1;
            GroupKey key = {nameAttr, portInt};
            // Finalise group leader's effective max (covers the case where the
            // leader was never updated by an extending acquire).
            if (groupLeader.count(key)) {
              mlir::Operation *leader = groupLeader[key];
              if (!acqGroupMax.count(leader))
                acqGroupMax[leader] = heldMax[key];
            }
            // Reset group tracking.
            heldMax[key] = 0;
            groupLeader.erase(key);
          }
        }
        // Finalise any open groups at end of block (no trailing release).
        for (auto &[key, leader] : groupLeader) {
          if (!acqGroupMax.count(leader))
            acqGroupMax[leader] = heldMax[key];
        }
      } // end pre-scan

      // Per-block group-window map: fifo name → current group leader's
      // conduit.acquire SSA value.  Used to reuse the group window for
      // suppressed (sub-max) acquires and for releases.
      llvm::DenseMap<mlir::StringAttr, mlir::Value> blockGroupWindow;
      // Per-block held count: tracks how many elements are currently held
      // so we know when to update the group window after an extending acquire.
      llvm::DenseMap<mlir::StringAttr, int64_t> blockHeldCount;

      for (mlir::Operation &rawOp : llvm::make_early_inc_range(*block)) {
        if (auto op = mlir::dyn_cast<AIE::ObjectFifoAcquireOp>(rawOp)) {
          builder.setInsertionPoint(op);
          mlir::Location loc = op.getLoc();

          std::string name = op.getObjFifoName().str();
          int64_t count = op.acqNumber();
          Port port = (op.getPort() == AIE::ObjectFifoPort::Produce)
                          ? Port::Produce
                          : Port::Consume;

          auto nameAttr = mlir::StringAttr::get(ctx, name);

          // Skip acquire ops for aie_stream fifos — left intact for stateful
          // transform. Do not emit conduit.acquire or touch these ops.
          if (aieStreamFifoNames.count(nameAttr))
            continue;

          mlir::MemRefType elemType;
          auto it = fifoInfoMap.find(nameAttr);
          if (it != fifoInfoMap.end())
            elemType = it->second.elemType;
          if (!elemType)
            elemType = mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));

          // Cascade path: two sub-cases.
          //
          // Consume: emit get_cascade with the memref's element type (scalar),
          //   find all memref.load users of %elem0 and replace their results with
          //   the get value, then erase the loads.  The subview and acquire are
          //   collected for deferred erasure in the normal order.
          //
          // Produce: do NOT process at acquire time — the stored value isn't
          //   available until the user's memref.store runs.  The release handler
          //   below walks the produce acquire, finds the subview, finds the store,
          //   extracts the stored value, emits put_cascade, then erases the store,
          //   subview, and acquire in dependency order (store → subview → acquire).
          //   The acquire is NOT added to acquiresToErase here; the release handler
          //   takes ownership of erasure.
          if (cascadeFifoNames.count(nameAttr)) {
            if (port == Port::Consume) {
              // Scalar element type (e.g., i32 from memref<1xi32>).
              mlir::Type elemTy = elemType.getElementType();
              auto getCascOp = builder.create<GetCascade>(
                  loc, elemTy, mlir::StringAttr::get(ctx, name));
              mlir::Value cascVal = getCascOp.getValue(); // scalar i32/vector

              mlir::Value subviewResult = op.getResult();
              for (mlir::Operation *user :
                   llvm::make_early_inc_range(subviewResult.getUsers())) {
                if (auto accessOp =
                        mlir::dyn_cast<AIE::ObjectFifoSubviewAccessOp>(user)) {
                  // Replace memref.load users of %elem0 with the cascade value.
                  mlir::Value elem0 = accessOp.getResult(); // memref<1xi32>
                  for (mlir::Operation *loadUser :
                       llvm::make_early_inc_range(elem0.getUsers())) {
                    if (auto loadOp =
                            mlir::dyn_cast<mlir::memref::LoadOp>(loadUser)) {
                      loadOp.getResult().replaceAllUsesWith(cascVal);
                      loadOp.erase();
                    }
                  }
                  subviewsToErase.push_back(accessOp);
                }
              }
              acquiresToErase.push_back(op);
            }
            // Produce: skip acquire; release handler does everything.
            continue;
          }

          // Sequential acquire pattern (P2-D):
          // The pre-scan determined whether this acquire is a group leader
          // (first/max in its release-group) or is subsumed by the leader.
          //
          // - Group leader: emit conduit.acquire{count=effective_max}, record
          //   as blockGroupWindow.  The count used is the pre-scan max (not the
          //   raw op count) so that sub-max acquires in the same group are
          //   covered by this single window — M8 sees release_count ≤ max.
          // - Subsumed acquire: no conduit.acquire emitted; subview_access ops
          //   are rewritten to use the current blockGroupWindow instead.
          //
          // Cross-block subsumption (AIE2_dynamic_locks pattern):
          //   When an acquire in a nested block (e.g., scf.for body) requests
          //   the same or fewer elements than an open acquire in a dominating
          //   parent block, the nested acquire is subsumed — no new use_lock
          //   is needed because the elements are already held.  Matches the
          //   stateful transform which tracks held counts across block boundaries.

          mlir::Value winVal;
          bool isLeader = acqIsGroupLeader.lookup(&rawOp);
          if (isLeader || !blockGroupWindow.count(nameAttr)) {
            // Before emitting a new conduit.acquire, check if a dominating
            // parent block already holds a window for this fifo.  If so, the
            // acquire in the enclosing scope already covers the needed count
            // and no additional use_lock is required (cross-block subsumption).
            mlir::Value parentWin = findWindowInDominatingBlock(
                block->getParentOp() ? block->getParentOp()->getBlock()
                                     : nullptr,
                nameAttr);
            if (parentWin) {
              // Reuse the dominating block's window — cross-block subsumed.
              winVal = parentWin;
              blockGroupWindow[nameAttr] = winVal;
            } else {
              // Emit one conduit.acquire for the group.
              // Use the pre-scanned group max as the effective count so the
              // single window covers all elements that will be released.
              int64_t effectiveCount = acqGroupMax.count(&rawOp)
                                           ? acqGroupMax[&rawOp]
                                           : count;
              auto winTy = WindowType::get(ctx, elemType);
              winVal = builder.create<Acquire>(
                  loc, winTy, mlir::StringAttr::get(ctx, name),
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                         effectiveCount),
                  PortAttr::get(ctx, port));
              // Record as the group leader window.
              blockGroupWindow[nameAttr] = winVal;
            }
          } else {
            // Subsumed acquire: reuse the existing group window.
            winVal = blockGroupWindow[nameAttr];
          }

          // Record the window for subsequent releases in this block and for
          // cross-block lookups in dominated nested blocks.
          blockWindowMap[nameAttr] = winVal;

          // Rewrite subview.access users immediately.
          mlir::Value subviewResult = op.getResult();
          for (mlir::Operation *user :
               llvm::make_early_inc_range(subviewResult.getUsers())) {
            if (auto accessOp =
                    mlir::dyn_cast<AIE::ObjectFifoSubviewAccessOp>(user)) {
              builder.setInsertionPoint(accessOp);
              int64_t idx = accessOp.getIndex();
              mlir::Type resultTy = accessOp.getResult().getType();
              auto condAccess = builder.create<SubviewAccess>(
                  accessOp.getLoc(), resultTy, winVal,
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), idx));
              accessOp.getResult().replaceAllUsesWith(condAccess.getResult());
              subviewsToErase.push_back(accessOp);
            } else {
              // Fix B2: unexpected user of objectfifo.acquire result.
              // Only subview_access users are lowered; other user types will
              // have dangling references after the acquire op is erased.
              op.emitWarning(
                  "objectfifo-to-conduit: unexpected user of "
                  "objectfifo.acquire result — only subview_access users are "
                  "lowered; other users will have dangling references");
            }
          }
          acquiresToErase.push_back(op);

        } else if (auto op = mlir::dyn_cast<AIE::ObjectFifoReleaseOp>(rawOp)) {
          builder.setInsertionPoint(op);
          mlir::Location loc = op.getLoc();

          std::string name = op.getObjFifoName().str();
          int64_t count = op.getSize();
          Port port = (op.getPort() == AIE::ObjectFifoPort::Produce)
                          ? Port::Produce
                          : Port::Consume;

          auto nameAttr = mlir::StringAttr::get(ctx, name);

          // Skip release ops for aie_stream fifos — left intact for stateful
          // transform.
          if (aieStreamFifoNames.count(nameAttr))
            continue;

          // Cascade release handling.
          if (cascadeFifoNames.count(nameAttr)) {
            if (port == Port::Produce) {
              // Find the corresponding acquire for this fifo in this block
              // (scanning forward is safe since we process in PreOrder).
              // The acquire was NOT added to acquiresToErase; we own it here.
              AIE::ObjectFifoAcquireOp acqOp;
              for (mlir::Operation &scan : *block) {
                if (auto a = mlir::dyn_cast<AIE::ObjectFifoAcquireOp>(scan)) {
                  if (a.getObjFifoName() == name &&
                      a.getPort() == AIE::ObjectFifoPort::Produce)
                    acqOp = a;
                }
              }

              if (!acqOp) {
                op->emitWarning(
                    "objectfifo-to-conduit: cascade Produce release for '")
                    << name << "' has no matching acquire — put_cascade skipped";
                releasesToErase.push_back(op);
                continue;
              }

              // Find the subview.access op (user of the acquire result).
              AIE::ObjectFifoSubviewAccessOp accessOp;
              for (mlir::Operation *user : acqOp.getResult().getUsers()) {
                if (auto a = mlir::dyn_cast<AIE::ObjectFifoSubviewAccessOp>(user))
                  accessOp = a;
              }

              if (!accessOp) {
                op->emitWarning(
                    "objectfifo-to-conduit: cascade Produce acquire for '")
                    << name << "' has no subview.access user — put_cascade skipped";
                releasesToErase.push_back(op);
                acqOp->erase();
                continue;
              }

              // Find the memref.store into elem0 and extract the stored value.
              // The stored value becomes the cascade stream value.
              mlir::Value elem0 = accessOp.getResult(); // memref<T>
              mlir::Value storedVal;
              llvm::SmallVector<mlir::memref::StoreOp> storesToErase;
              for (mlir::Operation *storeUser :
                   llvm::make_early_inc_range(elem0.getUsers())) {
                if (auto storeOp =
                        mlir::dyn_cast<mlir::memref::StoreOp>(storeUser)) {
                  storedVal = storeOp.getValueToStore();
                  storesToErase.push_back(storeOp);
                }
              }

              if (storedVal) {
                builder.setInsertionPoint(op);
                builder.create<PutCascade>(
                    loc, mlir::StringAttr::get(ctx, name), storedVal);
              } else {
                op->emitWarning(
                    "objectfifo-to-conduit: cascade Produce '")
                    << name << "' has no memref.store into elem0 — "
                       "put_cascade skipped (no value to send)";
              }

              // Erase in dependency order: stores → subview → acquire.
              for (auto s : storesToErase)
                s.erase();
              if (accessOp.getResult().use_empty())
                accessOp.erase();
              if (acqOp.getResult().use_empty())
                acqOp.erase();

              releasesToErase.push_back(op);
              continue;
            }

            // Cascade Consume releases are no-ops (no hardware lock to release).
            releasesToErase.push_back(op);
            continue;
          }

          // Sequential acquire pattern (P2-D): for releases in the same block,
          // prefer the group leader window (blockGroupWindow) over the most
          // recently seen acquire window (blockWindowMap).  The group leader
          // window was emitted with count=max_in_group, so M8's invariant
          // release_count ≤ acquired_count is satisfied.
          //
          // After the release, clear the group window so the next acquire in
          // this block starts a new group.
          mlir::Value winVal = blockGroupWindow.lookup(nameAttr);
          if (winVal) {
            // Same-block sequential acquire group: use group leader window.
            blockGroupWindow.erase(nameAttr);
            blockHeldCount.erase(nameAttr);
          } else {
            // No group window in this block — try blockWindowMap (same block).
            winVal = blockWindowMap.lookup(nameAttr);
          }

          if (!winVal) {
            // Cross-block case: look for a window value in a dominating
            // enclosing block (e.g., acquire in core entry block, release
            // inside scf.if true region).
            winVal = findWindowInDominatingBlock(
                block->getParentOp() ? block->getParentOp()->getBlock()
                                     : nullptr,
                nameAttr);
          }

          if (!winVal) {
            // C1 diagnostic: no dominating acquire found anywhere in the
            // parent block chain.  Must synthesize a phantom window.
            // On Phoenix hardware this injects an extra AcquireGreaterEqual
            // use_lock that stalls the producer.  Manual review required.
            op->emitWarning(
                "ObjectFifoToConduit: cross-block acquire/release pattern "
                "detected for conduit '")
                << name
                << "'; phantom AcquireGreaterEqual synthesized. Hardware stall "
                   "risk on Phoenix. Manual review required.";

            mlir::MemRefType elemType;
            auto it = fifoInfoMap.find(nameAttr);
            if (it != fifoInfoMap.end())
              elemType = it->second.elemType;
            if (!elemType)
              elemType =
                  mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));
            auto winTy = WindowType::get(ctx, elemType);
            winVal = builder.create<Acquire>(
                loc, winTy, mlir::StringAttr::get(ctx, name),
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), count),
                PortAttr::get(ctx, port));
            blockWindowMap[nameAttr] = winVal;
          }

          builder.create<Release>(
              loc, winVal,
              mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), count),
              PortAttr::get(ctx, port));

          releasesToErase.push_back(op);
        }
      }
    });
  }

  // -----------------------------------------------------------------------
  // Phase 4.5+: eraseOriginalOps
  // -----------------------------------------------------------------------
  //
  // Erase all original ObjectFIFO ops (subview, acquire, release, create)
  // and emit shim DMA allocation symbols + allocate delegate tile transfer.

  void eraseOriginalOps(mlir::ModuleOp module, mlir::OpBuilder &builder,
                        mlir::MLIRContext *ctx) {
    // Deferred erasure: erase subviews before acquires (subview uses acquire result).
    for (auto op : subviewsToErase)
      op.erase();
    for (auto op : acquiresToErase)
      op.erase();
    for (auto op : releasesToErase)
      op.erase();

    // Lower aie.objectfifo.register_external_buffers →
    // conduit.register_external_buffers.
    //
    // ORDERING: must run BEFORE Phase 4.5's replaceAllSymbolUses() because
    // that rewrite changes @fifo_name → @fifo_name_shim_alloc in all
    // FlatSymbolRefAttr references (including the register_external_buffers
    // op's objFifo_name attribute).  We need the original name to match the
    // conduit.create emitted in Phase 2.
    //
    // Collects ops first to avoid walk-while-erase.
    llvm::SmallVector<AIE::ObjectFifoRegisterExternalBuffersOp> extBufOps;
    module.walk([&](AIE::ObjectFifoRegisterExternalBuffersOp op) {
      extBufOps.push_back(op);
    });
    for (auto extBufOp : extBufOps) {
      builder.setInsertionPoint(extBufOp);

      // Extract conduit name from the objectfifo symbol reference.
      std::string name = extBufOp.getObjFifoName().str();

      // Extract shim tile coordinates.
      auto shimTile =
          mlir::cast<AIE::TileOp>(extBufOp.getTile().getDefiningOp());
      llvm::SmallVector<int64_t> tileCoord = {shimTile.getCol(),
                                               shimTile.getRow()};

      // Collect external buffer SSA values.
      llvm::SmallVector<mlir::Value> extBufs(extBufOp.getExternalBuffers());

      builder.create<RegisterExternalBuffers>(
          extBufOp.getLoc(), mlir::StringAttr::get(ctx, name),
          mlir::DenseI64ArrayAttr::get(ctx, tileCoord), extBufs);

      extBufOp.erase();
    }

    // Phase 4.5: preserve shim DMA symbols for runtime_sequence.
    //
    // When an objectfifo connects to a shim tile (row==0), the
    // runtime_sequence contains aiex.npu.dma_wait / aiex.npu.dma_memcpy_nd
    // ops that reference the objectfifo symbol.  Erasing the objectfifo
    // without providing a replacement symbol causes the verifier to fail.
    // Fix: emit aie.shim_dma_allocation @<name>_shim_alloc and rewrite all
    // symbol uses before erasing the objectfifo.
    for (AIE::ObjectFifoCreateOp op : fifosToErase) {
      auto prodTile =
          mlir::cast<AIE::TileOp>(op.getProducerTile().getDefiningOp());
      int64_t prodRow = prodTile.getRow();

      AIE::TileOp shimTile;
      AIE::DMAChannelDir channelDir;

      if (prodRow == 0) {
        shimTile = prodTile;
        channelDir = AIE::DMAChannelDir::MM2S;
      } else {
        for (mlir::Value cons : op.getConsumerTiles()) {
          auto consTile = mlir::cast<AIE::TileOp>(cons.getDefiningOp());
          if (consTile.getRow() == 0) {
            shimTile = consTile;
            channelDir = AIE::DMAChannelDir::S2MM;
            break;
          }
        }
      }

      if (!shimTile)
        continue;

      auto deviceOp = op->getParentOfType<AIE::DeviceOp>();
      if (!deviceOp)
        continue;

      std::string allocSym = op.getSymName().str() + "_shim_alloc";

      builder.setInsertionPoint(deviceOp.getBody()->getTerminator());
      builder.create<AIE::ShimDMAAllocationOp>(
          op.getLoc(), allocSym, shimTile.getResult(),
          channelDir,
          /*channel_index=*/static_cast<int64_t>(0),
          /*plio=*/false,
          /*packet=*/nullptr);

      if (mlir::failed(mlir::SymbolTable::replaceAllSymbolUses(
              op.getSymNameAttr(), builder.getStringAttr(allocSym),
              deviceOp))) {
        op.emitWarning("ObjectFifoToConduit: failed to rewrite symbol uses "
                       "for shim-connected objectfifo '")
            << op.getSymName() << "'";
      }
    }

    // Phase 4.5b: transfer objectfifo.allocate delegate tile info into the
    // alloc_tile attribute on the matching conduit.create, then erase.
    //
    // The delegate tile from objectfifo.allocate controls buffer placement
    // in the stateful transform.  We now propagate this into conduit.create's
    // alloc_tile attribute so Pass C can use it for tile selection.
    //
    // Erasure order matters: ObjectFifoAllocateOp's verifier does a
    // symbol-table lookup for the referenced ObjectFifoCreateOp.  If we
    // erase the create op first, any surviving allocate op fires the
    // verifier.  Fix: process and erase all allocate ops before erasing
    // the create ops they reference.
    //
    // Fix 3.5: Erasure order is intentional: allocate ops before create ops.
    // Intermediate state is invalid but MLIR does not re-verify within a pass.

    // Fix 4i: Build a name→conduit.create map to avoid O(n²) inner walk.
    // Previously this used a nested module.walk to find the matching
    // conduit.create for each allocate op; now we do a single pre-scan.
    llvm::DenseMap<mlir::StringAttr, Create> conduitCreateMap;
    module.walk([&](Create conduitOp) {
      auto nameAttr = mlir::StringAttr::get(ctx, conduitOp.getName().str());
      conduitCreateMap[nameAttr] = conduitOp;
    });

    llvm::SmallVector<AIE::ObjectFifoAllocateOp> allocatesToErase;
    module.walk([&](AIE::ObjectFifoAllocateOp op) {
      // Find the matching conduit.create by direct map lookup (O(1)).
      llvm::StringRef fifoName = op.getObjFifoName();
      auto nameAttr = mlir::StringAttr::get(ctx, fifoName);
      auto it = conduitCreateMap.find(nameAttr);
      if (it != conduitCreateMap.end()) {
        Create conduitOp = it->second;
        // Extract delegate tile coordinates.
        auto delegateTile =
            mlir::cast<AIE::TileOp>(op.getDelegateTile().getDefiningOp());
        int64_t col = delegateTile.getCol();
        int64_t row = delegateTile.getRow();
        llvm::SmallVector<int64_t> tileCoord = {col, row};
        conduitOp.setAllocTileAttr(
            mlir::DenseI64ArrayAttr::get(op.getContext(), tileCoord));
      }
      allocatesToErase.push_back(op);
    });
    for (AIE::ObjectFifoAllocateOp op : allocatesToErase)
      op.erase();

    for (AIE::ObjectFifoCreateOp op : fifosToErase)
      op.erase();
  }

  // -----------------------------------------------------------------------
  // runOnOperation: orchestrate the three phases
  // -----------------------------------------------------------------------

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    mlir::MLIRContext *ctx = module.getContext();

    passFailed = false;

    collectFifoInfo(module, ctx);
    if (passFailed)
      return;
    transformFifos(module, builder, ctx);
    if (passFailed)
      return;
    eraseOriginalOps(module, builder, ctx);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory + registration
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createObjectFifoToConduitPass() {
  return std::make_unique<ObjectFifoToConduitPass>();
}

} // namespace xilinx::conduit
