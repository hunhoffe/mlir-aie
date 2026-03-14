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
//      emits conduit.objectfifo_link {srcs, dsts, mode, memtile, offsets}
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
};

// ---------------------------------------------------------------------------
// Main pass
// ---------------------------------------------------------------------------

struct ObjectFifoToConduitPass
    : impl::ObjectFifoToConduitBase<ObjectFifoToConduitPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    mlir::MLIRContext *ctx = module.getContext();

    // Phase 1: collect FifoInfo for all aie.objectfifo ops.
    llvm::DenseMap<mlir::StringAttr, FifoInfo> fifoInfoMap;

    module.walk([&](AIE::ObjectFifoCreateOp op) {
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
      // Depth
      info.depth = op.size(0);
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
    llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<int64_t>>
        consumeAcquireCounts;

    module.walk([&](AIE::ObjectFifoAcquireOp op) {
      // Only Consume-port acquires determine the consumer access pattern.
      if (op.getPort() != AIE::ObjectFifoPort::Consume)
        return;
      auto nameAttr = mlir::StringAttr::get(ctx, op.getObjFifoName().str());
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

    // Phase 2: rewrite each aie.objectfifo → conduit.create with typed attrs.
    // NOTE: do NOT erase the objectfifo op here — the AIE verifier requires
    // aie.objectfifo.acquire to reference a live objectfifo symbol.  Collect
    // for deferred erasure after Phase 4 completes.

    llvm::SmallVector<AIE::ObjectFifoCreateOp> fifosToErase;

    module.walk([&](AIE::ObjectFifoCreateOp op) {
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
          /*routing_mode=*/mlir::StringAttr{},
          /*producer_rates=*/mlir::DenseI64ArrayAttr{},
          /*consumer_rates=*/mlir::DenseI64ArrayAttr{});

      fifosToErase.push_back(op);
    });

    // Phase 3: rewrite aie.objectfifo.link → conduit.objectfifo_link.
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
      else
        mode = "distribute";

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

      // Heuristic memtile: consumer tile of the first src fifo
      std::string memtileStr = "unknown";
      if (!fifoIns.empty()) {
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

      builder.create<ObjectFifoLink>(
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

    llvm::SmallVector<AIE::ObjectFifoSubviewAccessOp> subviewsToErase;
    llvm::SmallVector<AIE::ObjectFifoAcquireOp> acquiresToErase;
    llvm::SmallVector<AIE::ObjectFifoReleaseOp> releasesToErase;

    // Phase 4: rewrite acquire/release/subview.access in one pass per block.
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
    //
    // Per-block window maps: block → (fifo name → window SSA value).
    // Populated as each block is visited; used for cross-block lookups.
    llvm::DenseMap<mlir::Block *, llvm::DenseMap<mlir::StringAttr, mlir::Value>>
        allBlockWindowMaps;

    // Helper: walk the region/block parent chain from `startBlock` upward,
    // returning the first window value found for `nameAttr`, or null if none.
    // This handles the common case where the release is inside a nested region
    // (scf.if, scf.for body) and the acquire is in an enclosing block.
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

      for (mlir::Operation &rawOp : llvm::make_early_inc_range(*block)) {
        if (auto op = mlir::dyn_cast<AIE::ObjectFifoAcquireOp>(rawOp)) {
          builder.setInsertionPoint(op);
          mlir::Location loc = op.getLoc();

          std::string name = op.getObjFifoName().str();
          int64_t count = op.acqNumber();
          std::string portStr =
              (op.getPort() == AIE::ObjectFifoPort::Produce) ? "Produce"
                                                             : "Consume";

          auto nameAttr = mlir::StringAttr::get(ctx, name);
          mlir::MemRefType elemType;
          auto it = fifoInfoMap.find(nameAttr);
          if (it != fifoInfoMap.end())
            elemType = it->second.elemType;
          if (!elemType)
            elemType = mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));

          auto winTy = WindowType::get(ctx, elemType);
          mlir::Value winVal = builder.create<Acquire>(
              loc, winTy, mlir::StringAttr::get(ctx, name),
              mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), count),
              mlir::StringAttr::get(ctx, portStr));

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
            }
          }
          acquiresToErase.push_back(op);

        } else if (auto op = mlir::dyn_cast<AIE::ObjectFifoReleaseOp>(rawOp)) {
          builder.setInsertionPoint(op);
          mlir::Location loc = op.getLoc();

          std::string name = op.getObjFifoName().str();
          int64_t count = op.getSize();
          std::string portStr =
              (op.getPort() == AIE::ObjectFifoPort::Produce) ? "Produce"
                                                             : "Consume";

          auto nameAttr = mlir::StringAttr::get(ctx, name);

          // First try the local block's window map (same-block acquire).
          mlir::Value winVal = blockWindowMap.lookup(nameAttr);

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
                mlir::StringAttr::get(ctx, portStr));
            blockWindowMap[nameAttr] = winVal;
          }

          builder.create<Release>(
              loc, winVal,
              mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), count),
              mlir::StringAttr::get(ctx, portStr));

          releasesToErase.push_back(op);
        }
      }
    });

    // Deferred erasure: erase subviews before acquires (subview uses acquire result).
    for (auto op : subviewsToErase)
      op.erase();
    for (auto op : acquiresToErase)
      op.erase();
    for (auto op : releasesToErase)
      op.erase();

    // Phase 5 (deferred erase): erase objectfifo create ops now that all
    // acquire/release/subview ops have been rewritten.
    for (AIE::ObjectFifoCreateOp op : fifosToErase)
      op.erase();
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
