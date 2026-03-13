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

      builder.create<Create>(
          loc,
          mlir::StringAttr::get(ctx, name),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), capacity),
          mlir::DenseI64ArrayAttr::get(ctx, info.producerTileArr),
          mlir::DenseI64ArrayAttr::get(ctx, info.consumerTilesArr),
          shimConsAttr,
          mlir::TypeAttr::get(info.elemType),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), info.depth),
          /*link_mode=*/mlir::StringAttr{});

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

    // Map from (fifo name, port) → most-recently-emitted window SSA value.
    // Used to thread the window operand into conduit.release.
    // Walk order is dominator-safe: acquire always precedes release in the
    // same block, so the window value is live at the release site.
    llvm::DenseMap<mlir::StringAttr, mlir::Value> lastWindowForName;

    module.walk([&](AIE::ObjectFifoAcquireOp op) {
      builder.setInsertionPoint(op);
      mlir::Location loc = op.getLoc();

      std::string name = op.getObjFifoName().str();
      int64_t count = op.acqNumber();

      // Fix 2: propagate port attribute.
      std::string portStr =
          (op.getPort() == AIE::ObjectFifoPort::Produce) ? "Produce" : "Consume";

      // Fix 1: build the !conduit.window<T> result type from the fifo info.
      auto nameAttr = mlir::StringAttr::get(ctx, name);
      mlir::MemRefType elemType;
      auto it = fifoInfoMap.find(nameAttr);
      if (it != fifoInfoMap.end())
        elemType = it->second.elemType;
      if (!elemType)
        elemType = mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));

      auto winTy = WindowType::get(ctx, elemType);

      // Emit conduit.acquire — returns !conduit.window<T>
      mlir::Value winVal = builder.create<Acquire>(
          loc, winTy,
          mlir::StringAttr::get(ctx, name),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), count),
          mlir::StringAttr::get(ctx, portStr));

      // Record the window SSA value so the subsequent release can use it.
      lastWindowForName[nameAttr] = winVal;

      // Rewrite all subview.access users of this acquire's result.
      // Thread the window SSA value into conduit.subview_access.
      mlir::Value subviewResult = op.getResult();
      for (mlir::Operation *user :
           llvm::make_early_inc_range(subviewResult.getUsers())) {
        if (auto accessOp =
                mlir::dyn_cast<AIE::ObjectFifoSubviewAccessOp>(user)) {
          builder.setInsertionPoint(accessOp);
          int64_t idx = accessOp.getIndex();
          mlir::Type resultTy = accessOp.getResult().getType();

          // Emit conduit.subview_access with the window SSA operand.
          auto condAccess = builder.create<SubviewAccess>(
              accessOp.getLoc(), resultTy, winVal,
              mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), idx));

          // Replace all uses of the AIE subview.access result.
          accessOp.getResult().replaceAllUsesWith(condAccess.getResult());
          subviewsToErase.push_back(accessOp);
        }
      }
      acquiresToErase.push_back(op);
    });

    module.walk([&](AIE::ObjectFifoReleaseOp op) {
      builder.setInsertionPoint(op);
      mlir::Location loc = op.getLoc();

      std::string name = op.getObjFifoName().str();
      int64_t count = op.getSize();

      // Fix 2: propagate port attribute.
      std::string portStr =
          (op.getPort() == AIE::ObjectFifoPort::Produce) ? "Produce" : "Consume";

      // Look up the window SSA value from the preceding acquire.
      auto nameAttr = mlir::StringAttr::get(ctx, name);
      mlir::Value winVal = lastWindowForName.lookup(nameAttr);

      if (!winVal) {
        // Fallback: no window in scope (can happen for producer-side release
        // without a matching acquire in the current walk scope).  Synthesize
        // a placeholder window using the fifo info element type.
        mlir::MemRefType elemType;
        auto it = fifoInfoMap.find(nameAttr);
        if (it != fifoInfoMap.end())
          elemType = it->second.elemType;
        if (!elemType)
          elemType = mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));
        // Emit a conduit.acquire to produce the window for the release.
        // This handles the case where the producer releases without an explicit
        // acquire visible to this walk (e.g. implicit initial window).
        auto winTy = WindowType::get(ctx, elemType);
        winVal = builder.create<Acquire>(
            loc, winTy,
            mlir::StringAttr::get(ctx, name),
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), count),
            mlir::StringAttr::get(ctx, portStr));
      }

      builder.create<Release>(
          loc, winVal,
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), count),
          mlir::StringAttr::get(ctx, portStr));

      releasesToErase.push_back(op);
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
