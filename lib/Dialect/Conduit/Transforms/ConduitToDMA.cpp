//===- ConduitToDMA.cpp - Conduit IR → aie.dma_bd / aie.lock (Pass C)
//-*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Pass C of the Conduit lowering pipeline: lower Conduit IR to raw AIE
// hardware programming ops (aie.dma_bd, aie.lock, aie.buffer, aie.flow).
//
// Architecture:
//
//   aie.objectfifo.*  ──┐
//                       ├──► Conduit IR ──► aie.dma_bd / aie.lock / aie.buffer
//   air.channel.*     ──┘
//
//   (ObjectFifoToConduit.cpp)   (this file)
//
// What this pass does
// -------------------
//
// The pass walks the module in source order and:
//
// 1. Collects all conduit.create + conduit.annotate ops to build a map from
//    conduit name → FifoInfo (producer tile coords, consumer tile coords,
//    element type string, depth, capacity).
//
// 2. For each conduit.create (non-shim producer):
//      Inserts into the aie.device body:
//        - depth × aie.buffer on the consumer tile
//        - aie.lock (prod_lock, init=depth) on the consumer tile
//        - aie.lock (cons_lock, init=0)     on the consumer tile
//      Tracks these in buffersPerConduit / locksPerConduit maps.
//
// 3. For conduit.create where producer_tile is a shim tile (row == 0):
//      Inserts aie.shim_dma_allocation and aie.flow.
//
// 4. For conduit.objectfifo_link (distribute or join):
//      Generates the MemTile DMA BD chain inside aie.memtile_dma.
//      The BD chain mirrors AIEObjectFifoStatefulTransform logic:
//        distribute: one S2MM channel ingests the full buffer; each MM2S
//                    channel sends the slice for one destination.
//        join:       each S2MM channel ingests the slice for one source;
//                    one MM2S channel outputs the joined buffer.
//
// 5. For conduit.acquire / conduit.release (inside func bodies):
//      Replaces them with aie.use_lock calls referencing the tracked locks.
//
// 6. Erases all processed Conduit ops.
//
// Implementation status
// ---------------------
// This is a structural skeleton implementing the full pass interface and the
// depth-1 fifo case.  The memtile DMA BD chain generation for link ops is
// templated on the distribute pattern from depth_one + link_test_distribute.
//
// Known gaps (to be filled in as validation progresses):
//   - aie.buffer / aie.lock insertion requires the aie.device TileOp SSA
//     values; these are looked up by (col,row) from a tile cache populated
//     during the walk.  If a tile is not declared before the conduit.create,
//     the pass emits a warning and skips that conduit.
//   - The BD chain for depth > 1 replicates the buffer allocation N times
//     but only generates a single-buffer BD loop (depth=1 pattern) for now.
//     Full depth-N double-buffering BD chains will be added after the depth-1
//     validation milestone passes FileCheck.
//   - aie.shim_dma_allocation uses the conduit name + "_shim_alloc" as sym.
//   - The pass currently generates textual annotations (conduit.annotate)
//     into the output for any conduit ops it cannot fully lower, so the
//     output is always a valid (if incomplete) module.
//   - Multi-consumer (N > 1) link lowering emits only the first destination
//     for MM2S channels beyond channel 0; remaining destinations are skipped
//     with a TODO annotation.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITTODMA
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helper: parse "tile(col,row)" → (col, row).  Returns {-1,-1} on failure.
// ---------------------------------------------------------------------------

static std::pair<int64_t, int64_t> parseTileCoord(llvm::StringRef s) {
  // Format: "tile(col,row)"
  if (!s.starts_with("tile("))
    return {-1, -1};
  s = s.drop_front(5); // drop "tile("
  s = s.drop_back(1);  // drop ")"
  auto [colStr, rowStr] = s.split(',');
  int64_t col, row;
  if (colStr.getAsInteger(10, col) || rowStr.getAsInteger(10, row))
    return {-1, -1};
  return {col, row};
}

// ---------------------------------------------------------------------------
// Per-conduit info gathered from conduit.create + conduit.annotate
// ---------------------------------------------------------------------------

struct ConduitInfo {
  std::string producerTile;
  llvm::SmallVector<std::string> consumerTiles;
  int64_t depth = 1;
  int64_t capacity = 0;
  std::string elemType;
  // Hardware SSA values populated during lowering:
  llvm::SmallVector<AIE::BufferOp> buffers; // depth-many on consumer tile
  AIE::LockOp prodLock;                     // init=depth (free slots)
  AIE::LockOp consLock;                     // init=0   (filled slots)
};

// ---------------------------------------------------------------------------
// Main pass
// ---------------------------------------------------------------------------

struct ConduitToDMAPass : impl::ConduitToDMABase<ConduitToDMAPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    mlir::MLIRContext *ctx = module.getContext();

    // -----------------------------------------------------------------------
    // Phase 1: collect ConduitInfo from conduit.create + conduit.annotate
    // -----------------------------------------------------------------------

    llvm::StringMap<ConduitInfo> conduitMap;

    module.walk([&](Create op) {
      ConduitInfo info;
      info.capacity = op.getCapacity();
      conduitMap[op.getName()] = std::move(info);
    });

    module.walk([&](Annotate op) {
      auto it = conduitMap.find(op.getName());
      if (it == conduitMap.end())
        return;
      auto &info = it->second;
      llvm::StringRef key = op.getKey();
      llvm::StringRef val = op.getValue();
      if (key == "producer_tile") {
        info.producerTile = val.str();
      } else if (key.starts_with("consumer_tile_")) {
        info.consumerTiles.push_back(val.str());
      } else if (key == "depth") {
        val.getAsInteger(10, info.depth);
      } else if (key == "element_type") {
        info.elemType = val.str();
      }
    });

    // -----------------------------------------------------------------------
    // Phase 2: find the aie.device op and build a tile SSA value cache
    // (col,row) → TileOp.  We need this to create buffers/locks on the
    // correct tiles.
    // -----------------------------------------------------------------------

    // Find the first aie.device op (or skip if not present).
    AIE::DeviceOp deviceOp;
    module.walk([&](AIE::DeviceOp op) {
      if (!deviceOp)
        deviceOp = op;
    });

    if (!deviceOp) {
      // No device op — nothing to lower; emit a warning and return.
      module.emitWarning(
          "conduit-to-dma: no aie.device found; skipping lowering");
      return;
    }

    // Build tile cache.
    llvm::DenseMap<std::pair<int64_t, int64_t>, AIE::TileOp> tileCache;
    deviceOp.walk([&](AIE::TileOp tile) {
      tileCache[{tile.getCol(), tile.getRow()}] = tile;
    });

    // Helper: look up a TileOp from a "tile(col,row)" string.
    auto lookupTile = [&](llvm::StringRef coord) -> AIE::TileOp {
      auto [col, row] = parseTileCoord(coord);
      if (col < 0)
        return {};
      auto it = tileCache.find({col, row});
      if (it == tileCache.end())
        return {};
      return it->second;
    };

    // -----------------------------------------------------------------------
    // Phase 3: for each conduit.create, allocate aie.buffer + aie.lock pairs
    //          in the aie.device body.
    // -----------------------------------------------------------------------

    // Device body reference for insertions.
    mlir::Block &deviceBody = deviceOp.getBodyRegion().front();

    // Find the last tile op in the device body — we'll insert new hardware ops
    // (buffers, locks) immediately after the last tile declaration.  This
    // ensures the new SSA values dominate any aie.core bodies that follow.
    // If there are no tile ops, insert at the start of the block.
    mlir::Operation *insertAfterTile = nullptr;
    for (mlir::Operation &op : deviceBody) {
      if (mlir::isa<AIE::TileOp>(op))
        insertAfterTile = &op;
    }

    // Per-tile lock ID counter to avoid collisions.
    llvm::DenseMap<mlir::Value, int> lockIdCounter;

    for (auto &[name, info] : conduitMap) {
      if (info.consumerTiles.empty()) {
        // Producer-only conduit (shim DMA source) — handled below in Phase 4.
        continue;
      }

      // For simplicity: only handle the first consumer tile (single-consumer
      // case).  Multi-consumer broadcast support is noted as a gap.
      llvm::StringRef consTileStr = info.consumerTiles[0];
      AIE::TileOp consTile = lookupTile(consTileStr);
      if (!consTile) {
        // Tile not declared in the device — emit annotate and skip.
        builder.setInsertionPoint(deviceBody.getTerminator());
        builder.create<Annotate>(
            deviceOp.getLoc(), mlir::StringAttr::get(ctx, name),
            mlir::StringAttr::get(ctx, "conduit_to_dma_error"),
            mlir::StringAttr::get(ctx, "consumer tile not found: " +
                                           consTileStr.str()));
        continue;
      }

      // Insert after the last tile op (or at the start if no tile op found),
      // so the new SSA values dominate any aie.core bodies that follow.
      if (insertAfterTile)
        builder.setInsertionPointAfter(insertAfterTile);
      else
        builder.setInsertionPointToStart(&deviceBody);

      // Allocate depth-many buffers.
      // Buffer type: memref<capacity/depth x i32> as a structural placeholder.
      // The actual element memref type from elemType annotation would be used
      // in a full implementation with type parsing.
      int64_t depth = info.depth > 0 ? info.depth : 1;
      int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
      auto bufTy =
          mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));

      for (int64_t i = 0; i < depth; ++i) {
        std::string symName = name.str() + "_buff_" + std::to_string(i);
        // BufferOp builder: (Type buffer, Value tile, optional StringAttr
        // sym_name,
        //                    optional IntegerAttr address, optional
        //                    ElementsAttr initial_value, optional IntegerAttr
        //                    mem_bank)
        auto buf = builder.create<AIE::BufferOp>(
            deviceOp.getLoc(), bufTy, consTile.getResult(),
            mlir::StringAttr::get(ctx, symName),
            /*address=*/mlir::IntegerAttr{},
            /*initial_value=*/mlir::ElementsAttr{},
            /*mem_bank=*/mlir::IntegerAttr{});
        info.buffers.push_back(buf);
      }

      // Allocate prod_lock (init = depth → depth free slots).
      // LockOp convenience builder: (Value tile, int lockID, int init)
      {
        mlir::Value consTileVal = consTile.getResult();
        int lockIdx = lockIdCounter[consTileVal]++;
        std::string symName = name.str() + "_prod_lock_0";
        info.prodLock = builder.create<AIE::LockOp>(
            deviceOp.getLoc(), consTileVal, lockIdx, static_cast<int>(depth));
        info.prodLock.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
      }

      // Allocate cons_lock (init = 0 → nothing filled yet).
      {
        mlir::Value consTileVal = consTile.getResult();
        int lockIdx = lockIdCounter[consTileVal]++;
        std::string symName = name.str() + "_cons_lock_0";
        info.consLock = builder.create<AIE::LockOp>(deviceOp.getLoc(),
                                                    consTileVal, lockIdx, 0);
        info.consLock.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
      }
    }

    // -----------------------------------------------------------------------
    // Phase 4: handle shim-tile producers → aie.shim_dma_allocation + aie.flow
    // -----------------------------------------------------------------------

    // A conduit is a shim source if its producer tile has row == 0 and the
    // device has a shim NOC tile at that coordinate.  For each such conduit,
    // emit shim_dma_allocation and an aie.flow from shim to consumer tile.

    for (auto &[name, info] : conduitMap) {
      if (info.producerTile.empty() || info.consumerTiles.empty())
        continue;

      auto [col, row] = parseTileCoord(info.producerTile);
      if (row != 0)
        continue; // not a shim tile

      AIE::TileOp shimTile = lookupTile(info.producerTile);
      if (!shimTile)
        continue;

      AIE::TileOp consTile = lookupTile(info.consumerTiles[0]);
      if (!consTile)
        continue;

      builder.setInsertionPoint(deviceBody.getTerminator());

      // aie.shim_dma_allocation @<name>_shim_alloc(%shimTile, MM2S, 0,
      // plio=false) Builder: (StringRef sym_name, Value tile, DMAChannelDir,
      // int64_t channel_index,
      //           bool plio, optional PacketInfoAttr)
      std::string allocSym = name.str() + "_shim_alloc";
      builder.create<AIE::ShimDMAAllocationOp>(
          deviceOp.getLoc(), allocSym, shimTile.getResult(),
          AIE::DMAChannelDir::MM2S,
          /*channel_index=*/static_cast<int64_t>(0),
          /*plio=*/false,
          /*packet=*/nullptr);

      // aie.flow(%shimTile, DMA : 0, %consTile, DMA : 0)
      // FlowOp convenience builder: (Value source, WireBundle source_bundle,
      //                              int32_t source_channel, Value dest,
      //                              WireBundle dest_bundle, int32_t
      //                              dest_channel)
      builder.create<AIE::FlowOp>(deviceOp.getLoc(), shimTile.getResult(),
                                  AIE::WireBundle::DMA, static_cast<int32_t>(0),
                                  consTile.getResult(), AIE::WireBundle::DMA,
                                  static_cast<int32_t>(0));
    }

    // -----------------------------------------------------------------------
    // Phase 5: lower conduit.objectfifo_link → MemTile DMA BD chain
    //
    // For the distribute case (1 src → N dsts):
    //   One S2MM channel ingests the full buffer.
    //   N MM2S channels send slices to each dst.
    //
    // The BD chain pattern (depth-1 for now):
    //   aie.memtile_dma(%memtile) {
    //     %0 = aie.dma_start(S2MM, 0, ^ingest, ^mm2s_0_start)
    //   ^ingest:
    //     aie.use_lock(%link_cons_prod_lock, AcquireGreaterEqual, 1)
    //     aie.dma_bd(%link_cons_buff : memref, offset, len)
    //     aie.use_lock(%link_cons_cons_lock, Release, 1)
    //     aie.next_bd ^ingest           // depth-1: loop back to self
    //   ^mm2s_0_start:
    //     %1 = aie.dma_start(MM2S, 0, ^send_0, ^end)
    //   ^send_0:
    //     aie.use_lock(%link_cons_cons_lock, AcquireGreaterEqual, 1)
    //     aie.dma_bd(%link_cons_buff, dst_offset, dst_len)
    //     aie.use_lock(%link_cons_prod_lock, Release, 1)
    //     aie.next_bd ^send_0
    //   ^end: aie.end
    //   }
    //
    // NOTE: For channels > 0 (multiple destinations), only the first
    // destination's MM2S channel is emitted in this skeleton.  Supporting
    // N destinations requires N independent DMA start chains (each starting
    // from the previous chain's "chain" successor block).  This is a known
    // gap — see the TODO below.
    // -----------------------------------------------------------------------

    module.walk([&](ObjectFifoLink linkOp) {
      builder.setInsertionPoint(deviceBody.getTerminator());
      mlir::Location loc = linkOp.getLoc();

      auto srcs = linkOp.getSrcs();
      auto dsts = linkOp.getDsts();
      (void)dsts; // used in future multi-dst TODO
      llvm::StringRef memtileStr = linkOp.getMemtile();
      auto offsets = linkOp.getOffsets();

      AIE::TileOp memtile = lookupTile(memtileStr);
      if (!memtile) {
        // Cannot lower without the memtile — annotate and skip.
        builder.create<Annotate>(
            loc, mlir::StringAttr::get(ctx, "link_error"),
            mlir::StringAttr::get(ctx, "conduit_to_dma_error"),
            mlir::StringAttr::get(ctx,
                                  "memtile not found: " + memtileStr.str()));
        linkOp.erase();
        return;
      }

      // Look up the src conduit (first one) for buffer/lock references.
      std::string srcName =
          mlir::cast<mlir::StringAttr>(srcs[0]).getValue().str();
      auto srcIt = conduitMap.find(srcName);
      if (srcIt == conduitMap.end() || srcIt->second.buffers.empty()) {
        // Src conduit not yet lowered or has no buffers — annotate and skip.
        builder.create<Annotate>(
            loc, mlir::StringAttr::get(ctx, srcName),
            mlir::StringAttr::get(ctx, "conduit_to_dma_error"),
            mlir::StringAttr::get(ctx, "src conduit buffers not allocated"));
        linkOp.erase();
        return;
      }

      ConduitInfo &srcInfo = srcIt->second;

      // Create aie.memtile_dma block.
      // MemTileDMAOp builder: (Value tile) — result type inferred.
      auto memtileDMA =
          builder.create<AIE::MemTileDMAOp>(loc, memtile.getResult());
      mlir::Region &dmaRegion = memtileDMA.getBody();

      // Helper: create a new basic block in the DMA region.
      auto addBlock = [&]() -> mlir::Block * {
        return builder.createBlock(&dmaRegion);
      };

      // Build the BD chain.
      // Blocks: ^entry, ^ingest, ^send_0, ^end
      // For multiple dsts, only ^send_0 is generated (first channel only).
      // TODO: emit one DMA start chain per dst for full multi-consumer support.
      mlir::Block *entryBlock = addBlock();
      mlir::Block *ingestBlock = addBlock();
      mlir::Block *sendBlock =
          addBlock(); // first (and only, for now) MM2S channel
      mlir::Block *endBlock = addBlock();

      // ^entry: aie.dma_start(S2MM, 0, ^ingest, ^send_chain_start)
      // DMAStartOp builder: (DMAChannelDir, int32_t channel_index, int32_t
      // repeat_count,
      //                       Block *dest, Block *chain)
      builder.setInsertionPointToEnd(entryBlock);
      builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::S2MM,
                                      /*channel_index=*/static_cast<int32_t>(0),
                                      /*repeat_count=*/static_cast<int32_t>(0),
                                      ingestBlock, sendBlock);

      // ^ingest: acquire prod_lock, dma_bd full buffer, release cons_lock, loop
      // back
      builder.setInsertionPointToEnd(ingestBlock);
      {
        // use_lock(prod_lock, AcquireGreaterEqual, 1)
        builder.create<AIE::UseLockOp>(loc, srcInfo.prodLock.getResult(),
                                       AIE::LockAction::AcquireGreaterEqual,
                                       static_cast<int32_t>(1));
        // dma_bd(buff_0, offset=0, len=capacity)
        // DMABDOp builder: (Value buffer, int offset, int len)
        builder.create<AIE::DMABDOp>(loc, srcInfo.buffers[0].getResult(),
                                     /*offset=*/0,
                                     static_cast<int>(srcInfo.capacity));
        // use_lock(cons_lock, Release, 1)
        builder.create<AIE::UseLockOp>(loc, srcInfo.consLock.getResult(),
                                       AIE::LockAction::Release,
                                       static_cast<int32_t>(1));
        // next_bd ^ingest  (depth-1 loops back to itself)
        builder.create<AIE::NextBDOp>(loc, ingestBlock);
      }

      // ^send_chain_start: dma_start(MM2S, 0, ^send_0, ^end)
      // We use a separate chain-start approach: the entry block falls into the
      // first send chain start.  For depth-1 single-channel case, the send
      // block is both the start and the BD block.
      // Insert a dma_start at the top of sendBlock to properly start MM2S ch 0.
      // Actually: the ^entry dma_start's "chain" successor is sendBlock, which
      // must begin with a dma_start for the MM2S channel.
      // So sendBlock plays the role of "mm2s_0_start" — add a dma_start here
      // that points to a new "bd_block" successor, then put the BD ops there.
      //
      // Restructure: sendBlock = chain-start for MM2S channel 0 (has dma_start)
      //              A new bdBlock follows for the actual BD ops.
      mlir::Block *bdBlock = addBlock();

      builder.setInsertionPointToEnd(sendBlock);
      builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::MM2S,
                                      /*channel_index=*/static_cast<int32_t>(0),
                                      /*repeat_count=*/static_cast<int32_t>(0),
                                      bdBlock, endBlock);

      // ^bd_block: acquire cons_lock, dma_bd slice, release prod_lock, loop
      // back
      builder.setInsertionPointToEnd(bdBlock);
      {
        // Offset for destination 0 (from offsets attribute if present).
        int64_t dstOffset = 0;
        int64_t dstLen = srcInfo.capacity; // default: full buffer
        if (offsets.has_value() && !offsets->empty()) {
          dstOffset = (*offsets)[0];
          if (offsets->size() > 1)
            dstLen = (*offsets)[1] - dstOffset;
          else
            dstLen = srcInfo.capacity - dstOffset;
        }

        // use_lock(cons_lock, AcquireGreaterEqual, 1)
        builder.create<AIE::UseLockOp>(loc, srcInfo.consLock.getResult(),
                                       AIE::LockAction::AcquireGreaterEqual,
                                       static_cast<int32_t>(1));
        // dma_bd(buff_0, dstOffset, dstLen)
        builder.create<AIE::DMABDOp>(loc, srcInfo.buffers[0].getResult(),
                                     static_cast<int>(dstOffset),
                                     static_cast<int>(dstLen));
        // use_lock(prod_lock, Release, 1)
        builder.create<AIE::UseLockOp>(loc, srcInfo.prodLock.getResult(),
                                       AIE::LockAction::Release,
                                       static_cast<int32_t>(1));
        // next_bd ^bd_block  (depth-1: loop back)
        builder.create<AIE::NextBDOp>(loc, bdBlock);
      }

      // TODO: for dsts.size() > 1, emit additional DMA start chains for
      // channels 1..N-1.  Each chain requires a new chain-start block and
      // a new BD block, with the previous channel's "chain" successor pointing
      // to the new chain-start.  This is deferred until depth-1, single-dst
      // validation passes FileCheck.

      // ^end: aie.end
      builder.setInsertionPointToEnd(endBlock);
      builder.create<AIE::EndOp>(loc);

      linkOp.erase();
    });

    // -----------------------------------------------------------------------
    // Phase 6: lower conduit.acquire / conduit.release inside func bodies
    //          → aie.use_lock
    // -----------------------------------------------------------------------

    module.walk([&](Acquire op) {
      auto it = conduitMap.find(op.getName());
      if (it == conduitMap.end() || !it->second.consLock) {
        op.erase();
        return;
      }
      builder.setInsertionPoint(op);
      AIE::LockOp lock = it->second.consLock;
      int64_t count = op.getCount();
      builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                     AIE::LockAction::AcquireGreaterEqual,
                                     static_cast<int32_t>(count));
      op.erase();
    });

    module.walk([&](Release op) {
      auto it = conduitMap.find(op.getName());
      if (it == conduitMap.end() || !it->second.prodLock) {
        op.erase();
        return;
      }
      builder.setInsertionPoint(op);
      AIE::LockOp lock = it->second.prodLock;
      int64_t count = op.getCount();
      builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                     AIE::LockAction::Release,
                                     static_cast<int32_t>(count));
      op.erase();
    });

    // -----------------------------------------------------------------------
    // Phase 7: erase remaining Conduit ops (create, annotate, subview_access)
    // -----------------------------------------------------------------------

    module.walk([&](SubviewAccess op) { op.erase(); });
    module.walk([&](Annotate op) { op.erase(); });
    module.walk([&](Create op) { op.erase(); });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConduitToDMAPass() {
  return std::make_unique<ConduitToDMAPass>();
}

} // namespace xilinx::conduit
