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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
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
// Per-conduit info gathered from conduit.create typed attributes
// ---------------------------------------------------------------------------

struct ConduitInfo {
  // Tile coordinates parsed from the typed Create attributes.
  std::pair<int64_t, int64_t> producerTileCoord = {-1, -1};
  llvm::SmallVector<std::pair<int64_t, int64_t>> consumerTileCoords;
  // Shim consumer tiles (row==0): DMA endpoints, no local memory.
  // These get aie.shim_dma_allocation + aie.flow, not aie.buffer + aie.lock.
  llvm::SmallVector<std::pair<int64_t, int64_t>> shimConsumerTileCoords;
  int64_t depth = 1;
  int64_t capacity = 0;
  mlir::Type elemType; // actual element memref type (may be null)
  // Hardware SSA values populated during lowering:
  llvm::SmallVector<AIE::BufferOp> buffers; // depth-many on consumer tile
  AIE::LockOp prodLock;                     // init=depth (free slots)
  AIE::LockOp consLock;                     // init=0   (filled slots)
  // Legacy string form for the ObjectFifoLink memtile lookup.
  // Populated from producer_tile for shim detection.
  std::string producerTileStr; // "tile(col,row)"
  llvm::SmallVector<std::string> consumerTileStrs; // for link memtile lookup
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
    // Phase 1: collect ConduitInfo from conduit.create typed attributes.
    //
    // The new conduit.create carries all metadata as typed attributes:
    //   producer_tile   : DenseI64Array [col, row]
    //   consumer_tiles  : DenseI64Array [col0, row0, col1, row1, ...]
    //   element_type    : TypeAttr
    //   depth           : I64Attr
    // -----------------------------------------------------------------------

    llvm::StringMap<ConduitInfo> conduitMap;

    module.walk([&](Create op) {
      ConduitInfo info;
      info.capacity = op.getCapacity();

      // Extract depth. getDepth() returns std::optional<uint64_t>.
      if (auto depthOpt = op.getDepth())
        info.depth = static_cast<int64_t>(*depthOpt);

      // Extract element type. getElementType() returns std::optional<mlir::Type>.
      if (auto etOpt = op.getElementType())
        info.elemType = *etOpt;

      // Extract producer tile. getProducerTile() returns
      // std::optional<ArrayRef<int64_t>> with [col, row].
      if (auto pt = op.getProducerTile()) {
        if (pt->size() >= 2) {
          int64_t col = (*pt)[0], row = (*pt)[1];
          info.producerTileCoord = std::make_pair(col, row);
          std::string s;
          llvm::raw_string_ostream os(s);
          os << "tile(" << col << "," << row << ")";
          info.producerTileStr = os.str();
        }
      }

      // Extract consumer tiles. getConsumerTiles() returns
      // std::optional<ArrayRef<int64_t>> with [col0,row0,col1,row1,...].
      // These are non-shim (row > 0) compute tiles only (Pass A filters shim).
      if (auto ct = op.getConsumerTiles()) {
        for (size_t i = 0; i + 1 < ct->size(); i += 2) {
          int64_t col = (*ct)[i], row = (*ct)[i + 1];
          info.consumerTileCoords.push_back(std::make_pair(col, row));
          std::string s;
          llvm::raw_string_ostream os(s);
          os << "tile(" << col << "," << row << ")";
          info.consumerTileStrs.push_back(os.str());
        }
      }

      // Extract shim consumer tiles. getShimConsumerTiles() returns
      // std::optional<ArrayRef<int64_t>> with [col0,row0,...] for row==0 tiles.
      // Pass A stores shim tiles here instead of consumer_tiles.
      if (auto sct = op.getShimConsumerTiles()) {
        for (size_t i = 0; i + 1 < sct->size(); i += 2) {
          int64_t col = (*sct)[i], row = (*sct)[i + 1];
          info.shimConsumerTileCoords.push_back(std::make_pair(col, row));
        }
      }

      conduitMap[op.getName()] = std::move(info);
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

    // Helper: look up a TileOp from (col, row) coordinates.
    auto lookupTileByCoord = [&](int64_t col, int64_t row) -> AIE::TileOp {
      auto it = tileCache.find({col, row});
      if (it == tileCache.end())
        return {};
      return it->second;
    };

    for (auto &[name, info] : conduitMap) {
      if (info.consumerTileCoords.empty() &&
          info.shimConsumerTileCoords.empty()) {
        // Producer-only conduit (shim DMA source) — handled below in Phase 4.
        continue;
      }

      // Phase 3b: conduit has shim consumer(s) but no compute consumer.
      // In this case the buffer and locks live on the PRODUCER tile (compute
      // tile), not on the shim tile.  The DMA runs MM2S from the producer tile
      // to the shim receiver (Phase 4b generates shim_dma_allocation + flow).
      if (info.consumerTileCoords.empty() &&
          !info.shimConsumerTileCoords.empty()) {
        auto [prodCol, prodRow] = info.producerTileCoord;
        if (prodCol < 0 || prodRow == 0)
          continue; // skip if producer is also shim (degenerate)

        AIE::TileOp prodTile = lookupTileByCoord(prodCol, prodRow);
        if (!prodTile)
          continue;

        int64_t depth = info.depth > 0 ? info.depth : 1;
        mlir::Type bufTy = info.elemType;
        if (!bufTy) {
          int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
          bufTy = mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
        }

        if (insertAfterTile)
          builder.setInsertionPointAfter(insertAfterTile);
        else
          builder.setInsertionPointToStart(&deviceBody);

        mlir::Value prodTileVal = prodTile.getResult();

        // Allocate depth-many buffers on the producer tile.
        for (int64_t i = 0; i < depth; ++i) {
          std::string symName = name.str() + "_buff_" + std::to_string(i);
          auto buf = builder.create<AIE::BufferOp>(
              deviceOp.getLoc(), bufTy, prodTileVal,
              mlir::StringAttr::get(ctx, symName),
              /*address=*/mlir::IntegerAttr{},
              /*initial_value=*/mlir::ElementsAttr{},
              /*mem_bank=*/mlir::IntegerAttr{});
          info.buffers.push_back(buf);
        }

        // prod_lock (init=depth → free slots for the producer to fill).
        {
          int lockIdx = lockIdCounter[prodTileVal]++;
          std::string symName = name.str() + "_prod_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), prodTileVal, lockIdx,
              static_cast<int>(depth));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          info.prodLock = lk;
        }

        // cons_lock (init=0 → nothing filled yet).
        {
          int lockIdx = lockIdCounter[prodTileVal]++;
          std::string symName = name.str() + "_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), prodTileVal, lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          info.consLock = lk;
        }
        continue;
      }

      // Handle all consumer tiles for multi-consumer broadcast.
      int64_t depth = info.depth > 0 ? info.depth : 1;

      // Buffer type: use the actual element type from the conduit.create
      // element_type attribute if available; fall back to per-slot i32.
      mlir::Type bufTy = info.elemType;
      if (!bufTy) {
        int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
        bufTy = mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
      }

      for (unsigned consIdx = 0; consIdx < info.consumerTileCoords.size();
           ++consIdx) {
        auto [consCol, consRow] = info.consumerTileCoords[consIdx];
        AIE::TileOp consTile = lookupTileByCoord(consCol, consRow);
        if (!consTile) {
          // Tile not declared in the device — skip this consumer with warning.
          module.emitWarning(
              "conduit-to-dma: consumer tile (" + std::to_string(consCol) +
              "," + std::to_string(consRow) +
              ") for conduit '" + name.str() + "' not found in device");
          continue;
        }

        // Insert after the last tile op so SSA values dominate aie.core bodies.
        if (insertAfterTile)
          builder.setInsertionPointAfter(insertAfterTile);
        else
          builder.setInsertionPointToStart(&deviceBody);

        // For the first consumer, track buffers and locks in info.buffers /
        // info.prodLock / info.consLock (used by acquire/release lowering and
        // the link phase).  Additional consumers get their own buffers/locks
        // but are not yet wired into the acquire/release path.
        mlir::Value consTileVal = consTile.getResult();

        // Allocate depth-many buffers on this consumer tile.
        // The "_cons_N" infix distinguishes consumer-tile resources from
        // shim-tile resources (Phase 4) which use the base name.
        // With multiple consumers, append the consumer index: _cons_0, _cons_1...
        // With a single consumer, use the plain _cons_ infix to match the
        // naming convention of --aie-objectFifo-stateful-transform.
        std::string bufSuffix =
            info.consumerTileCoords.size() > 1
                ? "_cons_" + std::to_string(consIdx)
                : "_cons";

        llvm::SmallVector<AIE::BufferOp> consBuffers;
        for (int64_t i = 0; i < depth; ++i) {
          std::string symName =
              name.str() + bufSuffix + "_buff_" + std::to_string(i);
          auto buf = builder.create<AIE::BufferOp>(
              deviceOp.getLoc(), bufTy, consTileVal,
              mlir::StringAttr::get(ctx, symName),
              /*address=*/mlir::IntegerAttr{},
              /*initial_value=*/mlir::ElementsAttr{},
              /*mem_bank=*/mlir::IntegerAttr{});
          consBuffers.push_back(buf);
          if (consIdx == 0)
            info.buffers.push_back(buf);
        }

        // Allocate prod_lock (init = depth → depth free slots).
        // Named with _cons_ infix to distinguish from shim-tile prod_lock_0.
        {
          int lockIdx = lockIdCounter[consTileVal]++;
          std::string symName =
              name.str() + bufSuffix + "_prod_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), consTileVal, lockIdx,
              static_cast<int>(depth));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          if (consIdx == 0)
            info.prodLock = lk;
        }

        // Allocate cons_lock (init = 0 → nothing filled yet).
        {
          int lockIdx = lockIdCounter[consTileVal]++;
          std::string symName =
              name.str() + bufSuffix + "_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), consTileVal, lockIdx, 0);
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          if (consIdx == 0)
            info.consLock = lk;
        }
      }
    }

    // -----------------------------------------------------------------------
    // Phase 4: handle shim-tile endpoints → aie.shim_dma_allocation + aie.flow
    //
    // Two sub-cases:
    //   4a. Producer is shim (row==0): shim sends (MM2S) to compute consumer.
    //       Emits: shim_dma_allocation(MM2S), flow(shim→consumer),
    //              and shim-side prod/cons locks (needed by host runtime).
    //   4b. Consumer is shim (row==0): compute producer sends (MM2S) to shim
    //       receiver (S2MM). Emits: shim_dma_allocation(S2MM),
    //              flow(producer→shim).
    // -----------------------------------------------------------------------

    for (auto &[name, info] : conduitMap) {
      auto [prodCol, prodRow] = info.producerTileCoord;
      if (prodCol < 0)
        continue;

      // --- Sub-case 4a: producer is a shim tile (row==0) ---
      if (prodRow == 0) {
        if (info.consumerTileCoords.empty())
          continue;

        AIE::TileOp shimTile = lookupTileByCoord(prodCol, prodRow);
        if (!shimTile)
          continue;

        auto [consCol, consRow] = info.consumerTileCoords[0];
        AIE::TileOp consTile = lookupTileByCoord(consCol, consRow);
        if (!consTile)
          continue;

        builder.setInsertionPoint(deviceBody.getTerminator());

        // Emit shim-side prod/cons locks (matching stateful transform output).
        // These have init=0 on the shim tile.  The host runtime uses them.
        {
          int lockIdx = lockIdCounter[shimTile.getResult()]++;
          std::string symName = name.str() + "_prod_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), shimTile.getResult(), lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        }
        {
          int lockIdx = lockIdCounter[shimTile.getResult()]++;
          std::string symName = name.str() + "_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), shimTile.getResult(), lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        }

        // aie.shim_dma_allocation @<name>_shim_alloc(%shimTile, MM2S, 0)
        std::string allocSym = name.str() + "_shim_alloc";
        builder.create<AIE::ShimDMAAllocationOp>(
            deviceOp.getLoc(), allocSym, shimTile.getResult(),
            AIE::DMAChannelDir::MM2S,
            /*channel_index=*/static_cast<int64_t>(0),
            /*plio=*/false,
            /*packet=*/nullptr);

        // aie.flow(%shimTile, DMA : 0, %consTile, DMA : 0)
        builder.create<AIE::FlowOp>(deviceOp.getLoc(), shimTile.getResult(),
                                    AIE::WireBundle::DMA,
                                    static_cast<int32_t>(0),
                                    consTile.getResult(), AIE::WireBundle::DMA,
                                    static_cast<int32_t>(0));
      }

      // --- Sub-case 4b: consumer is a shim tile (row==0) ---
      // Compute producer sends MM2S to shim; shim receives S2MM.
      for (auto [shimCol, shimRow] : info.shimConsumerTileCoords) {
        if (shimRow != 0)
          continue; // safety check

        AIE::TileOp prodTile = lookupTileByCoord(prodCol, prodRow);
        if (!prodTile)
          continue;

        AIE::TileOp shimTile = lookupTileByCoord(shimCol, shimRow);
        if (!shimTile)
          continue;

        builder.setInsertionPoint(deviceBody.getTerminator());

        // aie.shim_dma_allocation @<name>_shim_alloc(%shimTile, S2MM, 0)
        // The shim is the receiver when the compute tile is the producer.
        std::string allocSym = name.str() + "_shim_alloc";
        builder.create<AIE::ShimDMAAllocationOp>(
            deviceOp.getLoc(), allocSym, shimTile.getResult(),
            AIE::DMAChannelDir::S2MM,
            /*channel_index=*/static_cast<int64_t>(0),
            /*plio=*/false,
            /*packet=*/nullptr);

        // aie.flow(%prodTile, DMA : 0, %shimTile, DMA : 0)
        builder.create<AIE::FlowOp>(deviceOp.getLoc(), prodTile.getResult(),
                                    AIE::WireBundle::DMA,
                                    static_cast<int32_t>(0),
                                    shimTile.getResult(), AIE::WireBundle::DMA,
                                    static_cast<int32_t>(0));
      }
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
      llvm::StringRef memtileStr = linkOp.getMemtile();
      auto offsets = linkOp.getOffsets();

      AIE::TileOp memtile = lookupTile(memtileStr);
      if (!memtile) {
        // Cannot lower without the memtile — emit a warning and skip.
        module.emitWarning("conduit-to-dma: memtile not found for link op: " +
                           memtileStr.str());
        linkOp.erase();
        return;
      }

      // Look up the src conduit (first one) for buffer/lock references.
      std::string srcName =
          mlir::cast<mlir::StringAttr>(srcs[0]).getValue().str();
      auto srcIt = conduitMap.find(srcName);
      if (srcIt == conduitMap.end() || srcIt->second.buffers.empty()) {
        // Src conduit not yet lowered or has no buffers — emit a warning and skip.
        module.emitWarning("conduit-to-dma: src conduit '" + srcName +
                           "' buffers not allocated for link op");
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

      // Determine depth from the src conduit (clamped to at least 1).
      int64_t linkDepth = srcInfo.depth > 0 ? srcInfo.depth : 1;
      // Per-buffer element count (capacity / depth).
      int64_t perBufLen =
          srcInfo.capacity > 0 ? srcInfo.capacity / linkDepth : 1;

      // Build the BD chain for the S2MM (ingest) path.
      //
      // Pattern for depth=N (double-buffering when N=2):
      //   ^entry: dma_start(S2MM, 0, ^ingest_0, ^mm2s_chain_start)
      //   ^ingest_0:
      //     use_lock(prod_lock, AcquireGreaterEqual, 1)
      //     dma_bd(buff_0, 0, perBufLen)
      //     use_lock(cons_lock, Release, 1)
      //     next_bd ^ingest_1          ← for depth>1; else next_bd ^ingest_0
      //   ^ingest_1:  (depth >= 2)
      //     use_lock(prod_lock, AcquireGreaterEqual, 1)
      //     dma_bd(buff_1, 0, perBufLen)
      //     use_lock(cons_lock, Release, 1)
      //     next_bd ^ingest_0          ← ring back to first
      //   ...
      //
      // The MM2S (send) path mirrors this ring with its own depth-N BD blocks.

      mlir::Block *entryBlock = addBlock();

      // Create depth-many ingest BD blocks.
      llvm::SmallVector<mlir::Block *> ingestBlocks;
      for (int64_t i = 0; i < linkDepth; ++i)
        ingestBlocks.push_back(addBlock());

      // ^entry: dma_start(S2MM, 0, ^ingest_0, ^mm2s_chain_start)
      // The "chain" successor points to the block that will hold the first
      // MM2S dma_start; we create it now as a placeholder and fill it next.
      mlir::Block *mm2sChainStartBlock = addBlock();

      builder.setInsertionPointToEnd(entryBlock);
      builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::S2MM,
                                      /*channel_index=*/static_cast<int32_t>(0),
                                      /*repeat_count=*/static_cast<int32_t>(0),
                                      ingestBlocks[0], mm2sChainStartBlock);

      // Fill each ingest BD block.
      for (int64_t i = 0; i < linkDepth; ++i) {
        builder.setInsertionPointToEnd(ingestBlocks[i]);
        // Acquire a free slot (prod_lock).
        builder.create<AIE::UseLockOp>(loc, srcInfo.prodLock.getResult(),
                                       AIE::LockAction::AcquireGreaterEqual,
                                       static_cast<int32_t>(1));
        // DMA descriptor for buff_i.
        builder.create<AIE::DMABDOp>(loc, srcInfo.buffers[i].getResult(),
                                     /*offset=*/0,
                                     static_cast<int>(perBufLen));
        // Mark the slot filled (cons_lock).
        builder.create<AIE::UseLockOp>(loc, srcInfo.consLock.getResult(),
                                       AIE::LockAction::Release,
                                       static_cast<int32_t>(1));
        // Ring: next buffer, wrapping back to 0.
        mlir::Block *nextIngest = ingestBlocks[(i + 1) % linkDepth];
        builder.create<AIE::NextBDOp>(loc, nextIngest);
      }

      // Build MM2S send chains — one chain per destination (dst) fifo.
      // Each chain: dma_start(MM2S, ch, ^bd_0, ^next_chain_or_end)
      //             ^bd_0 .. ^bd_{N-1}: use_lock / dma_bd / use_lock / next_bd
      //
      // Offsets list encodes per-dst slice boundaries:
      //   distribute: offsets[i] is the start offset for dst[i];
      //               slice len = offsets[i+1] - offsets[i]  (or capacity -
      //               offset for last).
      // If no offsets are specified, each destination gets the full buffer.

      // We need to track the previous chain's "chain" successor block so we
      // can fill it with the next dma_start.  The entry block already pointed
      // its chain successor to mm2sChainStartBlock; that becomes the start of
      // channel 0's dma_start.
      mlir::Block *prevChainBlock = mm2sChainStartBlock;
      // endBlock is created last so it appears at the end of the block list.
      mlir::Block *endBlock = nullptr;

      for (unsigned dstIdx = 0; dstIdx < dsts.size(); ++dstIdx) {
        // Compute slice offset and length for this destination.
        // Offsets are specified in units of elements within a single buffer
        // slot (not the total depth-scaled capacity).  perBufLen is the size
        // of one buffer slot.
        int64_t dstOffset = 0;
        int64_t dstLen = perBufLen;
        if (offsets.has_value() && !offsets->empty()) {
          if (dstIdx < static_cast<unsigned>(offsets->size()))
            dstOffset = (*offsets)[dstIdx];
          // Length extends to next offset boundary (or end of slot).
          if (dstIdx + 1 < static_cast<unsigned>(offsets->size()))
            dstLen = (*offsets)[dstIdx + 1] - dstOffset;
          else
            dstLen = perBufLen - dstOffset;
        }

        // Create depth-many send BD blocks for this dst channel.
        llvm::SmallVector<mlir::Block *> sendBDBlocks;
        for (int64_t i = 0; i < linkDepth; ++i)
          sendBDBlocks.push_back(addBlock());

        // The next chain block after this dst's chain.  For the last dst,
        // create the end block now (so it is placed last); for intermediate
        // dsts, create a new chain-start block.
        mlir::Block *nextChainBlock;
        if (dstIdx + 1 == dsts.size()) {
          endBlock = addBlock(); // created last → appears last in IR
          nextChainBlock = endBlock;
        } else {
          nextChainBlock = addBlock();
        }

        // Fill prevChainBlock with dma_start(MM2S, dstIdx, ^bd_0,
        // ^nextChain).
        builder.setInsertionPointToEnd(prevChainBlock);
        builder.create<AIE::DMAStartOp>(
            loc, AIE::DMAChannelDir::MM2S,
            /*channel_index=*/static_cast<int32_t>(dstIdx),
            /*repeat_count=*/static_cast<int32_t>(0), sendBDBlocks[0],
            nextChainBlock);

        // Fill each send BD block for this destination.
        for (int64_t i = 0; i < linkDepth; ++i) {
          builder.setInsertionPointToEnd(sendBDBlocks[i]);
          // Acquire a filled slot (cons_lock).
          builder.create<AIE::UseLockOp>(loc, srcInfo.consLock.getResult(),
                                         AIE::LockAction::AcquireGreaterEqual,
                                         static_cast<int32_t>(1));
          // DMA descriptor: slice of buff_i for this dst.
          builder.create<AIE::DMABDOp>(loc, srcInfo.buffers[i].getResult(),
                                       static_cast<int>(dstOffset),
                                       static_cast<int>(dstLen));
          // Release the slot (prod_lock).
          builder.create<AIE::UseLockOp>(loc, srcInfo.prodLock.getResult(),
                                         AIE::LockAction::Release,
                                         static_cast<int32_t>(1));
          // Ring back.
          mlir::Block *nextBD = sendBDBlocks[(i + 1) % linkDepth];
          builder.create<AIE::NextBDOp>(loc, nextBD);
        }

        prevChainBlock = nextChainBlock;
      }

      // If dsts is empty (degenerate case), create the end block now.
      if (!endBlock)
        endBlock = addBlock();

      // ^end: aie.end
      builder.setInsertionPointToEnd(endBlock);
      builder.create<AIE::EndOp>(loc);

      linkOp.erase();
    });

    // -----------------------------------------------------------------------
    // Phase 5.5: for each simple (non-link) conduit that has a non-shim
    // producer tile and at least one buffer, generate an aie.mem BD chain
    // on the consumer tile so the DMA engine can transfer data.
    //
    // The pattern mirrors createAIETileDMA in AIEObjectFifoStatefulTransform:
    //
    //   aie.mem(%consTile) {
    //     %0 = aie.dma_start(S2MM, 0, ^bd_0, ^end)
    //   ^bd_0:
    //     aie.use_lock(%prod_lock, AcquireGreaterEqual, 1)
    //     aie.dma_bd(%buff_0 : memref<...>, 0, len)
    //     aie.use_lock(%cons_lock, Release, 1)
    //     aie.next_bd ^bd_1           // depth-1: loops back to ^bd_0
    //   ^bd_1:                        // depth > 1 only
    //     ...
    //     aie.next_bd ^bd_0
    //   ^end: aie.end
    //   }
    //
    // We only emit aie.mem for compute tiles (row > 1).  MemTile (row==1) and
    // shim (row==0) use memtile_dma and shim_dma_allocation respectively, which
    // are handled in Phases 4 and 5.
    //
    // Limitation: If a conduit participates in a link (its buffers/locks were
    // consumed by Phase 5), the aie.mem will still reference the same lock SSA
    // values.  For now we only generate aie.mem for conduits whose consumer
    // tile is a compute tile and that did not appear as the src of any link op
    // processed above.  Conduits that are link sources are skipped here because
    // the link phase already handles the DMA for those buffers.
    // -----------------------------------------------------------------------

    // Collect conduit names referenced as link sources (to skip them here).
    llvm::StringSet<> linkSrcNames;
    module.walk([&](ObjectFifoLink linkOp) {
      auto srcs = linkOp.getSrcs();
      for (auto s : srcs) {
        auto name = mlir::cast<mlir::StringAttr>(s).getValue();
        linkSrcNames.insert(name);
      }
    });

    for (auto &[name, info] : conduitMap) {
      if (info.buffers.empty() || !info.prodLock || !info.consLock)
        continue;

      // Skip if this conduit was a link source — the link phase already set up
      // the DMA for those buffers in a memtile_dma.
      if (linkSrcNames.count(name))
        continue;

      // Determine which tile gets the aie.mem and what DMA direction to use.
      //
      // Case A: shim producer → compute consumer.
      //   aie.mem on the CONSUMER tile, S2MM (receives data from shim).
      //   This is the common case handled originally.
      //
      // Case B: compute producer → shim consumer.
      //   aie.mem on the PRODUCER tile, MM2S (sends data to shim).
      //   In this case consumerTileCoords is empty; shimConsumerTileCoords
      //   has the shim tile.  The buffer/locks were allocated on the producer
      //   tile by Phase 3b.
      //   BD lock semantics for MM2S:
      //     acquire consLock (data filled, ready to send)
      //     dma_bd(buff, offset, len)
      //     release prodLock (slot freed for next fill)

      AIE::TileOp dmaHostTile;
      AIE::DMAChannelDir dmaDir;
      AIE::LockOp acqLock;  // lock to acquire before DMA transfer
      AIE::LockOp relLock;  // lock to release after DMA transfer

      bool isProducerToShim = info.consumerTileCoords.empty() &&
                              !info.shimConsumerTileCoords.empty();

      if (isProducerToShim) {
        // Case B: compute tile sends MM2S to shim.
        auto [prodCol, prodRow] = info.producerTileCoord;
        if (prodCol < 0 || prodRow == 0)
          continue;
        dmaHostTile = lookupTileByCoord(prodCol, prodRow);
        if (!dmaHostTile)
          continue;
        // Only compute tiles (row >= 2) use aie.mem.
        if (prodRow < 2)
          continue;
        dmaDir = AIE::DMAChannelDir::MM2S;
        // MM2S BD: wait for data filled (consLock), send, signal slot free (prodLock).
        acqLock = info.consLock;
        relLock = info.prodLock;
      } else {
        // Case A: shim sends S2MM to consumer compute tile (or compute→compute).
        if (info.consumerTileCoords.empty())
          continue;
        auto [consCol, consRow] = info.consumerTileCoords[0];
        dmaHostTile = lookupTileByCoord(consCol, consRow);
        if (!dmaHostTile)
          continue;
        // Only compute tiles (row >= 2) use aie.mem.
        if (consRow < 2)
          continue;
        dmaDir = AIE::DMAChannelDir::S2MM;
        // S2MM BD: wait for free slot (prodLock), receive, signal filled (consLock).
        acqLock = info.prodLock;
        relLock = info.consLock;
      }

      int64_t depth = info.depth > 0 ? info.depth : 1;
      int64_t perBufLen =
          info.capacity > 0 ? info.capacity / depth : 1;

      // Create aie.mem on the chosen tile.
      builder.setInsertionPoint(deviceBody.getTerminator());
      auto memOp = builder.create<AIE::MemOp>(deviceOp.getLoc(),
                                              dmaHostTile.getResult());
      mlir::Region &memRegion = memOp.getBody();

      auto addMemBlock = [&]() -> mlir::Block * {
        return builder.createBlock(&memRegion);
      };

      // ^dma_start_block: dma_start(dir, 0, ^bd_0, ^end)
      mlir::Block *dmaStartBlock = addMemBlock();

      // Create depth-many BD blocks.
      llvm::SmallVector<mlir::Block *> bdBlocks;
      for (int64_t i = 0; i < depth; ++i)
        bdBlocks.push_back(addMemBlock());

      // ^end: aie.end
      mlir::Block *endMemBlock = addMemBlock();

      // Fill dma_start_block.
      builder.setInsertionPointToEnd(dmaStartBlock);
      builder.create<AIE::DMAStartOp>(deviceOp.getLoc(),
                                     dmaDir,
                                     /*channel_index=*/static_cast<int32_t>(0),
                                     /*repeat_count=*/static_cast<int32_t>(0),
                                     bdBlocks[0], endMemBlock);

      // Fill BD blocks.
      for (int64_t i = 0; i < depth; ++i) {
        builder.setInsertionPointToEnd(bdBlocks[i]);
        builder.create<AIE::UseLockOp>(deviceOp.getLoc(),
                                       acqLock.getResult(),
                                       AIE::LockAction::AcquireGreaterEqual,
                                       static_cast<int32_t>(1));
        builder.create<AIE::DMABDOp>(deviceOp.getLoc(),
                                     info.buffers[i].getResult(),
                                     /*offset=*/0,
                                     static_cast<int>(perBufLen));
        builder.create<AIE::UseLockOp>(deviceOp.getLoc(),
                                       relLock.getResult(),
                                       AIE::LockAction::Release,
                                       static_cast<int32_t>(1));
        // Ring: chain to next BD block, wrapping back to first.
        mlir::Block *nextBD = bdBlocks[(i + 1) % depth];
        builder.create<AIE::NextBDOp>(deviceOp.getLoc(), nextBD);
      }

      // ^end: aie.end
      builder.setInsertionPointToEnd(endMemBlock);
      builder.create<AIE::EndOp>(deviceOp.getLoc());
    }

    // -----------------------------------------------------------------------
    // Phase 6: lower conduit.acquire / conduit.release inside func bodies
    //          → aie.use_lock
    //
    // Port semantics:
    //   port="Consume" acquire → AcquireGreaterEqual on consLock
    //   port="Produce" acquire → AcquireGreaterEqual on prodLock
    //   port="Consume" release → Release on prodLock (free up the prod slot)
    //   port="Produce" release → Release on consLock (signal data ready)
    // -----------------------------------------------------------------------

    // Lowering order (use-before-def safety):
    //   1. SubviewAccess  — uses Acquire's window result; erase immediately.
    //   2. Release        — uses Acquire's window result to find conduit name;
    //                       emit use_lock now, but defer erase until after
    //                       Acquire is erased (Release is a user of Acquire).
    //   3. Acquire        — emit use_lock; erase (no remaining users now).
    //   4. Erase Release ops collected in step 2.

    // Step 1: SubviewAccess — replace memref result with aie.buffer, then erase.
    module.walk([&](SubviewAccess op) {
      llvm::StringRef conduitName;
      if (auto acqOp = mlir::dyn_cast_or_null<Acquire>(
              op.getWindow().getDefiningOp()))
        conduitName = acqOp.getName();

      bool replaced = false;
      if (!conduitName.empty()) {
        auto it = conduitMap.find(conduitName);
        if (it != conduitMap.end() && !it->second.buffers.empty()) {
          int64_t idx = op.getIndex();
          if (idx < static_cast<int64_t>(it->second.buffers.size())) {
            mlir::Value bufVal = it->second.buffers[idx].getResult();
            if (bufVal.getType() == op.getResult().getType()) {
              op.getResult().replaceAllUsesWith(bufVal);
              replaced = true;
            }
          }
        }
      }
      // If the buffer could not be resolved, emit a hard error and fail the
      // pass.  A silent memref.alloc fallback would re-introduce the garbage-
      // memory correctness bug that Fix 1 was designed to eliminate.
      if (!replaced) {
        op.emitError("conduit-to-dma: SubviewAccess could not be resolved to "
                     "an allocated aie.buffer — type mismatch, index out of "
                     "range, or conduit not in map");
        return signalPassFailure();
      }
      op.erase();
    });

    // Step 2: Release — emit use_lock; collect for deferred erase.
    // Must happen BEFORE Acquire is erased because Release holds a reference
    // to Acquire's window result (needed to look up the conduit name).
    llvm::SmallVector<Release> releasesToErase;
    module.walk([&](Release op) {
      // conduit.release takes a !conduit.window<T> operand; derive the conduit
      // name by walking up to the defining conduit.acquire op.
      llvm::StringRef conduitName;
      if (auto acqOp = mlir::dyn_cast_or_null<Acquire>(
              op.getWindow().getDefiningOp()))
        conduitName = acqOp.getName();

      if (conduitName.empty()) {
        releasesToErase.push_back(op);
        return;
      }
      auto it = conduitMap.find(conduitName);
      if (it == conduitMap.end()) {
        releasesToErase.push_back(op);
        return;
      }
      ConduitInfo &cinfo = it->second;
      builder.setInsertionPoint(op);
      int64_t count = static_cast<int64_t>(op.getCount());
      llvm::StringRef port = op.getPort();
      // Consumer releases prod-lock (freeing up producer slots);
      // Producer releases cons-lock (signalling data ready for consumer).
      AIE::LockOp lock =
          (port == "Consume") ? cinfo.prodLock : cinfo.consLock;
      if (lock) {
        builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                       AIE::LockAction::Release,
                                       static_cast<int32_t>(count));
      }
      releasesToErase.push_back(op);
    });

    // Step 3: Erase Release ops BEFORE erasing Acquire.
    // Release holds a use of the Acquire's window SSA result.  Erasing
    // Acquire first while Release is still live causes "destroyed with uses".
    for (auto op : releasesToErase)
      op.erase();

    // Step 4: Acquire — emit use_lock; erase.
    // SubviewAccess (step 1) and Release (step 3) users are gone; safe to erase.
    module.walk([&](Acquire op) {
      auto it = conduitMap.find(op.getName());
      if (it == conduitMap.end()) {
        op.erase();
        return;
      }
      ConduitInfo &cinfo = it->second;
      builder.setInsertionPoint(op);
      int64_t count = static_cast<int64_t>(op.getCount());
      llvm::StringRef port = op.getPort();
      // Consumer acquires consume-lock; producer acquires produce-lock.
      AIE::LockOp lock =
          (port == "Produce") ? cinfo.prodLock : cinfo.consLock;
      if (lock) {
        builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                       AIE::LockAction::AcquireGreaterEqual,
                                       static_cast<int32_t>(count));
      }
      op.erase();
    });

    // -----------------------------------------------------------------------
    // Phase 7: erase remaining Conduit ops (create).
    // SubviewAccess and Acquire/Release were already erased in Phase 6.
    // conduit.annotate was removed from the dialect; no Annotate erase.
    // -----------------------------------------------------------------------

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
