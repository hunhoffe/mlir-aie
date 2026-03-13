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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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
  // Cyclostatic (CSDF) access pattern from conduit.create access_pattern attr.
  // Empty means uniform SDF (all acquires use the same count = capacity/depth).
  // Non-empty means CSDF: access_pattern[i % period] is the acquire count for
  // iteration i, where period = access_pattern.size().
  // Pass C uses this in Phase 6 (Acquire lowering) to emit the correct
  // per-iteration lock acquire count, and in Phase 5.5 (aie.mem BD chain) to
  // emit a BD block per pattern element rather than a uniform ring.
  llvm::SmallVector<int64_t> accessPattern;
  // Shared memory flag: set by Phase 3 when producer and consumer tiles are
  // adjacent and can share memory without DMA.
  //
  // When sharedMemory is true:
  //   - Buffers and locks are allocated on the PRODUCER tile (not consumer).
  //   - No aie.flow, no aie.dma_bd, no aie.mem, no aie.shim_dma_allocation.
  //   - Phase 5.5 (BD chain generation) is skipped.
  //   - Phase 6 (acquire/release lowering) still emits aie.use_lock; locks are
  //     on the producer tile but accessible from both producer and consumer cores
  //     via the shared memory interface.
  //   - The consumerTileLocks map is keyed on the CONSUMER tile SSA value (so
  //     Phase 6 can look them up from the consumer core), but the LockOp itself
  //     is on the producer tile.
  bool sharedMemory = false;
  // Hardware SSA values populated during lowering:
  llvm::SmallVector<AIE::BufferOp> buffers; // depth-many on consumer_tile[0]
  // Per-consumer-tile lock pairs for multi-consumer (broadcast) correctness.
  // Keyed on the tile SSA Value (result of aie.tile op).
  // Phase 3 populates this for every consumer tile; Phase 6 looks up the pair
  // for the tile that contains the acquire/release being lowered.
  // NOTE: for sharedMemory conduits the LockOps are physically on the PRODUCER
  // tile, but the key is still the consumer tile value for Phase 6 lookup.
  llvm::DenseMap<mlir::Value, std::pair<AIE::LockOp, AIE::LockOp>>
      consumerTileLocks; // tile → (prodLock, consLock)
  // Per-consumer-tile buffer vectors for SubviewAccess resolution.
  // Each entry holds depth-many BufferOps for that tile.
  // NOTE: for sharedMemory conduits the BufferOps are physically on the PRODUCER
  // tile, but the key is still the consumer tile value for Phase 6 lookup.
  llvm::DenseMap<mlir::Value, llvm::SmallVector<AIE::BufferOp>>
      consumerTileBuffers; // tile → [buff_0, ..., buff_{depth-1}]
  // Convenience accessors for the single-consumer case and for the link phase
  // (which always uses consumer_tile[0]).  Populated from consumerTileLocks[0].
  AIE::LockOp prodLock; // prod lock (init=depth)
  AIE::LockOp consLock; // cons lock (init=0)
  // For depth>1: a memref<1xi32> rotation counter on the consumer tile.
  // The core body loads this counter before each acquire to select the correct
  // ping-pong buffer (buff_{counter % depth}), then increments it after the
  // release.  Mirrors the stateful transform's buffer_0_N pattern.
  AIE::BufferOp rotationBuf;
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

      // Extract cyclostatic access pattern (CSDF).
      // getAccessPattern() returns std::optional<ArrayRef<int64_t>>.
      // When present, the consumer acquires access_pattern[i % period] elements
      // on iteration i instead of a uniform count.
      if (auto ap = op.getAccessPattern()) {
        for (int64_t v : *ap)
          info.accessPattern.push_back(v);
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

    // Obtain the target model for adjacency queries (isLegalMemAffinity).
    const AIE::AIETargetModel &targetModel = AIE::getTargetModel(deviceOp);

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
    // Pre-Phase 3: collect link source names before Phase 3 runs.
    //
    // Phase 5 (conduit.objectfifo_link lowering) allocates its OWN per-slice
    // lock pairs on the MemTile for link source conduits.  If Phase 3 also
    // allocates prodLock/consLock for those conduits (as it would for any
    // conduit with a consumer tile), the result is 2 orphan locks that waste
    // lock IDs and shift all subsequent lock indices.
    //
    // Fix: collect the set of conduit names that are link sources BEFORE
    // Phase 3 runs, then skip lock (and buffer) allocation for those conduits
    // inside Phase 3.  Phase 5 handles their lock allocation.
    //
    // Note: buffers on the MemTile consumer tile are still needed by Phase 5
    // (srcInfo.buffers is used for dma_bd references in the BD chain).  Only
    // the LOCKS are skipped; the buffers are still allocated by Phase 3.
    // -----------------------------------------------------------------------

    // Collect link source names for DISTRIBUTE links only.
    //
    // For a distribute link (1 src → N dsts), Phase 5 allocates its own
    // per-slice lock pairs on the MemTile.  Phase 3 would also allocate a
    // prodLock/consLock pair for the same conduit (as a consumer of the
    // MemTile), resulting in 2 orphan locks with wasted IDs.
    //
    // For a join link (N srcs → 1 dst), Phase 5 uses the EXISTING per-source
    // lock pairs that Phase 3 allocates.  Skipping Phase 3 locks for join
    // sources would leave Phase 5 with null lock references.
    //
    // Therefore, only skip Phase 3 lock allocation for distribute sources.
    llvm::StringSet<> linkSrcNamesEarly;
    module.walk([&](ObjectFifoLink linkOp) {
      if (linkOp.getMode() != "distribute")
        return; // join sources still need Phase 3 locks
      for (auto s : linkOp.getSrcs())
        linkSrcNamesEarly.insert(mlir::cast<mlir::StringAttr>(s).getValue());
    });

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

      // -----------------------------------------------------------------------
      // Phase 3c: Shared memory detection.
      //
      // If the producer and the single consumer are adjacent tiles (i.e.,
      // targetModel.isLegalMemAffinity() returns true in either direction),
      // the producer and consumer cores can access a common memory bank without
      // any DMA transfer.  In this case we allocate buffers and locks on the
      // PRODUCER tile (so the producer owns the buffer) and skip all DMA
      // setup — no aie.flow, no aie.dma_bd, no aie.mem.
      //
      // Conditions for shared memory path:
      //   1. Exactly one compute consumer (no multi-consumer broadcast, no
      //      shim consumers — those always require DMA).
      //   2. isLegalMemAffinity(prodCol, prodRow, consCol, consRow) OR
      //      isLegalMemAffinity(consCol, consRow, prodCol, prodRow).
      //   3. Not a link source (link ops always require DMA on the MemTile).
      //
      // The stateful transform uses the same check (requiresDMAs → isSharedMemory)
      // and allocates on the producer tile when share_direction != 1.
      // -----------------------------------------------------------------------
      if (info.consumerTileCoords.size() == 1 &&
          info.shimConsumerTileCoords.empty() &&
          !linkSrcNamesEarly.count(name)) {
        auto [prodCol, prodRow] = info.producerTileCoord;
        auto [consCol, consRow] = info.consumerTileCoords[0];
        // Neither tile may be a shim tile (row==0) or MemTile (row==1);
        // shared memory only applies between compute tiles.
        bool prodIsShim = (prodRow == 0);
        bool consIsShim = (consRow == 0);
        bool prodIsMemtile = targetModel.isMemTile(prodCol, prodRow);
        bool consIsMemtile = targetModel.isMemTile(consCol, consRow);
        if (!prodIsShim && !consIsShim && !prodIsMemtile && !consIsMemtile) {
          bool rightShared = targetModel.isLegalMemAffinity(
              prodCol, prodRow, consCol, consRow);
          bool leftShared = targetModel.isLegalMemAffinity(
              consCol, consRow, prodCol, prodRow);
          if (rightShared || leftShared) {
            // --- Shared memory path ---
            info.sharedMemory = true;

            AIE::TileOp prodTile = lookupTileByCoord(prodCol, prodRow);
            AIE::TileOp consTile = lookupTileByCoord(consCol, consRow);
            if (!prodTile || !consTile) {
              module.emitWarning(
                  "conduit-to-dma: shared memory conduit '" + name.str() +
                  "' has missing tile op; falling back to DMA path");
              info.sharedMemory = false;
              // Fall through to normal consumer loop below.
            } else {
              int64_t depth = info.depth > 0 ? info.depth : 1;
              mlir::Type bufTy = info.elemType;
              if (!bufTy) {
                int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
                bufTy = mlir::MemRefType::get({bufSize},
                                              mlir::IntegerType::get(ctx, 32));
              }

              if (insertAfterTile)
                builder.setInsertionPointAfter(insertAfterTile);
              else
                builder.setInsertionPointToStart(&deviceBody);

              mlir::Value prodTileVal = prodTile.getResult();
              mlir::Value consTileVal = consTile.getResult();

              // Allocate depth-many buffers on the PRODUCER tile.
              // Both producer and consumer cores can access these via the
              // shared memory interface (no DMA needed).
              llvm::SmallVector<AIE::BufferOp> sharedBuffers;
              for (int64_t i = 0; i < depth; ++i) {
                std::string symName =
                    name.str() + "_buff_" + std::to_string(i);
                auto buf = builder.create<AIE::BufferOp>(
                    deviceOp.getLoc(), bufTy, prodTileVal,
                    mlir::StringAttr::get(ctx, symName),
                    /*address=*/mlir::IntegerAttr{},
                    /*initial_value=*/mlir::ElementsAttr{},
                    /*mem_bank=*/mlir::IntegerAttr{});
                sharedBuffers.push_back(buf);
                info.buffers.push_back(buf);
              }

              // prod_lock on the PRODUCER tile (init=depth → free slots).
              AIE::LockOp sharedProdLock;
              {
                int lockIdx = lockIdCounter[prodTileVal]++;
                std::string symName = name.str() + "_prod_lock_0";
                AIE::LockOp lk = builder.create<AIE::LockOp>(
                    deviceOp.getLoc(), prodTileVal, lockIdx,
                    static_cast<int>(depth));
                lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                sharedProdLock = lk;
                info.prodLock = lk;
              }

              // cons_lock on the PRODUCER tile (init=0 → nothing filled yet).
              AIE::LockOp sharedConsLock;
              {
                int lockIdx = lockIdCounter[prodTileVal]++;
                std::string symName = name.str() + "_cons_lock_0";
                AIE::LockOp lk = builder.create<AIE::LockOp>(
                    deviceOp.getLoc(), prodTileVal, lockIdx,
                    static_cast<int>(0));
                lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                sharedConsLock = lk;
                info.consLock = lk;
              }

              // Register locks keyed on the CONSUMER tile so Phase 6
              // (acquire/release lowering inside the consumer aie.core) can
              // look them up by the tile SSA value of the core it walks into.
              info.consumerTileLocks[consTileVal] = {sharedProdLock,
                                                     sharedConsLock};
              // Register buffers keyed on consumer tile for SubviewAccess.
              info.consumerTileBuffers[consTileVal] = sharedBuffers;

              // Also register on producer tile for producer-side acquires.
              info.consumerTileLocks[prodTileVal] = {sharedProdLock,
                                                     sharedConsLock};
              info.consumerTileBuffers[prodTileVal] = sharedBuffers;

              // Rotation counter for depth>1 (on consumer tile — the consumer
              // core uses it, so it lives in the consumer's local memory).
              if (depth > 1) {
                auto counterTy = mlir::MemRefType::get(
                    {1}, mlir::IntegerType::get(ctx, 32));
                info.rotationBuf = builder.create<AIE::BufferOp>(
                    deviceOp.getLoc(), counterTy, consTileVal,
                    /*sym_name=*/mlir::StringAttr{},
                    /*address=*/mlir::IntegerAttr{},
                    /*initial_value=*/mlir::ElementsAttr{},
                    /*mem_bank=*/mlir::IntegerAttr{});
              }

              continue; // skip the normal DMA consumer loop for this conduit
            }
          }
        }
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

        // Skip lock allocation for link source conduits.
        //
        // Link source conduits (those that appear as srcs in a
        // conduit.objectfifo_link op) have their lock pairs allocated by
        // Phase 5 — one independent per-slice lock pair per destination.
        // Allocating locks here in Phase 3 would produce 2 orphan locks on
        // the MemTile that waste lock IDs and shift all Phase 5 lock indices.
        //
        // The buffers above are still needed: Phase 5 references
        // srcInfo.buffers for the dma_bd descriptors in the BD chain.
        if (linkSrcNamesEarly.count(name)) {
          // Link source: register buffers but no locks.
          // consumerTileLocks is left empty; Phase 5.5 skips this conduit.
          info.consumerTileBuffers[consTileVal] = consBuffers;
          continue; // advance to next consIdx (or exit the for loop)
        }

        // Allocate prod_lock (init = depth → depth free slots).
        // Named with _cons_ infix to distinguish from shim-tile prod_lock_0.
        AIE::LockOp thisProdLock;
        {
          int lockIdx = lockIdCounter[consTileVal]++;
          std::string symName =
              name.str() + bufSuffix + "_prod_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), consTileVal, lockIdx,
              static_cast<int>(depth));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          thisProdLock = lk;
          if (consIdx == 0)
            info.prodLock = lk;
        }

        // Allocate cons_lock (init = 0 → nothing filled yet).
        AIE::LockOp thisConsLock;
        {
          int lockIdx = lockIdCounter[consTileVal]++;
          std::string symName =
              name.str() + bufSuffix + "_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), consTileVal, lockIdx, 0);
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          thisConsLock = lk;
          if (consIdx == 0)
            info.consLock = lk;
        }

        // Register the lock pair for this consumer tile so Phase 6 can look
        // up the correct locks for cores on any consumer tile (not just [0]).
        info.consumerTileLocks[consTileVal] = {thisProdLock, thisConsLock};
        // Register the buffer vector for this consumer tile so Phase 6
        // SubviewAccess lowering can use the correct per-tile buffers.
        info.consumerTileBuffers[consTileVal] = consBuffers;

        // For depth>1, allocate a rotation counter buffer on the consumer tile.
        // The core body uses this to select the correct ping-pong buffer on
        // each iteration (load counter → index_switch → select buff_N).
        // Mirrors the stateful transform's `buffer_col_row : memref<1xi32>`.
        if (depth > 1 && consIdx == 0) {
          auto counterTy =
              mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));
          info.rotationBuf = builder.create<AIE::BufferOp>(
              deviceOp.getLoc(), counterTy, consTileVal,
              /*sym_name=*/mlir::StringAttr{},
              /*address=*/mlir::IntegerAttr{},
              /*initial_value=*/mlir::ElementsAttr{},
              /*mem_bank=*/mlir::IntegerAttr{});
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
      //
      // Fix NF6: broadcast correctness — emit one aie.flow per consumer tile.
      //
      // For a 1-producer → N-consumer broadcast, the shim DMA has N MM2S
      // channels (one per consumer).  Each consumer tile's DMA runs S2MM on
      // channel 0 and has its own independent buffer and lock pair (allocated
      // by Phase 3).  The stateful transform emits N flows; we must too.
      //
      // Single-consumer (N==1) case: channel index is 0 (unchanged from
      // the original implementation).  Multi-consumer (N>1): channel i sends
      // to consumer tile i via DMA channel i → DMA 0 on the consumer tile.
      if (prodRow == 0) {
        if (info.consumerTileCoords.empty())
          continue;

        AIE::TileOp shimTile = lookupTileByCoord(prodCol, prodRow);
        if (!shimTile)
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
        // Note: shim_dma_allocation always uses channel 0 — the allocation
        // registers the shim tile as a DMA endpoint; the per-consumer channel
        // routing is expressed via aie.flow ops below.
        std::string allocSym = name.str() + "_shim_alloc";
        builder.create<AIE::ShimDMAAllocationOp>(
            deviceOp.getLoc(), allocSym, shimTile.getResult(),
            AIE::DMAChannelDir::MM2S,
            /*channel_index=*/static_cast<int64_t>(0),
            /*plio=*/false,
            /*packet=*/nullptr);

        // Emit one aie.flow per consumer tile (broadcast fix NF6).
        // Channel index i on the shim MM2S side → channel 0 on consumer i.
        for (unsigned consIdx = 0;
             consIdx < info.consumerTileCoords.size(); ++consIdx) {
          auto [consCol, consRow] = info.consumerTileCoords[consIdx];
          AIE::TileOp consTile = lookupTileByCoord(consCol, consRow);
          if (!consTile)
            continue;
          builder.create<AIE::FlowOp>(
              deviceOp.getLoc(), shimTile.getResult(), AIE::WireBundle::DMA,
              static_cast<int32_t>(consIdx), consTile.getResult(),
              AIE::WireBundle::DMA, static_cast<int32_t>(0));
        }
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
    //   N MM2S channels each send their own slice to one dst.
    //
    //   FIXED (Defects 1, 2, 3): N independent lock pairs on the MemTile —
    //   one pair per destination slice.  The S2MM sequences through all N
    //   slices per buffer slot (depth*N total ingest BD blocks).  Each MM2S
    //   channel i uses its own independent cons_lock_i/prod_lock_i so the
    //   channels can run concurrently without racing.  aie.flow ops connect
    //   each MM2S channel i to the corresponding compute tile's DMA S2MM 0.
    //
    // For the join case (N srcs → 1 dst):
    //   N S2MM channels each ingest their own slice.
    //   One MM2S channel outputs the joined buffer.
    //   Lock protocol uses the src conduit's existing lock pair (simple ring).
    //
    // The BD chain pattern (distribute, depth=D, N destinations):
    //   aie.memtile_dma(%memtile) {
    //     %s = aie.dma_start(S2MM, 0, ^ingest_0_0, ^mm2s_chain_0)
    //   ^ingest_0_0:   // buff_0, slice 0
    //     use_lock(prod_lock_0, AcquireGreaterEqual, 1)
    //     dma_bd(buff_0, off_0, len_0)
    //     use_lock(cons_lock_0, Release, 1)
    //     next_bd ^ingest_0_1
    //   ^ingest_0_1:   // buff_0, slice 1
    //     use_lock(prod_lock_1, AcquireGreaterEqual, 1)
    //     dma_bd(buff_0, off_1, len_1)
    //     use_lock(cons_lock_1, Release, 1)
    //     next_bd ^ingest_0_{N-1}
    //   ...
    //   ^ingest_{D-1}_{N-1}:   // last buff, last slice
    //     ...
    //     next_bd ^ingest_0_0  // ring back to buff_0, slice_0
    //   ^mm2s_chain_0:
    //     %0 = aie.dma_start(MM2S, 0, ^send_0_0, ^mm2s_chain_1)
    //   ^send_0_0:   // MM2S ch 0, buff_0
    //     use_lock(cons_lock_0, AcquireGreaterEqual, 1)  // independent lock
    //     dma_bd(buff_0, off_0, len_0)
    //     use_lock(prod_lock_0, Release, 1)
    //     next_bd ^send_0_1
    //   ...
    //   ^mm2s_chain_1:
    //     %1 = aie.dma_start(MM2S, 1, ^send_1_0, ^mm2s_chain_2)
    //   ...
    //   ^end: aie.end
    //   }
    //
    //   aie.flow(memtile, DMA:0, compute_tile_0, DMA:0)
    //   aie.flow(memtile, DMA:1, compute_tile_1, DMA:0)
    //   ...
    // -----------------------------------------------------------------------

    // linkSrcNames: for Phase 5.5 (skip aie.mem for link sources), we need ALL
    // link source names (both distribute and join).  linkSrcNamesEarly only
    // contains distribute sources (to avoid skipping Phase 3 lock allocation
    // for join sources, which Phase 5 join mode needs).  Build a full set here.
    llvm::StringSet<> linkSrcNames;
    module.walk([&](ObjectFifoLink linkOp) {
      for (auto s : linkOp.getSrcs())
        linkSrcNames.insert(mlir::cast<mlir::StringAttr>(s).getValue());
    });

    module.walk([&](ObjectFifoLink linkOp) {
      builder.setInsertionPoint(deviceBody.getTerminator());
      mlir::Location loc = linkOp.getLoc();

      auto srcs = linkOp.getSrcs();
      auto dsts = linkOp.getDsts();
      llvm::StringRef memtileStr = linkOp.getMemtile();
      llvm::StringRef mode = linkOp.getMode();
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

      // Determine depth from the src conduit (clamped to at least 1).
      int64_t linkDepth = srcInfo.depth > 0 ? srcInfo.depth : 1;
      // Per-buffer element count (capacity / depth).
      int64_t perBufLen =
          srcInfo.capacity > 0 ? srcInfo.capacity / linkDepth : 1;

      // -----------------------------------------------------------------------
      // FIX — Defects 1 & 3: Per-destination independent lock pairs on the
      // MemTile for distribute mode.
      //
      // The stateful transform allocates 2*N locks on the MemTile for an
      // 1→N distribute link: lock pair [i*2, i*2+1] is dedicated to slice i.
      // This allows all N MM2S channels to operate independently without
      // racing on a shared lock — a shared lock causes deadlock when N > 1.
      //
      // We allocate the per-slice locks here, BEFORE creating the memtile_dma,
      // so the lock SSA values are available in both the S2MM and MM2S BD
      // bodies.
      //
      // For join mode (N→1), we use the existing srcInfo locks (single pair
      // is correct: all S2MM channels compete for the same buffer slot, so
      // one cons_lock and one prod_lock is the right protocol).
      // -----------------------------------------------------------------------

      bool isDistribute = (mode == "distribute");
      unsigned numDsts = static_cast<unsigned>(dsts.size());

      // Per-slice lock pairs for distribute.  Index i → (prod_lock_i, cons_lock_i).
      llvm::SmallVector<AIE::LockOp> sliceProdLocks; // init = depth (free slots)
      llvm::SmallVector<AIE::LockOp> sliceConsLocks; // init = 0 (nothing filled)

      mlir::Value memtileVal = memtile.getResult();

      if (isDistribute && numDsts > 0) {
        // Insert lock allocs before the memtile_dma so they dominate the DMA body.
        builder.setInsertionPoint(deviceBody.getTerminator());
        for (unsigned sliceIdx = 0; sliceIdx < numDsts; ++sliceIdx) {
          // prod_lock_i: producer (S2MM) acquires before filling slice i.
          // init = depth → depth free slots available at startup.
          {
            int lockIdx = lockIdCounter[memtileVal]++;
            std::string symName = srcName + "_link_prod_lock_" +
                                  std::to_string(sliceIdx);
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                deviceOp.getLoc(), memtileVal, lockIdx,
                static_cast<int>(linkDepth));
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            sliceProdLocks.push_back(lk);
          }
          // cons_lock_i: consumer (MM2S) acquires when slice i is filled.
          // init = 0 → nothing filled yet.
          {
            int lockIdx = lockIdCounter[memtileVal]++;
            std::string symName = srcName + "_link_cons_lock_" +
                                  std::to_string(sliceIdx);
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                deviceOp.getLoc(), memtileVal, lockIdx, 0);
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            sliceConsLocks.push_back(lk);
          }
        }
      }

      // -----------------------------------------------------------------------
      // FIX — Defect 2: Emit aie.flow from MemTile MM2S channel i to each
      // destination compute tile's DMA S2MM channel 0.
      //
      // The stateful transform emits one aie.flow per (producer, consumer) pair
      // in splitFifos.  For a distribute link, each dst conduit is a separate
      // consumer fifo whose producer is the MemTile; the flow connects
      // MemTile MM2S channel i → compute_tile_i DMA S2MM 0.
      //
      // Without these flows, the MM2S channels have no NoC routing and no data
      // reaches the compute tiles.
      // -----------------------------------------------------------------------

      if (isDistribute) {
        // Distribute: emit aie.flow from MemTile MM2S channel i to each
        // destination compute tile's DMA S2MM channel 0.
        builder.setInsertionPoint(deviceBody.getTerminator());
        for (unsigned dstIdx = 0; dstIdx < numDsts; ++dstIdx) {
          std::string dstName =
              mlir::cast<mlir::StringAttr>(dsts[dstIdx]).getValue().str();
          auto dstIt = conduitMap.find(dstName);
          if (dstIt == conduitMap.end() ||
              dstIt->second.consumerTileCoords.empty())
            continue;

          auto [dstConsCol, dstConsRow] =
              dstIt->second.consumerTileCoords[0];
          AIE::TileOp dstConsTile =
              lookupTileByCoord(dstConsCol, dstConsRow);
          if (!dstConsTile)
            continue;

          // aie.flow(%memtile, DMA : dstIdx, %compute_tile, DMA : 0)
          builder.create<AIE::FlowOp>(
              deviceOp.getLoc(), memtileVal, AIE::WireBundle::DMA,
              static_cast<int32_t>(dstIdx), dstConsTile.getResult(),
              AIE::WireBundle::DMA, static_cast<int32_t>(0));
        }
      } else {
        // Join mode: emit aie.flow from each source compute tile MM2S 0 to
        // the MemTile S2MM channel srcIdx.
        // Also emit aie.flow from MemTile MM2S 0 to the destination (shim).
        builder.setInsertionPoint(deviceBody.getTerminator());

        // Per-source flows: compute_tile_i DMA:0 → memtile DMA:srcIdx
        for (unsigned srcIdx = 0; srcIdx < srcs.size(); ++srcIdx) {
          std::string sName =
              mlir::cast<mlir::StringAttr>(srcs[srcIdx]).getValue().str();
          auto sIt = conduitMap.find(sName);
          if (sIt == conduitMap.end())
            continue;
          // The source conduit's PRODUCER tile is the compute tile.
          auto [srcProdCol, srcProdRow] = sIt->second.producerTileCoord;
          if (srcProdCol < 0 || srcProdRow == 0)
            continue;
          AIE::TileOp srcProdTile = lookupTileByCoord(srcProdCol, srcProdRow);
          if (!srcProdTile)
            continue;
          // aie.flow(%compute_tile, DMA : 0, %memtile, DMA : srcIdx)
          builder.create<AIE::FlowOp>(
              deviceOp.getLoc(), srcProdTile.getResult(),
              AIE::WireBundle::DMA, static_cast<int32_t>(0),
              memtileVal, AIE::WireBundle::DMA,
              static_cast<int32_t>(srcIdx));
        }

        // Destination flow: memtile MM2S 0 → dst shim consumer.
        if (!dsts.empty()) {
          std::string dstName =
              mlir::cast<mlir::StringAttr>(dsts[0]).getValue().str();
          auto dstIt = conduitMap.find(dstName);
          if (dstIt != conduitMap.end() &&
              !dstIt->second.shimConsumerTileCoords.empty()) {
            auto [shimCol, shimRow] =
                dstIt->second.shimConsumerTileCoords[0];
            AIE::TileOp shimTile = lookupTileByCoord(shimCol, shimRow);
            if (shimTile) {
              // aie.flow(%memtile, DMA : 0, %shim, DMA : 0)
              builder.create<AIE::FlowOp>(
                  deviceOp.getLoc(), memtileVal, AIE::WireBundle::DMA,
                  static_cast<int32_t>(0), shimTile.getResult(),
                  AIE::WireBundle::DMA, static_cast<int32_t>(0));
            }
          }
        }
      }

      // Create aie.memtile_dma block.
      // MemTileDMAOp builder: (Value tile) — result type inferred.
      builder.setInsertionPoint(deviceBody.getTerminator());
      auto memtileDMA =
          builder.create<AIE::MemTileDMAOp>(loc, memtileVal);
      mlir::Region &dmaRegion = memtileDMA.getBody();

      // Helper: create a new basic block in the DMA region.
      auto addBlock = [&]() -> mlir::Block * {
        return builder.createBlock(&dmaRegion);
      };

      // -----------------------------------------------------------------------
      // Build the S2MM (ingest) path.
      //
      // Distribute mode (1 src → N dsts):
      //   One S2MM channel (channel 0) ingests the full source buffer.
      //   The S2MM BD ring has depth*N blocks — each buffer slot is broken
      //   into N slice blocks (one per destination).
      //
      // Join mode (N srcs → 1 dst):
      //   N S2MM channels (channels 0..N-1), one per source.
      //   Each channel i has an independent depth-many BD ring referencing
      //   source_i's buffers and lock pair.  The S2MM channel starts are
      //   chained: S2MM_entry_0 → S2MM_entry_1 → ... → S2MM_entry_{N-1}
      //   → MM2S_chain.
      // -----------------------------------------------------------------------

      // The final "next chain" block — connects the last S2MM chain to MM2S.
      mlir::Block *mm2sChainStartBlock = nullptr;

      if (isDistribute) {
        // ===== Distribute: single S2MM entry, depth*numDsts BD ring =====

        mlir::Block *entryBlock = addBlock();

        // Create depth * numDsts ingest blocks.
        llvm::SmallVector<mlir::Block *> ingestBlocks;
        for (int64_t bufIdx = 0; bufIdx < linkDepth; ++bufIdx)
          for (unsigned sliceIdx = 0; sliceIdx < numDsts; ++sliceIdx)
            ingestBlocks.push_back(addBlock());

        mm2sChainStartBlock = addBlock();

        // ^entry: dma_start(S2MM, 0, ^ingest_first, ^mm2s_chain_start)
        builder.setInsertionPointToEnd(entryBlock);
        builder.create<AIE::DMAStartOp>(
            loc, AIE::DMAChannelDir::S2MM,
            /*channel_index=*/static_cast<int32_t>(0),
            /*repeat_count=*/static_cast<int32_t>(0),
            ingestBlocks[0], mm2sChainStartBlock);

        // Fill each ingest BD block.
        unsigned totalIngest = static_cast<unsigned>(linkDepth) * numDsts;
        for (unsigned blkIdx = 0; blkIdx < totalIngest; ++blkIdx) {
          unsigned bufIdx   = blkIdx / numDsts;
          unsigned sliceIdx = blkIdx % numDsts;

          int64_t dstOffset = 0;
          int64_t dstLen    = perBufLen;
          if (offsets.has_value() && !offsets->empty()) {
            if (sliceIdx < static_cast<unsigned>(offsets->size()))
              dstOffset = (*offsets)[sliceIdx];
            if (sliceIdx + 1 < static_cast<unsigned>(offsets->size()))
              dstLen = (*offsets)[sliceIdx + 1] - dstOffset;
            else
              dstLen = perBufLen - dstOffset;
          }

          builder.setInsertionPointToEnd(ingestBlocks[blkIdx]);
          builder.create<AIE::UseLockOp>(
              loc, sliceProdLocks[sliceIdx].getResult(),
              AIE::LockAction::AcquireGreaterEqual,
              static_cast<int32_t>(1));
          builder.create<AIE::DMABDOp>(
              loc, srcInfo.buffers[bufIdx].getResult(),
              static_cast<int>(dstOffset),
              static_cast<int>(dstLen));
          builder.create<AIE::UseLockOp>(
              loc, sliceConsLocks[sliceIdx].getResult(),
              AIE::LockAction::Release,
              static_cast<int32_t>(1));
          mlir::Block *nextIngest = ingestBlocks[(blkIdx + 1) % totalIngest];
          builder.create<AIE::NextBDOp>(loc, nextIngest);
        }

      } else {
        // ===== Join: N independent S2MM channels, one per source =====
        //
        // Chain structure (all blocks pre-allocated, then filled):
        //   ^s2mm_entry_0: dma_start(S2MM, 0, ^ingest_0_0, ^s2mm_entry_1)
        //   ^ingest_0_{0..depth-1}: BD ring for source 0
        //   ^s2mm_entry_1: dma_start(S2MM, 1, ^ingest_1_0, ^s2mm_entry_2)
        //   ...
        //   ^s2mm_entry_{N-1}: dma_start(S2MM, N-1, ^ingest_{N-1}_0, ^mm2s)
        //   ^ingest_{N-1}_{0..depth-1}: BD ring for source N-1
        //   ^mm2s (= mm2sChainStartBlock)

        unsigned numSrcs = static_cast<unsigned>(srcs.size());

        // Pre-allocate: N entry blocks + N * sDepth ingest blocks.
        llvm::SmallVector<mlir::Block *> s2mmEntries(numSrcs);
        // Per-source ingest block vectors (size sDepth_i each).
        llvm::SmallVector<llvm::SmallVector<mlir::Block *>> srcIngestBlocks(numSrcs);
        // Per-source info cache (to avoid re-lookup).
        llvm::SmallVector<ConduitInfo *> srcInfoPtrs(numSrcs, nullptr);
        llvm::SmallVector<int64_t> srcOffsets(numSrcs, 0);
        llvm::SmallVector<int64_t> srcLens(numSrcs, 1);

        for (unsigned srcIdx = 0; srcIdx < numSrcs; ++srcIdx) {
          s2mmEntries[srcIdx] = addBlock();
          std::string sName =
              mlir::cast<mlir::StringAttr>(srcs[srcIdx]).getValue().str();
          auto sIt = conduitMap.find(sName);
          if (sIt == conduitMap.end() || sIt->second.buffers.empty())
            continue;
          srcInfoPtrs[srcIdx] = &sIt->second;
          int64_t sDepth = sIt->second.depth > 0 ? sIt->second.depth : 1;
          for (int64_t i = 0; i < sDepth; ++i)
            srcIngestBlocks[srcIdx].push_back(addBlock());
          // Offsets for this source's slice.
          if (offsets.has_value() && !offsets->empty()) {
            if (srcIdx < static_cast<unsigned>(offsets->size()))
              srcOffsets[srcIdx] = (*offsets)[srcIdx];
            if (srcIdx + 1 < static_cast<unsigned>(offsets->size()))
              srcLens[srcIdx] = (*offsets)[srcIdx + 1] - srcOffsets[srcIdx];
            else
              srcLens[srcIdx] = (sIt->second.capacity > 0
                                     ? sIt->second.capacity / sDepth
                                     : 1) -
                                srcOffsets[srcIdx];
          } else {
            srcLens[srcIdx] = sIt->second.capacity > 0
                                  ? sIt->second.capacity / sDepth
                                  : 1;
          }
        }

        // The MM2S chain start block.
        mm2sChainStartBlock = addBlock();

        // Fill the entry and ingest blocks for each source.
        for (unsigned srcIdx = 0; srcIdx < numSrcs; ++srcIdx) {
          ConduitInfo *sInfo = srcInfoPtrs[srcIdx];
          if (!sInfo)
            continue;

          int64_t sDepth = sInfo->depth > 0 ? sInfo->depth : 1;
          int64_t srcOffset = srcOffsets[srcIdx];
          int64_t srcLen    = srcLens[srcIdx];

          // The next block after this source's S2MM chain.
          mlir::Block *nextBlock =
              (srcIdx + 1 < numSrcs) ? s2mmEntries[srcIdx + 1]
                                     : mm2sChainStartBlock;

          // ^s2mm_entry_srcIdx: dma_start(S2MM, srcIdx, ^ingest_srcIdx_0, ^next)
          builder.setInsertionPointToEnd(s2mmEntries[srcIdx]);
          builder.create<AIE::DMAStartOp>(
              loc, AIE::DMAChannelDir::S2MM,
              /*channel_index=*/static_cast<int32_t>(srcIdx),
              /*repeat_count=*/static_cast<int32_t>(0),
              srcIngestBlocks[srcIdx].empty() ? nextBlock
                                              : srcIngestBlocks[srcIdx][0],
              nextBlock);

          // Fill each ingest BD block for this source.
          for (int64_t i = 0; i < sDepth; ++i) {
            if (static_cast<size_t>(i) >= srcIngestBlocks[srcIdx].size())
              break;
            builder.setInsertionPointToEnd(srcIngestBlocks[srcIdx][i]);
            // Acquire: wait for a free slot from the source producer.
            // sInfo->prodLock is the lock allocated by Phase 3 for this source.
            if (sInfo->prodLock) {
              builder.create<AIE::UseLockOp>(
                  loc, sInfo->prodLock.getResult(),
                  AIE::LockAction::AcquireGreaterEqual,
                  static_cast<int32_t>(1));
            }
            // DMA descriptor: source buffer slice at srcOffset/srcLen.
            if (!sInfo->buffers.empty()) {
              builder.create<AIE::DMABDOp>(
                  loc,
                  sInfo->buffers[i % sInfo->buffers.size()].getResult(),
                  static_cast<int>(srcOffset),
                  static_cast<int>(srcLen));
            }
            // Release: signal this slot is filled (consLock).
            if (sInfo->consLock) {
              builder.create<AIE::UseLockOp>(
                  loc, sInfo->consLock.getResult(),
                  AIE::LockAction::Release,
                  static_cast<int32_t>(1));
            }
            // Ring: chain back to first ingest block for this source.
            mlir::Block *nextIngest =
                srcIngestBlocks[srcIdx][(i + 1) % sDepth];
            builder.create<AIE::NextBDOp>(loc, nextIngest);
          }
        }
      }

      // -----------------------------------------------------------------------
      // Build MM2S send chains — one chain per destination (dst) fifo.
      //
      // Distribute: each destination has its own MM2S channel with independent
      //   per-slice lock pair (sliceConsLocks[i] / sliceProdLocks[i]).
      // Join: single MM2S channel (channel 0) outputs the joined dst buffer.
      //   Uses the destination conduit's lock pair (link4's prodLock/consLock).
      // -----------------------------------------------------------------------

      // For join mode, look up the destination conduit's lock pair.
      AIE::LockOp joinDstProdLock;
      AIE::LockOp joinDstConsLock;
      llvm::SmallVector<AIE::BufferOp> *joinDstBuffers = nullptr;
      int64_t joinDstDepth = linkDepth;
      int64_t joinDstPerBufLen = perBufLen;
      if (!isDistribute && !dsts.empty()) {
        std::string dstName =
            mlir::cast<mlir::StringAttr>(dsts[0]).getValue().str();
        auto dstIt = conduitMap.find(dstName);
        if (dstIt != conduitMap.end()) {
          ConduitInfo &dstInfo = dstIt->second;
          joinDstProdLock = dstInfo.prodLock;
          joinDstConsLock = dstInfo.consLock;
          if (!dstInfo.buffers.empty())
            joinDstBuffers = &dstInfo.buffers;
          joinDstDepth = dstInfo.depth > 0 ? dstInfo.depth : 1;
          joinDstPerBufLen =
              dstInfo.capacity > 0 ? dstInfo.capacity / joinDstDepth : 1;
        }
      }

      mlir::Block *prevChainBlock = mm2sChainStartBlock
                                        ? mm2sChainStartBlock
                                        : addBlock();
      mlir::Block *endBlock = nullptr;

      for (unsigned dstIdx = 0; dstIdx < dsts.size(); ++dstIdx) {
        // For join: use dst conduit depth and per-buf-len (full joined buffer).
        int64_t thisDstDepth = isDistribute ? linkDepth : joinDstDepth;
        int64_t thisDstPerBufLen = isDistribute ? perBufLen : joinDstPerBufLen;

        // Compute slice offset and length for this destination.
        // For join mode: the MM2S sends the FULL joined buffer (offset=0, len=full).
        // For distribute mode: offset/len from the offsets array (slice).
        int64_t dstOffset = 0;
        int64_t dstLen = thisDstPerBufLen;
        if (isDistribute && offsets.has_value() && !offsets->empty()) {
          if (dstIdx < static_cast<unsigned>(offsets->size()))
            dstOffset = (*offsets)[dstIdx];
          if (dstIdx + 1 < static_cast<unsigned>(offsets->size()))
            dstLen = (*offsets)[dstIdx + 1] - dstOffset;
          else
            dstLen = perBufLen - dstOffset;
        }

        // Select the lock pair for this MM2S channel.
        AIE::LockOp mm2sAcqLock;
        AIE::LockOp mm2sRelLock;
        if (isDistribute && dstIdx < sliceConsLocks.size()) {
          // Distribute: per-slice independent locks.
          mm2sAcqLock = sliceConsLocks[dstIdx];
          mm2sRelLock = sliceProdLocks[dstIdx];
        } else {
          // Join: destination conduit's lock pair.
          mm2sAcqLock = joinDstConsLock;
          mm2sRelLock = joinDstProdLock;
        }

        // Select buffer source for the DMA BDs.
        // Distribute: srcInfo.buffers (the single source's buffers).
        // Join: joinDstBuffers (the destination conduit's buffers on the MemTile).
        llvm::SmallVector<AIE::BufferOp> *mm2sBufs =
            isDistribute ? &srcInfo.buffers
                         : (joinDstBuffers ? joinDstBuffers : &srcInfo.buffers);

        // Create depth-many send BD blocks for this dst channel.
        llvm::SmallVector<mlir::Block *> sendBDBlocks;
        for (int64_t i = 0; i < thisDstDepth; ++i)
          sendBDBlocks.push_back(addBlock());

        // The next chain block after this dst's chain.
        mlir::Block *nextChainBlock;
        if (dstIdx + 1 == dsts.size()) {
          endBlock = addBlock();
          nextChainBlock = endBlock;
        } else {
          nextChainBlock = addBlock();
        }

        // Fill prevChainBlock with dma_start(MM2S, dstIdx, ^bd_0, ^nextChain).
        builder.setInsertionPointToEnd(prevChainBlock);
        builder.create<AIE::DMAStartOp>(
            loc, AIE::DMAChannelDir::MM2S,
            /*channel_index=*/static_cast<int32_t>(dstIdx),
            /*repeat_count=*/static_cast<int32_t>(0), sendBDBlocks[0],
            nextChainBlock);

        // Fill each send BD block for this destination.
        for (int64_t i = 0; i < thisDstDepth; ++i) {
          builder.setInsertionPointToEnd(sendBDBlocks[i]);
          if (mm2sAcqLock) {
            builder.create<AIE::UseLockOp>(loc, mm2sAcqLock.getResult(),
                                           AIE::LockAction::AcquireGreaterEqual,
                                           static_cast<int32_t>(1));
          }
          if (mm2sBufs && !mm2sBufs->empty()) {
            builder.create<AIE::DMABDOp>(
                loc, (*mm2sBufs)[i % mm2sBufs->size()].getResult(),
                static_cast<int>(dstOffset),
                static_cast<int>(dstLen));
          }
          if (mm2sRelLock) {
            builder.create<AIE::UseLockOp>(loc, mm2sRelLock.getResult(),
                                           AIE::LockAction::Release,
                                           static_cast<int32_t>(1));
          }
          mlir::Block *nextBD = sendBDBlocks[(i + 1) % thisDstDepth];
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

    // linkSrcNames was populated before Phase 5 (above) so it correctly
    // contains the source conduit names even though Phase 5 has erased all
    // ObjectFifoLink ops by the time Phase 5.5 runs.

    for (auto &[name, info] : conduitMap) {
      if (info.buffers.empty() || !info.prodLock || !info.consLock)
        continue;

      // Skip if this conduit was a link source — the link phase already set up
      // the DMA for those buffers in a memtile_dma.
      if (linkSrcNames.count(name))
        continue;

      // Skip shared memory conduits: they require no DMA BD chain.
      // Buffers and locks on the producer tile are directly accessible from
      // both producer and consumer cores via shared memory; no aie.mem needed.
      if (info.sharedMemory)
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

        int64_t depth = info.depth > 0 ? info.depth : 1;
        int64_t perBufLen = info.capacity > 0 ? info.capacity / depth : 1;

        // Create aie.mem on the producer tile (single tile for Case B).
        builder.setInsertionPoint(deviceBody.getTerminator());
        auto memOp = builder.create<AIE::MemOp>(deviceOp.getLoc(),
                                                dmaHostTile.getResult());
        mlir::Region &memRegion = memOp.getBody();
        auto addMemBlock = [&]() -> mlir::Block * {
          return builder.createBlock(&memRegion);
        };
        mlir::Block *dmaStartBlock = addMemBlock();
        llvm::SmallVector<mlir::Block *> bdBlocks;
        for (int64_t i = 0; i < depth; ++i)
          bdBlocks.push_back(addMemBlock());
        mlir::Block *endMemBlock = addMemBlock();
        builder.setInsertionPointToEnd(dmaStartBlock);
        builder.create<AIE::DMAStartOp>(deviceOp.getLoc(), dmaDir,
                                        static_cast<int32_t>(0),
                                        static_cast<int32_t>(0),
                                        bdBlocks[0], endMemBlock);
        for (int64_t i = 0; i < depth; ++i) {
          builder.setInsertionPointToEnd(bdBlocks[i]);
          builder.create<AIE::UseLockOp>(deviceOp.getLoc(), acqLock.getResult(),
                                         AIE::LockAction::AcquireGreaterEqual,
                                         static_cast<int32_t>(1));
          builder.create<AIE::DMABDOp>(deviceOp.getLoc(),
                                       info.buffers[i].getResult(), 0,
                                       static_cast<int>(perBufLen));
          builder.create<AIE::UseLockOp>(deviceOp.getLoc(), relLock.getResult(),
                                         AIE::LockAction::Release,
                                         static_cast<int32_t>(1));
          builder.create<AIE::NextBDOp>(deviceOp.getLoc(),
                                        bdBlocks[(i + 1) % depth]);
        }
        builder.setInsertionPointToEnd(endMemBlock);
        builder.create<AIE::EndOp>(deviceOp.getLoc());

      } else {
        // Case A: shim sends S2MM to compute consumer tile(s).
        //
        // Fix NF6 (Phase 5.5): for broadcast (N consumer tiles), emit one
        // aie.mem per consumer tile, each using that tile's own buffer vector
        // and lock pair from consumerTileBuffers / consumerTileLocks.
        //
        // Single-consumer case: consumerTileCoords.size() == 1 → exactly one
        // aie.mem is emitted (same behavior as before this fix).
        if (info.consumerTileCoords.empty())
          continue;

        int64_t depth = info.depth > 0 ? info.depth : 1;
        int64_t perBufLen = info.capacity > 0 ? info.capacity / depth : 1;
        dmaDir = AIE::DMAChannelDir::S2MM;

        for (unsigned consIdx = 0; consIdx < info.consumerTileCoords.size();
             ++consIdx) {
          auto [consCol, consRow] = info.consumerTileCoords[consIdx];
          dmaHostTile = lookupTileByCoord(consCol, consRow);
          if (!dmaHostTile)
            continue;
          // Only compute tiles (row >= 2) use aie.mem.
          if (consRow < 2)
            continue;

          // Resolve the per-consumer-tile buffer vector and lock pair.
          // Falls back to info.buffers / prodLock / consLock for single-
          // consumer conduits where consumerTileBuffers may be empty.
          llvm::SmallVector<AIE::BufferOp> *tileBuffers = &info.buffers;
          AIE::LockOp tileProdLock = info.prodLock;
          AIE::LockOp tileConsLock = info.consLock;
          {
            mlir::Value consTileVal = dmaHostTile.getResult();
            auto bufIt = info.consumerTileBuffers.find(consTileVal);
            if (bufIt != info.consumerTileBuffers.end() &&
                !bufIt->second.empty())
              tileBuffers = &bufIt->second;
            auto lockIt = info.consumerTileLocks.find(consTileVal);
            if (lockIt != info.consumerTileLocks.end()) {
              tileProdLock = lockIt->second.first;
              tileConsLock = lockIt->second.second;
            }
          }

          if (!tileProdLock || !tileConsLock || tileBuffers->empty())
            continue;

          // S2MM BD: wait for free slot (prodLock), receive, signal filled (consLock).
          acqLock = tileProdLock;
          relLock = tileConsLock;

          // Create aie.mem on this consumer tile.
          builder.setInsertionPoint(deviceBody.getTerminator());
          auto memOp = builder.create<AIE::MemOp>(deviceOp.getLoc(),
                                                  dmaHostTile.getResult());
          mlir::Region &memRegion = memOp.getBody();
          auto addMemBlock = [&]() -> mlir::Block * {
            return builder.createBlock(&memRegion);
          };
          mlir::Block *dmaStartBlock = addMemBlock();
          llvm::SmallVector<mlir::Block *> bdBlocks;
          for (int64_t i = 0; i < depth; ++i)
            bdBlocks.push_back(addMemBlock());
          mlir::Block *endMemBlock = addMemBlock();
          builder.setInsertionPointToEnd(dmaStartBlock);
          builder.create<AIE::DMAStartOp>(deviceOp.getLoc(), dmaDir,
                                          static_cast<int32_t>(0),
                                          static_cast<int32_t>(0),
                                          bdBlocks[0], endMemBlock);
          for (int64_t i = 0; i < depth; ++i) {
            builder.setInsertionPointToEnd(bdBlocks[i]);
            builder.create<AIE::UseLockOp>(
                deviceOp.getLoc(), acqLock.getResult(),
                AIE::LockAction::AcquireGreaterEqual, static_cast<int32_t>(1));
            builder.create<AIE::DMABDOp>(
                deviceOp.getLoc(), (*tileBuffers)[i % tileBuffers->size()].getResult(),
                0, static_cast<int>(perBufLen));
            builder.create<AIE::UseLockOp>(
                deviceOp.getLoc(), relLock.getResult(),
                AIE::LockAction::Release, static_cast<int32_t>(1));
            builder.create<AIE::NextBDOp>(deviceOp.getLoc(),
                                          bdBlocks[(i + 1) % depth]);
          }
          builder.setInsertionPointToEnd(endMemBlock);
          builder.create<AIE::EndOp>(deviceOp.getLoc());
        }
      }
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
    //
    // For depth == 1: direct replacement with buffers[idx] (static selection).
    //
    // For depth > 1: emit a dynamic ping-pong selection using the rotation
    // counter buffer allocated in Phase 3.  The pattern mirrors the stateful
    // transform's counter + scf.index_switch approach:
    //
    //   %ctr_val = memref.load %rotation_buf[%c0]
    //   %ctr_idx = arith.index_cast %ctr_val : i32 to index
    //   // offset by the subview index (for sliding windows with count>1)
    //   %abs_idx = arith.addi %ctr_idx, %idx_const  // (% depth handled by switch)
    //   %selected = scf.index_switch %abs_idx -> memref<T>
    //     case 0 { scf.yield %buff_0 }
    //     case 1 { scf.yield %buff_1 }
    //     ...
    //     default { scf.yield %buff_0 }
    //
    // The counter increment (counter = (counter+1) % depth) is emitted in
    // Step 4 after the Release use_lock, once per acquire-release pair.
    //
    // Fix NF3: the window operand of SubviewAccess may come from either:
    //   a) conduit.acquire  (blocking path)  — look up name via acqOp.getName()
    //   b) conduit.wait_window (async path)  — look up name via waitOp.getName()
    // If neither matches, the pass emits a hard error (no silent fallback).
    module.walk([&](SubviewAccess op) {
      llvm::StringRef conduitName;
      if (auto acqOp = mlir::dyn_cast_or_null<Acquire>(
              op.getWindow().getDefiningOp()))
        conduitName = acqOp.getName();
      else if (auto waitOp = mlir::dyn_cast_or_null<WaitWindow>(
                   op.getWindow().getDefiningOp()))
        conduitName = waitOp.getName();

      bool replaced = false;
      if (!conduitName.empty()) {
        auto it = conduitMap.find(conduitName);
        if (it != conduitMap.end() && !it->second.buffers.empty()) {
          int64_t idx = op.getIndex();
          ConduitInfo &cinfo = it->second;
          int64_t depth = cinfo.depth > 0 ? cinfo.depth : 1;

          // Broadcast buffer correctness: resolve to the buffer vector for the
          // specific consumer tile that contains this SubviewAccess.
          // Falls back to cinfo.buffers (consumer_tile[0]) when not found.
          llvm::SmallVector<AIE::BufferOp> *tileBuffers = &cinfo.buffers;
          {
            mlir::Operation *coreOp = op->getParentOp();
            while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
              coreOp = coreOp->getParentOp();
            if (coreOp) {
              mlir::Value coreTile =
                  mlir::cast<AIE::CoreOp>(coreOp).getTile();
              auto bufIt = cinfo.consumerTileBuffers.find(coreTile);
              if (bufIt != cinfo.consumerTileBuffers.end())
                tileBuffers = &bufIt->second;
            }
          }

          {
            // Clamp idx to valid buffer range via modulo.
            // For sliding-window acquires (count > depth), idx may exceed
            // depth-1: e.g., acquire(count=3) on a depth=1 fifo produces
            // subview_access at indices 0, 1, 2 but only buffers[0] exists.
            // Wrapping with % depth ensures all indices resolve to a valid
            // buffer without an out-of-range crash.
            int64_t bufIdx = static_cast<int64_t>(tileBuffers->size()) > 1
                                 ? idx % static_cast<int64_t>(tileBuffers->size())
                                 : 0;
            if (depth == 1 || !cinfo.rotationBuf) {
              // Depth-1 case (or no rotation buffer): static selection.
              mlir::Value bufVal = (*tileBuffers)[bufIdx].getResult();
              if (bufVal.getType() == op.getResult().getType()) {
                op.getResult().replaceAllUsesWith(bufVal);
                replaced = true;
              }
            } else {
              // Depth>1 case: dynamic selection via counter + index_switch.
              builder.setInsertionPoint(op);
              mlir::Location loc = op.getLoc();

              // Load the rotation counter.
              mlir::Value c0Idx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
              mlir::Value ctrI32 = builder.create<mlir::memref::LoadOp>(
                  loc, cinfo.rotationBuf.getResult(),
                  mlir::ValueRange{c0Idx});

              // Cast to index for index_switch.
              mlir::Value ctrIdx = builder.create<mlir::arith::IndexCastOp>(
                  loc, builder.getIndexType(), ctrI32);

              // Add the subview index offset (for sliding windows idx>0).
              mlir::Value absIdx = ctrIdx;
              if (idx > 0) {
                mlir::Value idxConst =
                    builder.create<mlir::arith::ConstantIndexOp>(loc, idx);
                // Compute (counter + idx) % depth for correct wrap-around.
                mlir::Value sum = builder.create<mlir::arith::AddIOp>(
                    loc, ctrIdx, idxConst);
                mlir::Value depthConst =
                    builder.create<mlir::arith::ConstantIndexOp>(loc, depth);
                absIdx = builder.create<mlir::arith::RemUIOp>(loc, sum,
                                                               depthConst);
              }

              // Emit scf.index_switch over [0..depth-1].
              mlir::Type bufTy = op.getResult().getType();
              llvm::SmallVector<int64_t> caseVals;
              for (int64_t i = 0; i < depth; ++i)
                caseVals.push_back(i);

              auto switchOp = builder.create<mlir::scf::IndexSwitchOp>(
                  loc, mlir::TypeRange{bufTy}, absIdx, caseVals,
                  /*numRegions=*/static_cast<int>(depth));

              // Fill each case region: yield tileBuffers[i].
              for (int64_t i = 0; i < depth; ++i) {
                mlir::Block *caseBlock =
                    &switchOp.getCaseRegions()[i].emplaceBlock();
                mlir::OpBuilder caseBuilder(ctx);
                caseBuilder.setInsertionPointToEnd(caseBlock);
                caseBuilder.create<mlir::scf::YieldOp>(
                    loc, (*tileBuffers)[i].getResult());
              }
              // Default region: yield tileBuffers[0].
              {
                mlir::Block *defBlock =
                    &switchOp.getDefaultRegion().emplaceBlock();
                mlir::OpBuilder defBuilder(ctx);
                defBuilder.setInsertionPointToEnd(defBlock);
                defBuilder.create<mlir::scf::YieldOp>(
                    loc, (*tileBuffers)[0].getResult());
              }

              op.getResult().replaceAllUsesWith(switchOp.getResult(0));
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
    //
    // Fix NF3: the window operand of Release may come from either:
    //   a) conduit.acquire  (blocking path)  — look up name via acqOp.getName()
    //   b) conduit.wait_window (async path)  — look up name via waitOp.getName()
    llvm::SmallVector<Release> releasesToErase;
    module.walk([&](Release op) {
      // conduit.release takes a !conduit.window<T> operand; derive the conduit
      // name by walking up to the defining conduit.acquire (or wait_window) op.
      llvm::StringRef conduitName;
      if (auto acqOp = mlir::dyn_cast_or_null<Acquire>(
              op.getWindow().getDefiningOp()))
        conduitName = acqOp.getName();
      else if (auto waitOp = mlir::dyn_cast_or_null<WaitWindow>(
                   op.getWindow().getDefiningOp()))
        conduitName = waitOp.getName();

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

      // Broadcast lock correctness: find the enclosing aie.core tile and look
      // up that tile's specific lock pair from consumerTileLocks.  Falls back
      // to cinfo.prodLock/consLock (consumer_tile[0]) if the tile is not found
      // (e.g., producer-side release outside a core, or no multi-consumer map).
      AIE::LockOp resolvedProdLock = cinfo.prodLock;
      AIE::LockOp resolvedConsLock = cinfo.consLock;
      {
        mlir::Operation *coreOp = op->getParentOp();
        while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
          coreOp = coreOp->getParentOp();
        if (coreOp) {
          mlir::Value coreTile = mlir::cast<AIE::CoreOp>(coreOp).getTile();
          auto lockIt = cinfo.consumerTileLocks.find(coreTile);
          if (lockIt != cinfo.consumerTileLocks.end()) {
            resolvedProdLock = lockIt->second.first;
            resolvedConsLock = lockIt->second.second;
          }
        }
      }

      // Consumer releases prod-lock (freeing up producer slots);
      // Producer releases cons-lock (signalling data ready for consumer).
      AIE::LockOp lock =
          (port == "Consume") ? resolvedProdLock : resolvedConsLock;
      if (lock) {
        builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                       AIE::LockAction::Release,
                                       static_cast<int32_t>(count));
      }
      // For depth>1 with Consume port, emit counter increment after the
      // release use_lock.  Counter advances by 'count' (the number of
      // elements released) mod depth, matching the stateful transform pattern.
      if (cinfo.rotationBuf && port == "Consume" && cinfo.depth > 1) {
        mlir::Location loc = op.getLoc();
        mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
        mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        // Load current counter.
        mlir::Value curI32 = builder.create<mlir::memref::LoadOp>(
            loc, cinfo.rotationBuf.getResult(), mlir::ValueRange{c0});
        // Increment by count.
        mlir::Value incI32 = mlir::arith::ConstantIntOp::create(
            builder, loc, i32Ty, count);
        mlir::Value newVal =
            builder.create<mlir::arith::AddIOp>(loc, curI32, incI32);
        // Wrap modulo depth: if newVal >= depth, subtract depth.
        mlir::Value depthI32 = mlir::arith::ConstantIntOp::create(
            builder, loc, i32Ty, cinfo.depth);
        mlir::Value cmp = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sge, newVal, depthI32);
        mlir::Value wrapped =
            builder.create<mlir::arith::SubIOp>(loc, newVal, depthI32);
        mlir::Value result =
            builder.create<mlir::arith::SelectOp>(loc, cmp, wrapped, newVal);
        builder.create<mlir::memref::StoreOp>(
            loc, result, cinfo.rotationBuf.getResult(), mlir::ValueRange{c0});
      }
      releasesToErase.push_back(op);
    });

    // Step 3: Erase Release ops BEFORE erasing Acquire.
    // Release holds a use of the Acquire's window SSA result.  Erasing
    // Acquire first while Release is still live causes "destroyed with uses".
    for (auto op : releasesToErase)
      op.erase();

    // Step 4: Acquire — emit use_lock (and counter init on first use); erase.
    // SubviewAccess (step 1) and Release (step 3) users are gone; safe to erase.
    //
    // Counter initialization: the rotation counter must be set to 0 before the
    // first acquire for each depth>1 conduit in each core.  Track which
    // (conduit, parent-op) pairs have been initialized to emit the store once.
    llvm::DenseSet<std::pair<llvm::StringRef, mlir::Operation *>>
        counterInitialized;

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

      // Broadcast lock correctness: resolve the lock pair for the specific
      // consumer tile that contains this acquire, using consumerTileLocks.
      AIE::LockOp resolvedProdLock = cinfo.prodLock;
      AIE::LockOp resolvedConsLock = cinfo.consLock;
      {
        mlir::Operation *coreOp = op->getParentOp();
        while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
          coreOp = coreOp->getParentOp();
        if (coreOp) {
          mlir::Value coreTile = mlir::cast<AIE::CoreOp>(coreOp).getTile();
          auto lockIt = cinfo.consumerTileLocks.find(coreTile);
          if (lockIt != cinfo.consumerTileLocks.end()) {
            resolvedProdLock = lockIt->second.first;
            resolvedConsLock = lockIt->second.second;
          }
        }
      }

      // Consumer acquires consume-lock; producer acquires produce-lock.
      AIE::LockOp lock =
          (port == "Produce") ? resolvedProdLock : resolvedConsLock;

      // For depth>1 Consume acquires: initialize the rotation counter to 0
      // at the start of the enclosing aie.core body (once per core+conduit).
      if (cinfo.rotationBuf && port == "Consume" && cinfo.depth > 1) {
        // Walk up to find the enclosing aie.core op; the acquire may be
        // nested inside an scf.for or other region within the core.
        mlir::Operation *coreOp = op->getParentOp();
        while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
          coreOp = coreOp->getParentOp();
        auto key = std::make_pair(op.getName(), coreOp);
        if (!counterInitialized.count(key)) {
          counterInitialized.insert(key);
          // Insert at the very start of the core body's entry block, before
          // any acquire-related ops.
          mlir::Block *entryBlock = &coreOp->getRegion(0).front();
          mlir::OpBuilder initBuilder(entryBlock, entryBlock->begin());
          mlir::Location loc = op.getLoc();
          mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
          mlir::Value zero = mlir::arith::ConstantIntOp::create(
              initBuilder, loc, i32Ty, 0);
          mlir::Value c0 =
              initBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
          initBuilder.create<mlir::memref::StoreOp>(
              loc, zero, cinfo.rotationBuf.getResult(),
              mlir::ValueRange{c0});
        }
      }

      if (lock) {
        builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                       AIE::LockAction::AcquireGreaterEqual,
                                       static_cast<int32_t>(count));
      }
      op.erase();
    });

    // -----------------------------------------------------------------------
    // Phase 7: erase remaining Conduit ops (create, wait).
    // SubviewAccess and Acquire/Release were already erased in Phase 6.
    // conduit.annotate was removed from the dialect; no Annotate erase.
    //
    // Fix NF5: conduit.wait ops (used as DMA token synchronization fences)
    // are not lowered to any hardware op — the program ordering they encode
    // is already captured by the surrounding lock acquire/release sequence.
    // They must be erased here so they do not survive into the output IR as
    // dangling Conduit ops.  Erase conduit.wait BEFORE conduit.create so
    // that any wait ops inside device body are caught first.
    // -----------------------------------------------------------------------

    module.walk([&](Wait op) { op.erase(); });
    module.walk([&](Create op) { op.erase(); });

    // -----------------------------------------------------------------------
    // Phase 8: lower conduit.acquire_async / conduit.wait_window / conduit.wait_all
    //
    // The async acquire path enables DMA/compute overlap:
    //
    //   %tok = conduit.acquire_async {name="x", count=1, port="Consume"}
    //                                 : !conduit.async.token
    //   // ... DMA or other async work here ...
    //   %win = conduit.wait_window %tok for "x"
    //                               : !conduit.async.token -> !conduit.window<T>
    //   // use %win via conduit.subview_access ...
    //
    // Lowering:
    //   acquire_async → emit NOTHING (deferred to wait_window)
    //   wait_window   → emit aie.use_lock(consLock, AcquireGreaterEqual, count)
    //                    and replace the window result with the buffer SSA value
    //   wait_all      → for each acquire_async token: emit use_lock
    //
    // Moving the use_lock later in the instruction stream gives the DMA more
    // time to complete before the core reaches the lock acquire — the key
    // performance claim for the async path.
    //
    // Step 8a: Collect acquire_async metadata BEFORE erasing the ops.
    // We record (token SSA value → {conduit name, port, count}) so that
    // wait_window and wait_all can look up the lock info after the acquire_async
    // is gone.
    // -----------------------------------------------------------------------

    struct AsyncAcquireInfo {
      std::string conduitName;
      std::string port;
      int64_t count;
    };
    llvm::DenseMap<mlir::Value, AsyncAcquireInfo> asyncAcquireMap;

    // Step 8a: Record acquire_async metadata (do NOT erase yet).
    // The AcquireAsync op must stay alive until after WaitWindow is processed,
    // because WaitWindow's getToken() SSA operand points to the AcquireAsync
    // result.  Erasing AcquireAsync before WaitWindow causes use-after-erase.
    // We only erase AcquireAsync ops in Step 8a-erase, after WaitWindow is done.
    llvm::SmallVector<AcquireAsync> asyncAcquiresToErase;
    module.walk([&](AcquireAsync op) {
      AsyncAcquireInfo info;
      info.conduitName = op.getName().str();
      // acquire_async has no port attribute — it always represents a consumer
      // acquire (waiting for data to become available for reading).
      info.port = "Consume";
      info.count = static_cast<int64_t>(op.getCount());
      asyncAcquireMap[op.getToken()] = info;
      asyncAcquiresToErase.push_back(op);
    });

    // Step 8b: Lower conduit.wait_window.
    // For each wait_window, the token's defining acquire_async recorded the
    // conduit name.  Emit use_lock at this point (the actual hardware wait),
    // then replace the window result with the allocated aie.buffer SSA value.
    llvm::SmallVector<WaitWindow> waitWindowsToErase;
    module.walk([&](WaitWindow op) {
      // The conduit name is an attribute on the wait_window op itself.
      llvm::StringRef conduitName = op.getName();
      auto it = conduitMap.find(conduitName);
      if (it == conduitMap.end()) {
        op.emitError("conduit-to-dma: wait_window references unknown conduit '")
            << conduitName << "'";
        signalPassFailure();
        return;
      }
      ConduitInfo &cinfo = it->second;

      // Determine port and count from the async acquire map (keyed by token).
      llvm::StringRef port = "Consume";
      int64_t count = 1;
      {
        auto ait = asyncAcquireMap.find(op.getToken());
        if (ait != asyncAcquireMap.end()) {
          port = ait->second.port;
          count = ait->second.count;
        }
      }

      // Resolve the lock pair for the core that contains this wait_window.
      AIE::LockOp resolvedProdLock = cinfo.prodLock;
      AIE::LockOp resolvedConsLock = cinfo.consLock;
      {
        mlir::Operation *coreOp = op->getParentOp();
        while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
          coreOp = coreOp->getParentOp();
        if (coreOp) {
          mlir::Value coreTile = mlir::cast<AIE::CoreOp>(coreOp).getTile();
          auto lockIt = cinfo.consumerTileLocks.find(coreTile);
          if (lockIt != cinfo.consumerTileLocks.end()) {
            resolvedProdLock = lockIt->second.first;
            resolvedConsLock = lockIt->second.second;
          }
        }
      }

      // Emit the deferred use_lock at this point in the instruction stream.
      builder.setInsertionPoint(op);
      AIE::LockOp lock = (port == "Produce") ? resolvedProdLock : resolvedConsLock;
      if (lock) {
        builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                       AIE::LockAction::AcquireGreaterEqual,
                                       static_cast<int32_t>(count));
      }

      // Replace the window result with the appropriate aie.buffer SSA value.
      // Same logic as Phase 6 Step 1 (SubviewAccess): use per-tile buffer map.
      llvm::SmallVector<AIE::BufferOp> *tileBuffers = &cinfo.buffers;
      {
        mlir::Operation *coreOp = op->getParentOp();
        while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
          coreOp = coreOp->getParentOp();
        if (coreOp) {
          mlir::Value coreTile = mlir::cast<AIE::CoreOp>(coreOp).getTile();
          auto bufIt = cinfo.consumerTileBuffers.find(coreTile);
          if (bufIt != cinfo.consumerTileBuffers.end())
            tileBuffers = &bufIt->second;
        }
      }

      // The window result type wraps the element memref — replace all uses of
      // the window with the buffer.  Users will be conduit.subview_access ops
      // which Phase 6 Step 1 will have already replaced with aie.buffer refs;
      // any remaining users get the first buffer as a fallback.
      if (!tileBuffers->empty() && op.getResult().use_empty() == false) {
        // If the window SSA value has surviving users (e.g., conduit.subview_access
        // that Phase 6 missed), replace with buffer[0].
        mlir::Value bufVal = (*tileBuffers)[0].getResult();
        if (bufVal.getType() == op.getResult().getType()) {
          op.getResult().replaceAllUsesWith(bufVal);
        }
      }

      waitWindowsToErase.push_back(op);
    });
    for (auto op : waitWindowsToErase)
      op.erase();

    // Step 8c: Lower conduit.wait_all containing acquire_async tokens.
    // Must happen BEFORE erasing AcquireAsync (Step 8a-erase) because
    // WaitAll holds SSA uses of the AcquireAsync token results.
    llvm::SmallVector<WaitAll> waitAllToErase;
    module.walk([&](WaitAll op) {
      builder.setInsertionPoint(op);
      for (mlir::Value tok : op.getTokens()) {
        auto ait = asyncAcquireMap.find(tok);
        if (ait == asyncAcquireMap.end())
          continue; // not an acquire_async token — skip (DMA token, etc.)

        const AsyncAcquireInfo &ainfo = ait->second;
        auto it = conduitMap.find(ainfo.conduitName);
        if (it == conduitMap.end())
          continue;
        ConduitInfo &cinfo = it->second;

        // Resolve per-tile locks.
        AIE::LockOp resolvedProdLock = cinfo.prodLock;
        AIE::LockOp resolvedConsLock = cinfo.consLock;
        {
          mlir::Operation *coreOp = op->getParentOp();
          while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
            coreOp = coreOp->getParentOp();
          if (coreOp) {
            mlir::Value coreTile = mlir::cast<AIE::CoreOp>(coreOp).getTile();
            auto lockIt = cinfo.consumerTileLocks.find(coreTile);
            if (lockIt != cinfo.consumerTileLocks.end()) {
              resolvedProdLock = lockIt->second.first;
              resolvedConsLock = lockIt->second.second;
            }
          }
        }

        AIE::LockOp lock =
            (ainfo.port == "Produce") ? resolvedProdLock : resolvedConsLock;
        if (lock) {
          builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                         AIE::LockAction::AcquireGreaterEqual,
                                         static_cast<int32_t>(ainfo.count));
        }
      }
      waitAllToErase.push_back(op);
    });
    for (auto op : waitAllToErase)
      op.erase();

    // Step 8a-erase: NOW safe to erase AcquireAsync ops.
    // WaitWindow (Step 8b) and WaitAll (Step 8c) are both gone, so no op
    // holds a use of the AcquireAsync token SSA value anymore.
    for (auto op : asyncAcquiresToErase)
      op.erase();

    // Step 8d: Hard error on any remaining async ops not handled above.
    // conduit.release_async is not yet lowered.
    // conduit.wait_all_async is not yet lowered.
    module.walk([&](ReleaseAsync op) {
      op.emitError("conduit-to-dma: unimplemented — conduit.release_async "
                   "lowering not yet supported");
      signalPassFailure();
    });
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
