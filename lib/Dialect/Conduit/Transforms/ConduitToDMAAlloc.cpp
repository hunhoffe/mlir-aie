//===- ConduitToDMAAlloc.cpp - Phase 3: buffer + lock allocation --*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Phase 3: For each conduit.create, allocate aie.buffer + aie.lock pairs
// in the aie.device body.
//
// Sub-phases:
//   3b: Shim consumer conduits (buffer/lock on producer tile)
//   3c: Shared memory detection (adjacent tiles, no DMA)
//   3j: Join source conduits (buffer/lock on producer compute tile)
//   3d: Non-adjacent compute→compute producer-side allocation
//   Consumer tile allocation (broadcast: per-tile buffers+locks)
//
// Rotation counter packing:
//   Multiple conduits on the same tile share a single memref<N xi32> counter
//   buffer (matching the oracle's allocation). A pre-scan pass counts how many
//   counter slots each tile needs; one shared buffer is created per tile and
//   each conduit is assigned a slot index within it.
//
//===----------------------------------------------------------------------===//

#include "ConduitToDMACommon.h"

namespace xilinx::conduit {

// ---------------------------------------------------------------------------
// Helper: assign the next rotation counter slot for a tile.
// Returns the slot index. The shared buffer for the tile must already exist
// in state.tileRotationBuf (populated by the pre-scan pass).
// ---------------------------------------------------------------------------
static int64_t assignRotationSlot(ConduitToDMAState &state,
                                  mlir::Value tileVal) {
  int64_t slot = state.tileRotationBufNextSlot[tileVal]++;
  return slot;
}

// ---------------------------------------------------------------------------
// Pre-scan: count how many rotation counter slots each tile needs, then
// create one shared memref<N xi32> buffer per tile.
//
// This mirrors exactly the conditions checked during the main allocation pass
// so that the shared buffer has exactly the right size.
//
// Overcount-safe invariant:
//   (1) conduitMap is iterated in insertion order; both this prescan and the
//       main allocPhase loop MUST traverse it in the same order so that
//       assignRotationSlot() calls happen in the same sequence and slot
//       indices remain consistent.
//   (2) info.sharedMemory is always false during prescan (it is set during
//       the main Phase 3c pass which runs after prescan).  The Phase 3d loop
//       therefore may overcount slots for conduits that will ultimately take
//       the shared-memory path.  This is safe — the buffer will be slightly
//       too large but never too small, so no slot index is out of bounds.
//   (3) info.consumerTileBuffers is always empty during prescan, so the
//       `info.consumerTileBuffers.count(...)` guards in the normal consumer
//       loop and Phase 3d loop are vacuously false and every eligible conduit
//       is counted.  This is also an overcount-safe case for the same reason.
//   (4) assignRotationSlot must NEVER be called more times than this prescan
//       counted for a given tile.  Violating this would exhaust the pre-sized
//       buffer and produce out-of-bounds slot indices.
// ---------------------------------------------------------------------------
static void prescanAndCreateRotationBufs(ConduitToDMAState &state) {
  mlir::OpBuilder &builder = *state.builder;
  mlir::MLIRContext *ctx = state.ctx;
  const AIE::AIETargetModel &targetModel = *state.targetModel;

  // Per-tile slot counts (consumer + producer counters).
  // MapVector preserves insertion order for deterministic IR output.
  llvm::MapVector<mlir::Value, int64_t> tileSlotCount;

  // Consumer and producer counters share the same per-tile pool; two names
  // for call-site clarity only.
  auto addConsumerSlot = [&](mlir::Value tileVal) { tileSlotCount[tileVal]++; };
  auto addProducerSlot = [&](mlir::Value tileVal) { tileSlotCount[tileVal]++; };

  for (auto &[name, info] : state.conduitMap) {
    if (info.consumerTileCoords.empty() && info.shimConsumerTileCoords.empty())
      continue;

    // Phase 3b path.
    if (info.consumerTileCoords.empty() &&
        !info.shimConsumerTileCoords.empty()) {
      if (state.linkDstNames.count(name))
        continue;
      auto [prodCol, prodRow] = info.producerTileCoord;
      if (prodCol < 0 || prodRow == 0)
        continue;
      AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
      if (!prodTile)
        continue;
      int64_t depth = info.depth > 0 ? info.depth : 1;
      int64_t prodDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
      if (prodDepth > 1 && state.conduitNamesWithProducerAcquire.count(name))
        addProducerSlot(prodTile.getResult());
      continue;
    }

    // Phase 3c path (shared memory, adjacent tiles).
    if (!info.viaDMA && info.consumerTileCoords.size() == 1 &&
        info.shimConsumerTileCoords.empty() &&
        !state.linkSrcNamesEarly.count(name) &&
        !state.linkJoinSrcNames.count(name)) {
      auto [prodCol, prodRow] = info.producerTileCoord;
      auto [consCol, consRow] = info.consumerTileCoords[0];
      bool prodIsShim = (prodRow == 0);
      bool consIsShim = (consRow == 0);
      bool prodIsMemtile = targetModel.isMemTile(prodCol, prodRow);
      bool consIsMemtile = targetModel.isMemTile(consCol, consRow);
      if (!prodIsShim && !consIsShim && !prodIsMemtile && !consIsMemtile) {
        bool rightShared =
            targetModel.isLegalMemAffinity(prodCol, prodRow, consCol, consRow);
        bool leftShared =
            targetModel.isLegalMemAffinity(consCol, consRow, prodCol, prodRow);
        if (rightShared || leftShared) {
          AIE::TileOp allocTile = state.lookupTileByCoord(prodCol, prodRow);
          AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
          AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
          if (info.hasAllocTile) {
            allocTile = state.lookupTileByCoord(info.allocTileCoord.first,
                                                info.allocTileCoord.second);
          }
          if (allocTile && consTile && prodTile) {
            int64_t depth = info.depth > 0 ? info.depth : 1;
            if (depth > 1 && state.conduitNamesWithConsumerAcquire.count(name))
              addConsumerSlot(consTile.getResult());
            if (depth > 1 && state.conduitNamesWithProducerAcquire.count(name))
              addProducerSlot(prodTile.getResult());
            continue;
          }
        }
      }
    }

    // Phase 3j path (join sources).
    if (state.linkJoinSrcNames.count(name)) {
      auto [prodCol, prodRow] = info.producerTileCoord;
      if (prodCol < 0 || prodRow < 2)
        continue;
      AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
      if (!prodTile)
        continue;
      int64_t depth = info.depth > 0 ? info.depth : 1;
      int64_t prodDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
      if (prodDepth > 1 && state.conduitNamesWithProducerAcquire.count(name))
        addProducerSlot(prodTile.getResult());
      continue;
    }

    // Normal consumer loop path.
    int64_t depth = info.depth > 0 ? info.depth : 1;
    for (unsigned consIdx = 0; consIdx < info.consumerTileCoords.size();
         ++consIdx) {
      auto [consCol, consRow] = info.consumerTileCoords[consIdx];
      AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
      if (!consTile)
        continue;
      mlir::Value consTileVal = consTile.getResult();

      if (state.linkSrcNamesEarly.count(name)) {
        // linkSrcNamesEarly: producer-side counter on compute producer tile.
        auto [pCol, pRow] = info.producerTileCoord;
        if (pCol >= 0 && pRow >= 2) {
          AIE::TileOp pTile = state.lookupTileByCoord(pCol, pRow);
          if (pTile && !info.consumerTileBuffers.count(pTile.getResult())) {
            int64_t prodDepth =
                info.effectiveDepth > 0 ? info.effectiveDepth : depth;
            if (prodDepth > 1 &&
                state.conduitNamesWithProducerAcquire.count(name))
              addProducerSlot(pTile.getResult());
          }
        }
        continue;
      }

      // Regular consumer tile rotation counter.
      if (depth > 1 && state.conduitNamesWithConsumerAcquire.count(name))
        addConsumerSlot(consTileVal);
    }
  }

  // Phase 3d: producer-side counters for non-adjacent compute→compute.
  for (auto &[name, info] : state.conduitMap) {
    if (info.sharedMemory)
      continue;
    if (state.linkSrcNamesEarly.count(name) ||
        state.linkJoinSrcNames.count(name))
      continue;
    if (state.linkDstNames.count(name))
      continue;
    auto [prodCol, prodRow] = info.producerTileCoord;
    if (prodCol < 0 || prodRow == 0)
      continue;
    if (info.consumerTileCoords.empty())
      continue;
    AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
    if (!prodTile)
      continue;
    mlir::Value prodTileVal = prodTile.getResult();
    if (info.consumerTileBuffers.count(prodTileVal))
      continue;

    bool needsProdSide = false;
    if (info.consumerTileCoords.size() > 1) {
      needsProdSide = true;
    } else {
      auto [consCol, consRow] = info.consumerTileCoords[0];
      if (consRow >= 1) {
        bool rightAdj = state.targetModel->isLegalMemAffinity(prodCol, prodRow,
                                                              consCol, consRow);
        bool leftAdj = state.targetModel->isLegalMemAffinity(consCol, consRow,
                                                             prodCol, prodRow);
        if (!rightAdj && !leftAdj)
          needsProdSide = true;
      }
    }
    if (!needsProdSide && !info.viaDMA)
      continue;

    int64_t depth = info.depth > 0 ? info.depth : 1;
    int64_t prodDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
    if (prodDepth > 1 && state.conduitNamesWithConsumerAcquire.count(name))
      addConsumerSlot(prodTileVal);
    if (prodDepth > 1 && state.conduitNamesWithProducerAcquire.count(name))
      addProducerSlot(prodTileVal);
  }

  // Create one shared memref<N xi32> buffer per tile that needs N > 0 slots.
  if (state.insertAfterTile)
    builder.setInsertionPointAfter(state.insertAfterTile);
  else
    builder.setInsertionPointToStart(state.deviceBody);

  for (auto &[tileVal, count] : tileSlotCount) {
    if (count <= 0)
      continue;
    auto counterTy =
        mlir::MemRefType::get({count}, mlir::IntegerType::get(ctx, 32));
    // Assign a deterministic sym_name so the buffer is identifiable in IR
    // dumps and FileCheck patterns are stable across recompilations.
    std::string symName;
    if (auto tileOp = tileVal.getDefiningOp<AIE::TileOp>()) {
      int64_t col = tileOp.getCol();
      int64_t row = tileOp.getRow();
      symName = "_conduit_rot_ctr_tile_" + std::to_string(col) + "_" +
                std::to_string(row);
    }
    AIE::BufferOp sharedBuf = builder.create<AIE::BufferOp>(
        state.deviceOp.getLoc(), counterTy, tileVal,
        symName.empty() ? mlir::StringAttr{}
                        : mlir::StringAttr::get(ctx, symName),
        /*address=*/mlir::IntegerAttr{},
        /*initial_value=*/mlir::ElementsAttr{},
        /*mem_bank=*/mlir::IntegerAttr{});
    state.tileRotationBuf[tileVal] = sharedBuf;
    state.tileRotationBufNextSlot[tileVal] = 0;
  }
}

// ---------------------------------------------------------------------------
// Helper: assign a consumer rotation counter slot for a conduit on a tile.
// Must only be called for tiles that have a shared buffer (created by
// prescanAndCreateRotationBufs).
// ---------------------------------------------------------------------------
static void assignConsumerRotationSlot(ConduitToDMAState &state,
                                       ConduitInfo &info, mlir::Value tileVal,
                                       bool isPrimary) {
  auto bufIt = state.tileRotationBuf.find(tileVal);
  if (bufIt == state.tileRotationBuf.end())
    return; // no counter needed for this tile
  AIE::BufferOp sharedBuf = bufIt->second;
  int64_t slot = assignRotationSlot(state, tileVal);
  info.consumerTileRotationBufs[tileVal] = sharedBuf;
  info.consumerTileRotationBufSlots[tileVal] = slot;
  if (isPrimary) {
    info.rotationBuf = sharedBuf;
    info.rotationBufSlot = slot;
  }
}

static void assignProducerRotationSlot(ConduitToDMAState &state,
                                       ConduitInfo &info, mlir::Value tileVal) {
  auto bufIt = state.tileRotationBuf.find(tileVal);
  if (bufIt == state.tileRotationBuf.end())
    return;
  AIE::BufferOp sharedBuf = bufIt->second;
  int64_t slot = assignRotationSlot(state, tileVal);
  info.producerTileRotationBufs[tileVal] = sharedBuf;
  info.producerTileRotationBufSlots[tileVal] = slot;
  info.producerRotationBuf = sharedBuf;
  info.producerRotationBufSlot = slot;
}

void allocPhase(ConduitToDMAState &state) {
  if (!state.deviceOp)
    return; // no device — nothing to allocate

  mlir::OpBuilder &builder = *state.builder;
  mlir::MLIRContext *ctx = state.ctx;
  const bool isAIE2 = state.isAIE2Plus();
  const AIE::AIETargetModel &targetModel = *state.targetModel;

  // Pre-scan: count rotation counter slots per tile and create shared buffers.
  prescanAndCreateRotationBufs(state);

  for (auto &[name, info] : state.conduitMap) {
    // Cascade conduits use no buffers, locks, or DMA — skip entirely.
    if (info.routingMode == "cascade")
      continue;

    if (info.consumerTileCoords.empty() &&
        info.shimConsumerTileCoords.empty()) {
      // Producer-only conduit (shim DMA source) — handled in Phase 4.
      continue;
    }

    // -------------------------------------------------------------------
    // Phase 3b: Shim consumer(s) but no compute consumer.
    // Buffer and locks live on the PRODUCER tile (compute).
    //
    // EXCEPTION: join destination conduits (in linkDstNames) are skipped
    // here. Phase 5 (join) reuses the existing destination buffers for
    // the join intermediate buffer and allocates per-source lock pairs
    // directly. Over-allocating buffers and a lock pair here produces
    // duplicate resources (+2 buffers, +2 locks per join destination).
    // -------------------------------------------------------------------
    if (info.consumerTileCoords.empty() &&
        !info.shimConsumerTileCoords.empty()) {
      // Join/distribute destination conduits get their MemTile buffers from
      // linkPhase() intermediate buffer allocation — skip here to avoid
      // duplicate buffers and locks on the MemTile.
      if (state.linkDstNames.count(name))
        continue;
      auto [prodCol, prodRow] = info.producerTileCoord;
      if (prodCol < 0 || prodRow == 0) {
        state.deviceOp.emitWarning(
            llvm::Twine("conduit-to-dma: shim-to-shim conduit '") + name +
            "' dropped (shim producer + shim consumer not supported)");
        continue;
      }

      AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
      if (!prodTile)
        continue;

      int64_t depth = info.depth > 0 ? info.depth : 1;
      int64_t prodDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
      mlir::Type bufTy = info.elemType;
      if (!bufTy) {
        int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
        bufTy =
            mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
      }

      if (state.insertAfterTile)
        builder.setInsertionPointAfter(state.insertAfterTile);
      else
        builder.setInsertionPointToStart(state.deviceBody);

      mlir::Value prodTileVal = prodTile.getResult();

      info.buffers = state.allocateBuffers(prodTileVal, name, bufTy, prodDepth);
      if (!info.disableSynchronization) {
        // repeat_count > 1 scales prod lock init: each buffer is DMA'd N times.
        int64_t repeatN =
            info.bdChainRepeatCount > 1 ? info.bdChainRepeatCount : 1;
        int64_t prodInit = prodDepth * repeatN;
        auto locks =
            state.allocateLockPair(prodTileVal, name, prodDepth, prodInit);
        info.prodLock = locks.prodLock;
        info.consLock = locks.consLock;
        info.aie1Locks = std::move(locks.aie1Locks);
      }
      // Producer rotation counter slot (shared buffer created by pre-scan).
      if (prodDepth > 1 && state.conduitNamesWithProducerAcquire.count(name))
        assignProducerRotationSlot(state, info, prodTileVal);
      continue;
    }

    // -------------------------------------------------------------------
    // Phase 3c: Shared memory detection.
    //
    // If producer and single consumer are adjacent tiles, buffers and locks
    // go on the producer (or alloc_tile delegate) — no DMA needed.
    // Skip when via_DMA=true: force DMA path even for adjacent tiles.
    // -------------------------------------------------------------------
    if (!info.viaDMA && info.consumerTileCoords.size() == 1 &&
        info.shimConsumerTileCoords.empty() &&
        !state.linkSrcNamesEarly.count(name) &&
        !state.linkJoinSrcNames.count(name)) {
      auto [prodCol, prodRow] = info.producerTileCoord;
      auto [consCol, consRow] = info.consumerTileCoords[0];
      bool prodIsShim = (prodRow == 0);
      bool consIsShim = (consRow == 0);
      bool prodIsMemtile = targetModel.isMemTile(prodCol, prodRow);
      bool consIsMemtile = targetModel.isMemTile(consCol, consRow);
      if (!prodIsShim && !consIsShim && !prodIsMemtile && !consIsMemtile) {
        bool rightShared =
            targetModel.isLegalMemAffinity(prodCol, prodRow, consCol, consRow);
        bool leftShared =
            targetModel.isLegalMemAffinity(consCol, consRow, prodCol, prodRow);
        if (rightShared || leftShared) {
          // Defensive adjacency guard: assert the tiles that established
          // shared-memory eligibility are still adjacent.  This should
          // always hold here (the condition above guarantees it), but
          // if alloc_tile is provided we additionally verify that the
          // alloc tile is adjacent to both producer and consumer tiles
          // so that the physical buffer is reachable from both cores.
          if (info.hasAllocTile) {
            int64_t aCol = info.allocTileCoord.first;
            int64_t aRow = info.allocTileCoord.second;
            bool allocAdjToProd = targetModel.isLegalMemAffinity(
                aCol, aRow, prodCol, prodRow) ||
                                  targetModel.isLegalMemAffinity(
                                      prodCol, prodRow, aCol, aRow);
            bool allocAdjToCons = targetModel.isLegalMemAffinity(
                aCol, aRow, consCol, consRow) ||
                                  targetModel.isLegalMemAffinity(
                                      consCol, consRow, aCol, aRow);
            if (!allocAdjToProd || !allocAdjToCons) {
              // Find an existing conduit.create op to emit the error on.
              state.module.walk([&](Create createOp) {
                if (createOp.getName() == name) {
                  createOp.emitError(
                      "shared-memory conduit requires adjacent tiles; "
                      "alloc_tile (" + std::to_string(aCol) + "," +
                      std::to_string(aRow) + ") is not adjacent to both "
                      "producer (" + std::to_string(prodCol) + "," +
                      std::to_string(prodRow) + ") and consumer (" +
                      std::to_string(consCol) + "," +
                      std::to_string(consRow) + ")");
                  state.passFailed = true;
                }
              });
              if (state.passFailed)
                return;
            }
          }

          info.sharedMemory = true;

          int64_t allocCol = prodCol, allocRow = prodRow;
          if (info.hasAllocTile) {
            allocCol = info.allocTileCoord.first;
            allocRow = info.allocTileCoord.second;
          }

          AIE::TileOp allocTile = state.lookupTileByCoord(allocCol, allocRow);
          AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
          AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
          if (!allocTile || !consTile || !prodTile) {
            state.module.emitWarning(
                "conduit-to-dma: shared memory conduit '" + name +
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

            if (state.insertAfterTile)
              builder.setInsertionPointAfter(state.insertAfterTile);
            else
              builder.setInsertionPointToStart(state.deviceBody);

            mlir::Value allocTileVal = allocTile.getResult();
            mlir::Value prodTileVal = prodTile.getResult();
            mlir::Value consTileVal = consTile.getResult();

            // Allocate depth-many buffers on the allocation tile.
            llvm::SmallVector<AIE::BufferOp> sharedBuffers =
                state.allocateBuffers(allocTileVal, name, bufTy, depth);
            info.buffers = sharedBuffers;

            // Allocate lock(s) on the allocation tile (skip if
            // disable_synchronization).
            AIE::LockOp sharedProdLock, sharedConsLock;
            if (!info.disableSynchronization) {
              int64_t repeatN =
                  info.bdChainRepeatCount > 1 ? info.bdChainRepeatCount : 1;
              int64_t prodInit = depth * repeatN;
              auto locks =
                  state.allocateLockPair(allocTileVal, name, depth, prodInit);
              sharedProdLock = locks.prodLock;
              sharedConsLock = locks.consLock;
              info.prodLock = locks.prodLock;
              info.consLock = locks.consLock;
              info.aie1Locks = std::move(locks.aie1Locks);
            }

            // Register locks keyed on consumer and producer tiles for Phase 6.
            info.consumerTileLocks[consTileVal] = {sharedProdLock,
                                                   sharedConsLock};
            info.consumerTileBuffers[consTileVal] = sharedBuffers;
            info.consumerTileLocks[prodTileVal] = {sharedProdLock,
                                                   sharedConsLock};
            info.consumerTileBuffers[prodTileVal] = sharedBuffers;

            if (!isAIE2) {
              info.consumerTileAIE1Locks[consTileVal] = info.aie1Locks;
              info.consumerTileAIE1Locks[prodTileVal] = info.aie1Locks;
            }

            // Consumer rotation counter slot (shared buffer, pre-created).
            if (depth > 1 && state.conduitNamesWithConsumerAcquire.count(name))
              assignConsumerRotationSlot(state, info, consTileVal,
                                         /*isPrimary=*/true);

            // Producer rotation counter slot.
            if (depth > 1 && state.conduitNamesWithProducerAcquire.count(name))
              assignProducerRotationSlot(state, info, prodTileVal);

            continue; // skip normal DMA consumer loop
          }
        }
      }
    }

    // -------------------------------------------------------------------
    // Phase 3j: Join source conduits — allocate on PRODUCER tile.
    // -------------------------------------------------------------------
    if (state.linkJoinSrcNames.count(name)) {
      auto [prodCol, prodRow] = info.producerTileCoord;
      if (prodCol < 0 || prodRow < 2)
        continue;

      AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
      if (!prodTile)
        continue;

      int64_t depth = info.depth > 0 ? info.depth : 1;
      int64_t prodDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
      mlir::Type bufTy = info.elemType;
      if (!bufTy) {
        int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
        bufTy =
            mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
      }

      if (state.insertAfterTile)
        builder.setInsertionPointAfter(state.insertAfterTile);
      else
        builder.setInsertionPointToStart(state.deviceBody);

      mlir::Value prodTileVal = prodTile.getResult();

      info.buffers = state.allocateBuffers(prodTileVal, name, bufTy, prodDepth);
      if (!info.disableSynchronization) {
        int64_t repeatN =
            info.bdChainRepeatCount > 1 ? info.bdChainRepeatCount : 1;
        int64_t prodInit = prodDepth * repeatN;
        auto locks =
            state.allocateLockPair(prodTileVal, name, prodDepth, prodInit);
        info.prodLock = locks.prodLock;
        info.consLock = locks.consLock;
        info.aie1Locks = std::move(locks.aie1Locks);
      }
      // Producer rotation counter slot.
      if (prodDepth > 1 && state.conduitNamesWithProducerAcquire.count(name))
        assignProducerRotationSlot(state, info, prodTileVal);
      continue;
    }

    // -------------------------------------------------------------------
    // Normal consumer tile allocation (handles broadcast).
    // -------------------------------------------------------------------
    int64_t depth = info.depth > 0 ? info.depth : 1;
    mlir::Type bufTy = info.elemType;
    if (!bufTy) {
      int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
      bufTy = mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
    }

    for (unsigned consIdx = 0; consIdx < info.consumerTileCoords.size();
         ++consIdx) {
      auto [consCol, consRow] = info.consumerTileCoords[consIdx];
      AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
      if (!consTile) {
        state.module.emitWarning("conduit-to-dma: consumer tile (" +
                                 std::to_string(consCol) + "," +
                                 std::to_string(consRow) + ") for conduit '" +
                                 name + "' not found in device");
        continue;
      }

      if (state.insertAfterTile)
        builder.setInsertionPointAfter(state.insertAfterTile);
      else
        builder.setInsertionPointToStart(state.deviceBody);

      mlir::Value consTileVal = consTile.getResult();

      std::string bufSuffix = info.consumerTileCoords.size() > 1
                                  ? "_cons_" + std::to_string(consIdx)
                                  : "_cons";

      std::string consPrefix = name + bufSuffix;
      llvm::SmallVector<AIE::BufferOp> consBuffers =
          state.allocateBuffers(consTileVal, consPrefix, bufTy, depth);
      // Intentionally assigned before the linkSrcNamesEarly branch so the
      // branch's continue does not skip it.
      if (consIdx == 0)
        info.buffers = consBuffers;

      // Link source conduits: register MemTile-side buffers but skip
      // MemTile-side lock allocation (Phase 5 handles those for distribute).
      // Also allocate producer-side buffers+locks for compute producers.
      if (state.linkSrcNamesEarly.count(name)) {
        info.consumerTileBuffers[consTileVal] = consBuffers;

        // Distribute sources with a compute producer: allocate producer-side
        // buffers+locks on the compute tile for the aie.mem MM2S.
        // Lock allocation is skipped when disable_synchronization is set:
        // oracle emits no locks on the compute tile for these conduits.
        {
          auto [pCol, pRow] = info.producerTileCoord;
          if (pCol >= 0 && pRow >= 2) {
            AIE::TileOp pTile = state.lookupTileByCoord(pCol, pRow);
            if (pTile) {
              mlir::Value pTileVal = pTile.getResult();
              if (!info.consumerTileBuffers.count(pTileVal)) {
                int64_t prodDepth =
                    info.effectiveDepth > 0 ? info.effectiveDepth : depth;
                auto pBufs =
                    state.allocateBuffers(pTileVal, name, bufTy, prodDepth);

                AIE::LockOp pProdLock, pConsLock;
                if (!info.disableSynchronization) {
                  auto pLocks = state.allocateLockPair(pTileVal, name, prodDepth);
                  pProdLock = pLocks.prodLock;
                  pConsLock = pLocks.consLock;
                  if (!isAIE2)
                    info.consumerTileAIE1Locks[pTileVal] =
                        std::move(pLocks.aie1Locks);
                }

                info.consumerTileBuffers[pTileVal] = pBufs;
                info.consumerTileLocks[pTileVal] = {pProdLock, pConsLock};

                if (prodDepth > 1 &&
                    state.conduitNamesWithProducerAcquire.count(name))
                  assignProducerRotationSlot(state, info, pTileVal);
              }
            }
          }
        }

        continue; // advance to next consIdx
      }

      // Allocate lock(s) on the consumer tile (skip if
      // disable_synchronization).
      //
      // Consumer-tile lock init = depth (the number of buffer slots at the
      // receiving tile). repeat_count does NOT multiply here: the DMA BD
      // chain fires repeat_count times per buffer slot, but the consumer FIFO
      // always has exactly depth slots. The repeat_count scaling belongs only
      // on the producer-side lock (allocated in Phase 3d below).
      AIE::LockOp thisProdLock, thisConsLock;
      if (!info.disableSynchronization) {
        int64_t prodInit = depth;
        auto consLocks =
            state.allocateLockPair(consTileVal, consPrefix, depth, prodInit);
        thisProdLock = consLocks.prodLock;
        thisConsLock = consLocks.consLock;
        if (consIdx == 0) {
          info.prodLock = consLocks.prodLock;
          info.consLock = consLocks.consLock;
          if (!isAIE2)
            info.aie1Locks = consLocks.aie1Locks;
        }
        if (!isAIE2)
          info.consumerTileAIE1Locks[consTileVal] = consLocks.aie1Locks;
      }

      info.consumerTileLocks[consTileVal] = {thisProdLock, thisConsLock};
      info.consumerTileBuffers[consTileVal] = consBuffers;

      // Consumer rotation counter slot (shared buffer, pre-created).
      if (depth > 1 && state.conduitNamesWithConsumerAcquire.count(name))
        assignConsumerRotationSlot(state, info, consTileVal,
                                   /*isPrimary=*/(consIdx == 0));
    }
  }

  // -------------------------------------------------------------------
  // Phase 3d: Allocate producer-side buffers and locks for non-adjacent
  //           compute→compute conduits.
  // -------------------------------------------------------------------
  for (auto &[name, info] : state.conduitMap) {
    if (info.routingMode == "cascade")
      continue;
    if (info.sharedMemory)
      continue;
    if (state.linkSrcNamesEarly.count(name) ||
        state.linkJoinSrcNames.count(name))
      continue;
    // Link destination conduits (distribute dsts or join dst) share the
    // MemTile buffer set owned by the source conduit — skip producer-side
    // allocation here to avoid over-allocating duplicate buffers and locks.
    if (state.linkDstNames.count(name))
      continue;
    auto [prodCol, prodRow] = info.producerTileCoord;
    if (prodCol < 0 || prodRow == 0)
      continue;
    if (info.consumerTileCoords.empty())
      continue;

    AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
    if (!prodTile)
      continue;
    mlir::Value prodTileVal = prodTile.getResult();

    if (info.consumerTileBuffers.count(prodTileVal))
      continue;

    bool needsProdSide = false;
    if (info.consumerTileCoords.size() > 1) {
      needsProdSide = true;
    } else {
      auto [consCol, consRow] = info.consumerTileCoords[0];
      if (consRow >= 1) {
        bool rightAdj = state.targetModel->isLegalMemAffinity(prodCol, prodRow,
                                                              consCol, consRow);
        bool leftAdj = state.targetModel->isLegalMemAffinity(consCol, consRow,
                                                             prodCol, prodRow);
        if (!rightAdj && !leftAdj)
          needsProdSide = true;
      }
    }
    // via_DMA forces DMA even for adjacent tiles.
    if (!needsProdSide && !info.viaDMA)
      continue;

    int64_t depth = info.depth > 0 ? info.depth : 1;
    int64_t prodDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
    mlir::Type bufTy = info.elemType;
    if (!bufTy) {
      int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
      bufTy = mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
    }

    if (state.insertAfterTile)
      builder.setInsertionPointAfter(state.insertAfterTile);
    else
      builder.setInsertionPointToStart(state.deviceBody);

    auto prodBuffers =
        state.allocateBuffers(prodTileVal, name, bufTy, prodDepth);

    AIE::LockOp prodLockProd, prodLockCons;
    if (!info.disableSynchronization) {
      int64_t repeatN =
          info.bdChainRepeatCount > 1 ? info.bdChainRepeatCount : 1;
      int64_t prodInit = prodDepth * repeatN;
      auto prodLocks =
          state.allocateLockPair(prodTileVal, name, prodDepth, prodInit);
      prodLockProd = prodLocks.prodLock;
      prodLockCons = prodLocks.consLock;
      if (!isAIE2)
        info.consumerTileAIE1Locks[prodTileVal] =
            std::move(prodLocks.aie1Locks);
    }

    info.consumerTileLocks[prodTileVal] = {prodLockProd, prodLockCons};
    info.consumerTileBuffers[prodTileVal] = prodBuffers;

    // Consumer rotation counter: the producer core acquires this conduit in
    // Consume mode before forwarding data via DMA (Phase 3d non-adjacent path).
    if (prodDepth > 1 && state.conduitNamesWithConsumerAcquire.count(name))
      assignConsumerRotationSlot(state, info, prodTileVal,
                                 /*isPrimary=*/true);
    // Producer rotation counter slot.
    if (prodDepth > 1 && state.conduitNamesWithProducerAcquire.count(name))
      assignProducerRotationSlot(state, info, prodTileVal);
  }

  // Debug assertion: every tile's next-slot cursor must not exceed the size
  // of the shared buffer that was pre-allocated by
  // prescanAndCreateRotationBufs. A violation means assignRotationSlot was
  // called more times than prescan counted, which would produce out-of-bounds
  // slot indices.
  for (auto &[tileVal, sharedBuf] : state.tileRotationBuf) {
    auto bufType = mlir::cast<mlir::MemRefType>(sharedBuf.getType());
    int64_t bufSize = bufType.getShape()[0];
    int64_t nextSlot = state.tileRotationBufNextSlot[tileVal];
    assert(nextSlot <= bufSize &&
           "allocPhase: rotation slot overrun — prescan count was too low");
    (void)bufSize;
    (void)nextSlot;
  }
}

} // namespace xilinx::conduit
