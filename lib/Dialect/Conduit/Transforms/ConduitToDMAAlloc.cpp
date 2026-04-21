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
//   allocated via aie.buffer at device level (matching stateful transform's
//   _anonymous buffer pattern).  A pre-scan pass counts how many counter slots
//   each tile needs; one shared buffer is created per tile and each conduit is
//   assigned a slot index.
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
    // Multi-device: ensure tile lookups target the correct device.
    if (state.isMultiDevice())
      state.switchToDeviceIndex(info.deviceIndex);

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
      int64_t effDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
      // Partial-release Produce-port: need max(effDepth, maxProduceAcquire+1).
      int64_t prodDepth = (info.maxProduceAcquire > 0)
                              ? std::max(effDepth, info.maxProduceAcquire + 1)
                              : effDepth;
      if (prodDepth > 1 && state.conduitNamesWithProducerAcquire.count(name))
        addProducerSlot(prodTile.getResult());
      continue;
    }

    // Phase 3c path (shared memory, adjacent tiles).
    // Link dst conduits skip shared-memory detection: the link relay changes
    // data routing, so the relay's adjacent producer tile is not a direct
    // shared-memory provider. These conduits take the normal DMA path.
    if (!info.forceDMA && info.consumerTileCoords.size() == 1 &&
        info.shimConsumerTileCoords.empty() &&
        !state.linkSrcNamesEarly.count(name) &&
        !state.linkJoinSrcNames.count(name) &&
        !state.linkDstNames.count(name)) {
      auto [prodCol, prodRow] = info.producerTileCoord;
      auto [consCol, consRow] = info.consumerTileCoords[0];
      bool prodIsShim = (prodRow == 0);
      bool consIsShim = (consRow == 0);
      bool prodIsMemtile = targetModel.isMemTile(prodCol, prodRow);
      bool consIsMemtile = targetModel.isMemTile(consCol, consRow);
      // Same-tile self-loop: not shared memory — core accesses its own
      // local memory directly.  Skip to normal consumer allocation.
      bool sameTile = (prodCol == consCol && prodRow == consRow);
      if (!prodIsShim && !consIsShim && !prodIsMemtile && !consIsMemtile &&
          !sameTile) {
        bool rightShared =
            targetModel.isLegalMemAffinity(prodCol, prodRow, consCol, consRow);
        bool leftShared =
            targetModel.isLegalMemAffinity(consCol, consRow, prodCol, prodRow);
        bool explicitSharedMem =
            (info.routingMode == RoutingMode::SharedMemory);
        if (explicitSharedMem || rightShared || leftShared) {
          AIE::TileOp allocTile = state.lookupTileByCoord(prodCol, prodRow);
          AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
          AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
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
      int64_t effDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
      // Partial-release Produce-port: need max(effDepth, maxProduceAcquire+1).
      int64_t prodDepth = (info.maxProduceAcquire > 0)
                              ? std::max(effDepth, info.maxProduceAcquire + 1)
                              : effDepth;
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
        // Stream conduits: no producer-side allocation, skip counter.
        if (info.routingMode != RoutingMode::Stream) {
          auto [pCol, pRow] = info.producerTileCoord;
          if (pCol >= 0 && pRow >= 2) {
            AIE::TileOp pTile = state.lookupTileByCoord(pCol, pRow);
            if (pTile && !info.consumerTileBuffers.count(pTile.getResult())) {
              int64_t effDepth =
                  info.effectiveDepth > 0 ? info.effectiveDepth : depth;
              int64_t prodDepth =
                  (info.maxProduceAcquire > 0)
                      ? std::max(effDepth, info.maxProduceAcquire + 1)
                      : effDepth;
              if (prodDepth > 1 &&
                  state.conduitNamesWithProducerAcquire.count(name))
                addProducerSlot(pTile.getResult());
            }
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
    // Multi-device: ensure tile lookups target the correct device.
    if (state.isMultiDevice())
      state.switchToDeviceIndex(info.deviceIndex);

    if (info.sharedMemory)
      continue;
    // Stream conduits: no producer-side DMA — skip producer-side counter.
    if (info.routingMode == RoutingMode::Stream)
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
    if (!info.shimConsumerTileCoords.empty())
      needsProdSide = true;
    if (!needsProdSide && !info.forceDMA)
      continue;

    int64_t depth = info.depth > 0 ? info.depth : 1;
    int64_t effDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
    int64_t prodDepth = (info.maxProduceAcquire > 0)
                            ? std::max(effDepth, info.maxProduceAcquire + 1)
                            : effDepth;
    if (prodDepth > 1 && state.conduitNamesWithConsumerAcquire.count(name))
      addConsumerSlot(prodTileVal);
    if (prodDepth > 1 && state.conduitNamesWithProducerAcquire.count(name))
      addProducerSlot(prodTileVal);
  }

  // Create one shared rotation counter per tile that needs N > 0 slots.
  // Allocated as aie.buffer at device level, matching the stateful
  // transform's _anonymous buffer pattern.  The buffer is placed on the
  // same tile as the core; AIEAssignBuffers will assign it an address and
  // an _anonymous name.  Init stores (zeroing) are emitted inside the core
  // body by the lowering phase (ConduitToDMALower.cpp).
  for (auto &[tileVal, count] : tileSlotCount) {
    if (count <= 0)
      continue;
    auto counterTy =
        mlir::MemRefType::get({count}, mlir::IntegerType::get(ctx, 32));
    // Find the core op that owns this tile.
    AIE::CoreOp coreOp = nullptr;
    state.deviceBody->walk([&](AIE::CoreOp core) {
      if (core.getTile() == tileVal)
        coreOp = core;
    });
    if (!coreOp)
      continue; // shim or memory tile without core — no buffer needed
    // Insert aie.buffer at device level, just before the core op.
    mlir::OpBuilder bufBuilder(coreOp);
    auto bufOp = AIE::BufferOp::create(
        bufBuilder, state.deviceOp.getLoc(), counterTy, tileVal,
        /*sym_name=*/mlir::StringAttr{}, /*address=*/mlir::IntegerAttr{},
        /*initial_value=*/mlir::ElementsAttr{},
        /*mem_bank=*/mlir::IntegerAttr{});
    state.tileRotationBuf[tileVal] = bufOp.getResult();
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
  mlir::Value sharedBuf = bufIt->second;
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
  mlir::Value sharedBuf = bufIt->second;
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
    // Multi-device: ensure tile lookups target the correct device.
    if (state.isMultiDevice())
      state.switchToDeviceIndex(info.deviceIndex);

    // Cascade conduits use no buffers, locks, or DMA — skip entirely.
    if (info.routingMode == RoutingMode::Cascade)
      continue;

    // Phase 5b: track whether buffers were pre-materialized.
    // When true, buffer allocation calls are skipped but lock allocation
    // and all other per-channel work continues normally.
    const bool preMaterialized = !info.buffers.empty();

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
      // Producer buffer count: effectiveDepth, inflated by partial-release when
      // needed. effectiveDepth = min(depth, maxProdAcquire+1) reduces
      // allocation for producers that never hold more than maxProdAcquire+1
      // slots simultaneously. When maxProduceAcquire > 0 (partial-release
      // pattern), inflate to at least maxProduceAcquire+1 so the DMA can drain
      // while the core holds extra slots.
      int64_t effDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
      // Partial-release Produce-port: need max(effDepth, maxProduceAcquire+1).
      int64_t prodDepth = (info.maxProduceAcquire > 0)
                              ? std::max(effDepth, info.maxProduceAcquire + 1)
                              : effDepth;
      mlir::Type bufTy = info.elemType;
      if (!bufTy) {
        int64_t bufSize = 1;
        bufTy =
            mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
      }

      // Multi-device: select the DeviceOp body that owns this producer tile.
      state.switchDeviceForTile(prodCol, prodRow);

      if (state.insertAfterTile)
        builder.setInsertionPointAfter(state.insertAfterTile);
      else
        builder.setInsertionPointToStart(state.deviceBody);

      mlir::Value prodTileVal = prodTile.getResult();

      info.buffers = state.allocateBuffers(prodTileVal, name, bufTy, prodDepth);
      if (!info.noLocks) {
        // bd_repeat > 1 scales prod lock init: each buffer is DMA'd N times.
        int64_t repeatN = info.bdRepeat > 1 ? info.bdRepeat : 1;
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
    // Skip when forceDMA: force DMA path even for adjacent tiles.
    // Skip link dst conduits: the link relay changes data routing, so the
    // relay's adjacent producer is not a direct shared-memory provider.
    //
    // Shared memory is triggered by either:
    //   (a) routing_mode == "shared_memory" (set explicitly by
    //       --conduit-infer-modes R3a), or
    //   (b) routing_mode absent/other AND tiles are adjacent.
    // -------------------------------------------------------------------
    if (!info.forceDMA && info.consumerTileCoords.size() == 1 &&
        info.shimConsumerTileCoords.empty() &&
        !state.linkSrcNamesEarly.count(name) &&
        !state.linkJoinSrcNames.count(name) &&
        !state.linkDstNames.count(name)) {
      auto [prodCol, prodRow] = info.producerTileCoord;
      auto [consCol, consRow] = info.consumerTileCoords[0];
      bool prodIsShim = (prodRow == 0);
      bool consIsShim = (consRow == 0);
      bool prodIsMemtile = targetModel.isMemTile(prodCol, prodRow);
      bool consIsMemtile = targetModel.isMemTile(consCol, consRow);
      // Same-tile self-loop: not shared memory — core accesses its own
      // local memory directly.  Skip to normal consumer allocation.
      bool sameTile = (prodCol == consCol && prodRow == consRow);
      if (!prodIsShim && !consIsShim && !prodIsMemtile && !consIsMemtile &&
          !sameTile) {
        bool rightShared =
            targetModel.isLegalMemAffinity(prodCol, prodRow, consCol, consRow);
        bool leftShared =
            targetModel.isLegalMemAffinity(consCol, consRow, prodCol, prodRow);
        bool explicitSharedMem =
            (info.routingMode == RoutingMode::SharedMemory);
        if (explicitSharedMem || rightShared || leftShared) {
          info.sharedMemory = true;

          AIE::TileOp allocTile = state.lookupTileByCoord(prodCol, prodRow);
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
            int64_t nBufs = info.nConsumerBuffers();
            mlir::Type bufTy = info.elemType;
            if (!bufTy) {
              int64_t bufSize = 1;
              bufTy = mlir::MemRefType::get({bufSize},
                                            mlir::IntegerType::get(ctx, 32));
            }

            // Multi-device: allocate into the device that owns the producer
            // tile.
            state.switchDeviceForTile(prodCol, prodRow);

            if (state.insertAfterTile)
              builder.setInsertionPointAfter(state.insertAfterTile);
            else
              builder.setInsertionPointToStart(state.deviceBody);

            mlir::Value allocTileVal = allocTile.getResult();
            mlir::Value prodTileVal = prodTile.getResult();
            mlir::Value consTileVal = consTile.getResult();

            // Allocate nBufs-many buffers on the allocation tile.
            // nBufs >= depth; extra slots support sliding-window
            // acquire>release.
            llvm::SmallVector<AIE::BufferOp> sharedBuffers =
                state.allocateBuffers(allocTileVal, name, bufTy, nBufs);
            info.buffers = sharedBuffers;

            // Allocate lock(s) on the allocation tile (skip if
            // disable_synchronization).
            AIE::LockOp sharedProdLock, sharedConsLock;
            if (!info.noLocks) {
              int64_t repeatN = info.bdRepeat > 1 ? info.bdRepeat : 1;
              int64_t prodInit = nBufs * repeatN;
              auto locks =
                  state.allocateLockPair(allocTileVal, name, nBufs, prodInit);
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
      int64_t effDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
      // Partial-release Produce-port: need max(effDepth, maxProduceAcquire+1).
      int64_t prodDepth = (info.maxProduceAcquire > 0)
                              ? std::max(effDepth, info.maxProduceAcquire + 1)
                              : effDepth;
      mlir::Type bufTy = info.elemType;
      if (!bufTy) {
        int64_t bufSize = 1;
        bufTy =
            mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
      }

      // Multi-device: select device owning this producer tile.
      state.switchDeviceForTile(prodCol, prodRow);

      if (state.insertAfterTile)
        builder.setInsertionPointAfter(state.insertAfterTile);
      else
        builder.setInsertionPointToStart(state.deviceBody);

      mlir::Value prodTileVal = prodTile.getResult();

      info.buffers = state.allocateBuffers(prodTileVal, name, bufTy, prodDepth);
      if (!info.noLocks) {
        int64_t repeatN = info.bdRepeat > 1 ? info.bdRepeat : 1;
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
    // nBufs >= depth: extra buffer slots support sliding-window patterns
    // where acquire_count > release_count (partial release).
    int64_t nBufs = info.nConsumerBuffers();
    mlir::Type bufTy = info.elemType;
    if (!bufTy) {
      int64_t bufSize = 1;
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

      // Multi-device: select device owning this consumer tile.
      state.switchDeviceForTile(consCol, consRow);

      if (state.insertAfterTile)
        builder.setInsertionPointAfter(state.insertAfterTile);
      else
        builder.setInsertionPointToStart(state.deviceBody);

      mlir::Value consTileVal = consTile.getResult();

      // MemTile relay buffer cap: when the consumer is a MemTile acting as a
      // relay, the putCount-inflated nConsumerBuffers() is wrong — the
      // ShimTile's runtime sequence put count drives the shim BD chain length,
      // not the MemTile relay's buffering.  The relay uses a repeating BD
      // chain whose buffer count is the conduit depth, not putCount.
      int64_t consNBufs = nBufs;
      if (targetModel.isMemTile(consCol, consRow) && info.putCount > 1 &&
          info.dmaRepeat == 0) {
        consNBufs = depth;
      }

      // Use indexed naming when total consumers (compute + shim) > 1 to avoid
      // symbol collisions between Phase 3 (compute consumer) and Phase 4b
      // (shim consumer) lock names.
      bool multiConsumer = (info.consumerTileCoords.size() +
                            info.shimConsumerTileCoords.size()) > 1;
      std::string bufSuffix =
          multiConsumer ? "_cons_" + std::to_string(consIdx) : "_cons";

      std::string consPrefix = name + bufSuffix;
      // Allocate consumer buffers (info.buffers populated in Phase 1.5 for
      // pre-materialized case).
      llvm::SmallVector<AIE::BufferOp> consBuffers;
      if (!preMaterialized) {
        consBuffers =
            state.allocateBuffers(consTileVal, consPrefix, bufTy, consNBufs);
        // Intentionally assigned before the linkSrcNamesEarly branch so the
        // branch's continue does not skip it.
        if (consIdx == 0)
          info.buffers = consBuffers;
      } else {
        // Re-use pre-materialized buffers for the per-consumer buffer vector.
        for (auto bufOp : info.buffers)
          consBuffers.push_back(bufOp);
      }

      // Link source conduits: register MemTile-side buffers but skip
      // MemTile-side lock allocation (Phase 5 handles those for distribute).
      // Also allocate producer-side buffers+locks for compute producers.
      if (state.linkSrcNamesEarly.count(name)) {
        info.consumerTileBuffers[consTileVal] = consBuffers;

        // Distribute sources with a compute producer: allocate producer-side
        // buffers+locks on the compute tile for the aie.mem MM2S.
        // Lock allocation is skipped when disable_synchronization is set:
        // oracle emits no locks on the compute tile for these conduits.
        //
        // Stream conduits (routing_mode="stream"): skip producer-side
        // allocation entirely. The producer core outputs data directly
        // through the Core AXI stream port — no DMA, buffers, or locks
        // on the producer tile.
        if (info.routingMode != RoutingMode::Stream) {
          auto [pCol, pRow] = info.producerTileCoord;
          if (pCol >= 0 && pRow >= 2) {
            AIE::TileOp pTile = state.lookupTileByCoord(pCol, pRow);
            if (pTile) {
              mlir::Value pTileVal = pTile.getResult();
              if (!info.consumerTileBuffers.count(pTileVal)) {
                int64_t effDepth =
                    info.effectiveDepth > 0 ? info.effectiveDepth : depth;
                int64_t prodDepth =
                    (info.maxProduceAcquire > 0)
                        ? std::max(effDepth, info.maxProduceAcquire + 1)
                        : effDepth;
                auto pBufs =
                    state.allocateBuffers(pTileVal, name, bufTy, prodDepth);

                AIE::LockOp pProdLock, pConsLock;
                if (!info.noLocks) {
                  auto pLocks =
                      state.allocateLockPair(pTileVal, name, prodDepth);
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
      // Consumer-tile prod_lock init = consNBufs (number of buffer slots,
      // including any extra for sliding-window partial release). bd_repeat
      // does NOT multiply here: the DMA BD chain fires bd_repeat times per
      // buffer slot, but the bd_repeat scaling belongs only on the
      // producer-side lock (allocated in Phase 3d below).
      AIE::LockOp thisProdLock, thisConsLock;
      if (!info.noLocks) {
        int64_t prodInit = consNBufs;
        auto consLocks = state.allocateLockPair(consTileVal, consPrefix,
                                                consNBufs, prodInit);
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
    // Multi-device: ensure tile lookups target the correct device.
    if (state.isMultiDevice())
      state.switchToDeviceIndex(info.deviceIndex);

    if (info.routingMode == RoutingMode::Cascade)
      continue;
    // Stream conduits: no producer-side DMA — skip producer-side allocation.
    if (info.routingMode == RoutingMode::Stream)
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
    // Shim consumers always need producer-side DMA allocation: data must
    // reach the shim tile via the switchbox network, not shared memory.
    if (!info.shimConsumerTileCoords.empty())
      needsProdSide = true;
    // via_DMA forces DMA even for adjacent tiles.
    if (!needsProdSide && !info.forceDMA)
      continue;

    int64_t depth = info.depth > 0 ? info.depth : 1;
    int64_t effDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
    int64_t prodDepth = (info.maxProduceAcquire > 0)
                            ? std::max(effDepth, info.maxProduceAcquire + 1)
                            : effDepth;
    mlir::Type bufTy = info.elemType;
    if (!bufTy) {
      int64_t bufSize = 1;
      bufTy = mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
    }

    // Multi-device: select device owning this producer tile.
    state.switchDeviceForTile(prodCol, prodRow);

    if (state.insertAfterTile)
      builder.setInsertionPointAfter(state.insertAfterTile);
    else
      builder.setInsertionPointToStart(state.deviceBody);

    auto prodBuffers =
        state.allocateBuffers(prodTileVal, name, bufTy, prodDepth);

    AIE::LockOp prodLockProd, prodLockCons;
    if (!info.noLocks) {
      int64_t repeatN = info.bdRepeat > 1 ? info.bdRepeat : 1;
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
  for (auto &[tileVal, sharedBufVal] : state.tileRotationBuf) {
    auto bufType = mlir::cast<mlir::MemRefType>(sharedBufVal.getType());
    int64_t bufSize = bufType.getShape()[0];
    int64_t nextSlot = state.tileRotationBufNextSlot[tileVal];
    assert(nextSlot <= bufSize &&
           "allocPhase: rotation slot overrun — prescan count was too low");
    (void)bufSize;
    (void)nextSlot;
  }
}

} // namespace xilinx::conduit
