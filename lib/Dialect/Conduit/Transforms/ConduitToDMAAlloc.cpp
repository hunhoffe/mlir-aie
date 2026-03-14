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
//===----------------------------------------------------------------------===//

#include "ConduitToDMACommon.h"

namespace xilinx::conduit {

void allocPhase(ConduitToDMAState &state) {
  if (!state.deviceOp)
    return; // no device — nothing to allocate

  mlir::OpBuilder &builder = *state.builder;
  mlir::MLIRContext *ctx = state.ctx;
  const bool isAIE2 = state.isAIE2Plus();
  const AIE::AIETargetModel &targetModel = *state.targetModel;

  for (auto &[name, info] : state.conduitMap) {
    if (info.consumerTileCoords.empty() &&
        info.shimConsumerTileCoords.empty()) {
      // Producer-only conduit (shim DMA source) — handled in Phase 4.
      continue;
    }

    // -------------------------------------------------------------------
    // Phase 3b: Shim consumer(s) but no compute consumer.
    // Buffer and locks live on the PRODUCER tile (compute).
    // -------------------------------------------------------------------
    if (info.consumerTileCoords.empty() &&
        !info.shimConsumerTileCoords.empty()) {
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
        bufTy = mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
      }

      if (state.insertAfterTile)
        builder.setInsertionPointAfter(state.insertAfterTile);
      else
        builder.setInsertionPointToStart(state.deviceBody);

      mlir::Value prodTileVal = prodTile.getResult();

      info.buffers = state.allocateBuffers(prodTileVal, name, bufTy, prodDepth);
      auto locks = state.allocateLockPair(prodTileVal, name, prodDepth);
      info.prodLock = locks.prodLock;
      info.consLock = locks.consLock;
      info.aie1Locks = std::move(locks.aie1Locks);
      continue;
    }

    // -------------------------------------------------------------------
    // Phase 3c: Shared memory detection.
    //
    // If producer and single consumer are adjacent tiles, buffers and locks
    // go on the producer (or alloc_tile delegate) — no DMA needed.
    // -------------------------------------------------------------------
    if (info.consumerTileCoords.size() == 1 &&
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
        bool rightShared = targetModel.isLegalMemAffinity(
            prodCol, prodRow, consCol, consRow);
        bool leftShared = targetModel.isLegalMemAffinity(
            consCol, consRow, prodCol, prodRow);
        if (rightShared || leftShared) {
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

            // Allocate lock(s) on the allocation tile.
            auto locks = state.allocateLockPair(allocTileVal, name, depth);
            AIE::LockOp sharedProdLock = locks.prodLock;
            AIE::LockOp sharedConsLock = locks.consLock;
            info.prodLock = locks.prodLock;
            info.consLock = locks.consLock;
            info.aie1Locks = std::move(locks.aie1Locks);

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

            // Rotation counter for depth>1 on consumer tile.
            if (depth > 1 && state.conduitNamesWithConsumerAcquire.count(name)) {
              auto counterTy = mlir::MemRefType::get(
                  {1}, mlir::IntegerType::get(ctx, 32));
              info.rotationBuf = builder.create<AIE::BufferOp>(
                  state.deviceOp.getLoc(), counterTy, consTileVal,
                  /*sym_name=*/mlir::StringAttr{},
                  /*address=*/mlir::IntegerAttr{},
                  /*initial_value=*/mlir::ElementsAttr{},
                  /*mem_bank=*/mlir::IntegerAttr{});
              info.consumerTileRotationBufs[consTileVal] = info.rotationBuf;
            }

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
        bufTy = mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
      }

      if (state.insertAfterTile)
        builder.setInsertionPointAfter(state.insertAfterTile);
      else
        builder.setInsertionPointToStart(state.deviceBody);

      mlir::Value prodTileVal = prodTile.getResult();

      info.buffers = state.allocateBuffers(prodTileVal, name, bufTy, prodDepth);
      auto locks = state.allocateLockPair(prodTileVal, name, prodDepth);
      info.prodLock = locks.prodLock;
      info.consLock = locks.consLock;
      info.aie1Locks = std::move(locks.aie1Locks);
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
        state.module.emitWarning(
            "conduit-to-dma: consumer tile (" + std::to_string(consCol) +
            "," + std::to_string(consRow) +
            ") for conduit '" + name + "' not found in device");
        continue;
      }

      if (state.insertAfterTile)
        builder.setInsertionPointAfter(state.insertAfterTile);
      else
        builder.setInsertionPointToStart(state.deviceBody);

      mlir::Value consTileVal = consTile.getResult();

      std::string bufSuffix =
          info.consumerTileCoords.size() > 1
              ? "_cons_" + std::to_string(consIdx)
              : "_cons";

      std::string consPrefix = name + bufSuffix;
      llvm::SmallVector<AIE::BufferOp> consBuffers =
          state.allocateBuffers(consTileVal, consPrefix, bufTy, depth);
      if (consIdx == 0)
        info.buffers = consBuffers;

      // Link source conduits: register MemTile-side buffers but skip
      // MemTile-side lock allocation (Phase 5 handles those for distribute).
      // Also allocate producer-side buffers+locks for compute producers.
      if (state.linkSrcNamesEarly.count(name)) {
        info.consumerTileBuffers[consTileVal] = consBuffers;

        // Distribute sources with a compute producer: allocate producer-side
        // buffers+locks on the compute tile for the aie.mem MM2S.
        {
          auto [pCol, pRow] = info.producerTileCoord;
          if (pCol >= 0 && pRow >= 2) {
            AIE::TileOp pTile = state.lookupTileByCoord(pCol, pRow);
            if (pTile) {
              mlir::Value pTileVal = pTile.getResult();
              if (!info.consumerTileBuffers.count(pTileVal)) {
                int64_t prodDepth = info.effectiveDepth > 0
                                        ? info.effectiveDepth
                                        : depth;
                auto pBufs = state.allocateBuffers(
                    pTileVal, name, bufTy, prodDepth);
                auto pLocks = state.allocateLockPair(
                    pTileVal, name, prodDepth);

                if (!isAIE2)
                  info.consumerTileAIE1Locks[pTileVal] =
                      std::move(pLocks.aie1Locks);

                info.consumerTileBuffers[pTileVal] = pBufs;
                info.consumerTileLocks[pTileVal] = {pLocks.prodLock,
                                                    pLocks.consLock};

                if (prodDepth > 1) {
                  auto counterTy = mlir::MemRefType::get(
                      {1}, mlir::IntegerType::get(ctx, 32));
                  AIE::BufferOp rotBuf = builder.create<AIE::BufferOp>(
                      state.deviceOp.getLoc(), counterTy, pTileVal,
                      /*sym_name=*/mlir::StringAttr{},
                      /*address=*/mlir::IntegerAttr{},
                      /*initial_value=*/mlir::ElementsAttr{},
                      /*mem_bank=*/mlir::IntegerAttr{});
                  info.consumerTileRotationBufs[pTileVal] = rotBuf;
                }
              }
            }
          }
        }

        continue; // advance to next consIdx
      }

      // Allocate lock(s) on the consumer tile.
      auto consLocks = state.allocateLockPair(consTileVal, consPrefix, depth);
      AIE::LockOp thisProdLock = consLocks.prodLock;
      AIE::LockOp thisConsLock = consLocks.consLock;
      if (consIdx == 0) {
        info.prodLock = consLocks.prodLock;
        info.consLock = consLocks.consLock;
        if (!isAIE2)
          info.aie1Locks = consLocks.aie1Locks;
      }
      if (!isAIE2)
        info.consumerTileAIE1Locks[consTileVal] = consLocks.aie1Locks;

      info.consumerTileLocks[consTileVal] = {thisProdLock, thisConsLock};
      info.consumerTileBuffers[consTileVal] = consBuffers;

      // Rotation counter for depth>1 on each consumer tile.
      if (depth > 1 && state.conduitNamesWithConsumerAcquire.count(name)) {
        auto counterTy =
            mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));
        AIE::BufferOp rotBuf = builder.create<AIE::BufferOp>(
            state.deviceOp.getLoc(), counterTy, consTileVal,
            /*sym_name=*/mlir::StringAttr{},
            /*address=*/mlir::IntegerAttr{},
            /*initial_value=*/mlir::ElementsAttr{},
            /*mem_bank=*/mlir::IntegerAttr{});
        info.consumerTileRotationBufs[consTileVal] = rotBuf;
        if (consIdx == 0)
          info.rotationBuf = rotBuf;
      }
    }
  }

  // -------------------------------------------------------------------
  // Phase 3d: Allocate producer-side buffers and locks for non-adjacent
  //           compute→compute conduits.
  // -------------------------------------------------------------------
  for (auto &[name, info] : state.conduitMap) {
    if (info.sharedMemory)
      continue;
    if (state.linkSrcNamesEarly.count(name) || state.linkJoinSrcNames.count(name))
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
        bool rightAdj = state.targetModel->isLegalMemAffinity(
            prodCol, prodRow, consCol, consRow);
        bool leftAdj = state.targetModel->isLegalMemAffinity(
            consCol, consRow, prodCol, prodRow);
        if (!rightAdj && !leftAdj)
          needsProdSide = true;
      }
    }
    if (!needsProdSide)
      continue;

    int64_t depth = info.depth > 0 ? info.depth : 1;
    int64_t prodDepth = info.effectiveDepth > 0 ? info.effectiveDepth : depth;
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

    auto prodBuffers = state.allocateBuffers(prodTileVal, name, bufTy, prodDepth);
    auto prodLocks = state.allocateLockPair(prodTileVal, name, prodDepth);

    info.consumerTileLocks[prodTileVal] = {prodLocks.prodLock,
                                           prodLocks.consLock};
    info.consumerTileBuffers[prodTileVal] = prodBuffers;
    if (!isAIE2)
      info.consumerTileAIE1Locks[prodTileVal] =
          std::move(prodLocks.aie1Locks);

    if (prodDepth > 1 && state.conduitNamesWithConsumerAcquire.count(name)) {
      auto counterTy =
          mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));
      AIE::BufferOp rotBuf = builder.create<AIE::BufferOp>(
          state.deviceOp.getLoc(), counterTy, prodTileVal,
          /*sym_name=*/mlir::StringAttr{},
          /*address=*/mlir::IntegerAttr{},
          /*initial_value=*/mlir::ElementsAttr{},
          /*mem_bank=*/mlir::IntegerAttr{});
      info.consumerTileRotationBufs[prodTileVal] = rotBuf;
    }
  }
}

} // namespace xilinx::conduit
