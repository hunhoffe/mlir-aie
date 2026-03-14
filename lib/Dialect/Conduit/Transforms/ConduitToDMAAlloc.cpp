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
  const bool isAIE2 = state.isAIE2;
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

      for (int64_t i = 0; i < prodDepth; ++i) {
        std::string symName = name + "_buff_" + std::to_string(i);
        auto buf = builder.create<AIE::BufferOp>(
            state.deviceOp.getLoc(), bufTy, prodTileVal,
            mlir::StringAttr::get(ctx, symName),
            /*address=*/mlir::IntegerAttr{},
            /*initial_value=*/mlir::ElementsAttr{},
            /*mem_bank=*/mlir::IntegerAttr{});
        info.buffers.push_back(buf);
      }

      if (isAIE2) {
        {
          int lockIdx = state.lockIdCounter[prodTileVal]++;
          std::string symName = name + "_prod_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), prodTileVal, lockIdx,
              static_cast<int>(prodDepth));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          info.prodLock = lk;
        }
        {
          int lockIdx = state.lockIdCounter[prodTileVal]++;
          std::string symName = name + "_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), prodTileVal, lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          info.consLock = lk;
        }
      } else {
        for (int64_t i = 0; i < prodDepth; ++i) {
          int lockIdx = state.lockIdCounter[prodTileVal]++;
          std::string symName = name + "_lock_" + std::to_string(i);
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), prodTileVal, lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          info.aie1Locks.push_back(lk);
        }
        if (!info.aie1Locks.empty()) {
          info.prodLock = info.aie1Locks[0];
          info.consLock = info.aie1Locks[0];
        }
      }
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
            llvm::SmallVector<AIE::BufferOp> sharedBuffers;
            for (int64_t i = 0; i < depth; ++i) {
              std::string symName = name + "_buff_" + std::to_string(i);
              auto buf = builder.create<AIE::BufferOp>(
                  state.deviceOp.getLoc(), bufTy, allocTileVal,
                  mlir::StringAttr::get(ctx, symName),
                  /*address=*/mlir::IntegerAttr{},
                  /*initial_value=*/mlir::ElementsAttr{},
                  /*mem_bank=*/mlir::IntegerAttr{});
              sharedBuffers.push_back(buf);
              info.buffers.push_back(buf);
            }

            // Allocate lock(s) on the allocation tile.
            AIE::LockOp sharedProdLock;
            AIE::LockOp sharedConsLock;
            if (isAIE2) {
              {
                int lockIdx = state.lockIdCounter[allocTileVal]++;
                std::string symName = name + "_prod_lock_0";
                AIE::LockOp lk = builder.create<AIE::LockOp>(
                    state.deviceOp.getLoc(), allocTileVal, lockIdx,
                    static_cast<int>(depth));
                lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                sharedProdLock = lk;
                info.prodLock = lk;
              }
              {
                int lockIdx = state.lockIdCounter[allocTileVal]++;
                std::string symName = name + "_cons_lock_0";
                AIE::LockOp lk = builder.create<AIE::LockOp>(
                    state.deviceOp.getLoc(), allocTileVal, lockIdx,
                    static_cast<int>(0));
                lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                sharedConsLock = lk;
                info.consLock = lk;
              }
            } else {
              for (int64_t i = 0; i < depth; ++i) {
                int lockIdx = state.lockIdCounter[allocTileVal]++;
                std::string symName = name + "_lock_" + std::to_string(i);
                AIE::LockOp lk = builder.create<AIE::LockOp>(
                    state.deviceOp.getLoc(), allocTileVal, lockIdx,
                    static_cast<int>(0));
                lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                info.aie1Locks.push_back(lk);
              }
              if (!info.aie1Locks.empty()) {
                sharedProdLock = info.aie1Locks[0];
                sharedConsLock = info.aie1Locks[0];
                info.prodLock = info.aie1Locks[0];
                info.consLock = info.aie1Locks[0];
              }
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

      for (int64_t i = 0; i < prodDepth; ++i) {
        std::string symName = name + "_buff_" + std::to_string(i);
        auto buf = builder.create<AIE::BufferOp>(
            state.deviceOp.getLoc(), bufTy, prodTileVal,
            mlir::StringAttr::get(ctx, symName),
            mlir::IntegerAttr{}, mlir::ElementsAttr{}, mlir::IntegerAttr{});
        info.buffers.push_back(buf);
      }

      if (isAIE2) {
        { int lockIdx = state.lockIdCounter[prodTileVal]++;
          std::string symName = name + "_prod_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), prodTileVal, lockIdx, static_cast<int>(prodDepth));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          info.prodLock = lk; }
        { int lockIdx = state.lockIdCounter[prodTileVal]++;
          std::string symName = name + "_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), prodTileVal, lockIdx, 0);
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          info.consLock = lk; }
      } else {
        for (int64_t i = 0; i < prodDepth; ++i) {
          int lockIdx = state.lockIdCounter[prodTileVal]++;
          std::string symName = name + "_lock_" + std::to_string(i);
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), prodTileVal, lockIdx, 0);
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          info.aie1Locks.push_back(lk);
        }
        if (!info.aie1Locks.empty()) {
          info.prodLock = info.aie1Locks[0];
          info.consLock = info.aie1Locks[0];
        }
      }
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

      llvm::SmallVector<AIE::BufferOp> consBuffers;
      for (int64_t i = 0; i < depth; ++i) {
        std::string symName =
            name + bufSuffix + "_buff_" + std::to_string(i);
        auto buf = builder.create<AIE::BufferOp>(
            state.deviceOp.getLoc(), bufTy, consTileVal,
            mlir::StringAttr::get(ctx, symName),
            /*address=*/mlir::IntegerAttr{},
            /*initial_value=*/mlir::ElementsAttr{},
            /*mem_bank=*/mlir::IntegerAttr{});
        consBuffers.push_back(buf);
        if (consIdx == 0)
          info.buffers.push_back(buf);
      }

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
                llvm::SmallVector<AIE::BufferOp> pBufs;
                for (int64_t i = 0; i < prodDepth; ++i) {
                  std::string symName =
                      name + "_buff_" + std::to_string(i);
                  auto buf = builder.create<AIE::BufferOp>(
                      state.deviceOp.getLoc(), bufTy, pTileVal,
                      mlir::StringAttr::get(ctx, symName),
                      /*address=*/mlir::IntegerAttr{},
                      /*initial_value=*/mlir::ElementsAttr{},
                      /*mem_bank=*/mlir::IntegerAttr{});
                  pBufs.push_back(buf);
                }

                AIE::LockOp pProdLock, pConsLock;
                if (isAIE2) {
                  {
                    int lockIdx = state.lockIdCounter[pTileVal]++;
                    std::string symName = name + "_prod_lock_0";
                    AIE::LockOp lk = builder.create<AIE::LockOp>(
                        state.deviceOp.getLoc(), pTileVal, lockIdx,
                        static_cast<int>(prodDepth));
                    lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                    pProdLock = lk;
                  }
                  {
                    int lockIdx = state.lockIdCounter[pTileVal]++;
                    std::string symName = name + "_cons_lock_0";
                    AIE::LockOp lk = builder.create<AIE::LockOp>(
                        state.deviceOp.getLoc(), pTileVal, lockIdx,
                        static_cast<int>(0));
                    lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                    pConsLock = lk;
                  }
                } else {
                  llvm::SmallVector<AIE::LockOp> pA1Locks;
                  for (int64_t i = 0; i < prodDepth; ++i) {
                    int lockIdx = state.lockIdCounter[pTileVal]++;
                    std::string symName =
                        name + "_lock_" + std::to_string(i);
                    AIE::LockOp lk = builder.create<AIE::LockOp>(
                        state.deviceOp.getLoc(), pTileVal, lockIdx,
                        static_cast<int>(0));
                    lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                    pA1Locks.push_back(lk);
                  }
                  if (!pA1Locks.empty()) {
                    pProdLock = pA1Locks[0];
                    pConsLock = pA1Locks[0];
                  }
                  info.consumerTileAIE1Locks[pTileVal] = pA1Locks;
                }

                info.consumerTileBuffers[pTileVal] = pBufs;
                info.consumerTileLocks[pTileVal] = {pProdLock, pConsLock};

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
      AIE::LockOp thisProdLock;
      AIE::LockOp thisConsLock;
      if (isAIE2) {
        {
          int lockIdx = state.lockIdCounter[consTileVal]++;
          std::string symName = name + bufSuffix + "_prod_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), consTileVal, lockIdx,
              static_cast<int>(depth));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          thisProdLock = lk;
          if (consIdx == 0)
            info.prodLock = lk;
        }
        {
          int lockIdx = state.lockIdCounter[consTileVal]++;
          std::string symName = name + bufSuffix + "_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), consTileVal, lockIdx, 0);
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          thisConsLock = lk;
          if (consIdx == 0)
            info.consLock = lk;
        }
      } else {
        llvm::SmallVector<AIE::LockOp> theseAIE1Locks;
        for (int64_t i = 0; i < depth; ++i) {
          int lockIdx = state.lockIdCounter[consTileVal]++;
          std::string symName =
              name + bufSuffix + "_lock_" + std::to_string(i);
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), consTileVal, lockIdx, 0);
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          theseAIE1Locks.push_back(lk);
        }
        if (!theseAIE1Locks.empty()) {
          thisProdLock = theseAIE1Locks[0];
          thisConsLock = theseAIE1Locks[0];
          if (consIdx == 0) {
            info.prodLock = theseAIE1Locks[0];
            info.consLock = theseAIE1Locks[0];
            info.aie1Locks = theseAIE1Locks;
          }
        }
        info.consumerTileAIE1Locks[consTileVal] = theseAIE1Locks;
      }

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

    llvm::SmallVector<AIE::BufferOp> prodBuffers;
    for (int64_t i = 0; i < prodDepth; ++i) {
      std::string symName = name + "_buff_" + std::to_string(i);
      auto buf = builder.create<AIE::BufferOp>(
          state.deviceOp.getLoc(), bufTy, prodTileVal,
          mlir::StringAttr::get(ctx, symName),
          /*address=*/mlir::IntegerAttr{},
          /*initial_value=*/mlir::ElementsAttr{},
          /*mem_bank=*/mlir::IntegerAttr{});
      prodBuffers.push_back(buf);
    }

    AIE::LockOp pProdLock, pConsLock;
    llvm::SmallVector<AIE::LockOp> pAIE1Locks;
    if (isAIE2) {
      {
        int lockIdx = state.lockIdCounter[prodTileVal]++;
        std::string symName = name + "_prod_lock_0";
        AIE::LockOp lk = builder.create<AIE::LockOp>(
            state.deviceOp.getLoc(), prodTileVal, lockIdx,
            static_cast<int>(prodDepth));
        lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        pProdLock = lk;
      }
      {
        int lockIdx = state.lockIdCounter[prodTileVal]++;
        std::string symName = name + "_cons_lock_0";
        AIE::LockOp lk = builder.create<AIE::LockOp>(
            state.deviceOp.getLoc(), prodTileVal, lockIdx,
            static_cast<int>(0));
        lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        pConsLock = lk;
      }
    } else {
      for (int64_t i = 0; i < prodDepth; ++i) {
        int lockIdx = state.lockIdCounter[prodTileVal]++;
        std::string symName = name + "_lock_" + std::to_string(i);
        AIE::LockOp lk = builder.create<AIE::LockOp>(
            state.deviceOp.getLoc(), prodTileVal, lockIdx,
            static_cast<int>(0));
        lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        pAIE1Locks.push_back(lk);
      }
      if (!pAIE1Locks.empty()) {
        pProdLock = pAIE1Locks[0];
        pConsLock = pAIE1Locks[0];
      }
    }

    info.consumerTileLocks[prodTileVal] = {pProdLock, pConsLock};
    info.consumerTileBuffers[prodTileVal] = prodBuffers;
    if (!isAIE2)
      info.consumerTileAIE1Locks[prodTileVal] = pAIE1Locks;

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
