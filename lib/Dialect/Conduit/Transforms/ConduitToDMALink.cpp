//===- ConduitToDMALink.cpp - Phase 5-5.5: link + BD chains -----*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Phase 5: Lower conduit.link → MemTile DMA BD chain.
//   Distribute (1 src → N dsts): S2MM ingests full buffer, N MM2S channels.
//   Join (N srcs → 1 dst): N S2MM channels, one MM2S output.
//
// Phase 5.5: Generate aie.mem BD chains for simple (non-link) conduits.
//   Case C: producer MM2S, consumer S2MM.
//   Handles broadcast, MemTile, compute tiles, fused channel groups.
//
// Phase 5.5 post-pass: Link fused BD chains into circular rings.
//
//===----------------------------------------------------------------------===//

#include "ConduitToDMACommon.h"

namespace xilinx::conduit {

void linkPhase(ConduitToDMAState &state) {
  if (!state.deviceOp)
    return;

  mlir::OpBuilder &builder = *state.builder;
  mlir::MLIRContext *ctx = state.ctx;
  const bool isAIE2 = state.isAIE2Plus();
  const AIE::AIETargetModel &targetModel = *state.targetModel;
  const AIE::LockAction acqAction = state.acqAction;

  // Collect ALL link source names for Phase 5.5 skip logic.
  state.module.walk([&](Link linkOp) {
    for (auto s : linkOp.getSrcs())
      state.linkSrcNames.insert(mlir::cast<mlir::StringAttr>(s).getValue());
  });

  // -----------------------------------------------------------------------
  // Phase 5: Lower conduit.link.
  // -----------------------------------------------------------------------

  // Collect link ops to erase after processing (avoid erase-inside-walk).
  llvm::SmallVector<Link> linkOpsToErase;

  state.module.walk([&](Link linkOp) {
    builder.setInsertionPoint(state.deviceBody->getTerminator());
    mlir::Location loc = linkOp.getLoc();

    auto srcs = linkOp.getSrcs();
    auto dsts = linkOp.getDsts();
    llvm::StringRef memtileStr = linkOp.getMemtile();
    llvm::StringRef mode = linkOp.getMode();
    auto offsets = linkOp.getOffsets();

    AIE::TileOp memtile = state.lookupTile(memtileStr);
    if (!memtile) {
      linkOp.emitError("conduit-to-dma: relay tile '" + memtileStr.str() +
                       "' not found — cannot lower link op");
      state.passFailed = true;
      return;
    }

    // Verify relay tile is a MemTile.
    bool relayIsMemTile =
        targetModel.isMemTile(memtile.getCol(), memtile.getRow());
    if (!relayIsMemTile) {
      linkOp.emitError("conduit-to-dma: relay tile '" + memtileStr.str() +
                       "' is not a MemTile — conduit.link requires "
                       "a MemTile relay for DMA forwarding");
      state.passFailed = true;
      return;
    }

    std::string srcName =
        mlir::cast<mlir::StringAttr>(srcs[0]).getValue().str();
    ConduitInfo *srcInfoPtr = state.lookupConduit(srcName);
    if (!srcInfoPtr || srcInfoPtr->buffers.empty()) {
      linkOp.emitError("conduit-to-dma: src conduit '" + srcName +
                       "' buffers not allocated for link op");
      state.passFailed = true;
      return;
    }

    ConduitInfo &srcInfo = *srcInfoPtr;

    // Resolve MemTile-side buffers for the BD chain.
    mlir::Value memTileResult = memtile.getResult();
    auto memBufsIt = srcInfo.consumerTileBuffers.find(memTileResult);
    auto &linkBufs = (memBufsIt != srcInfo.consumerTileBuffers.end())
        ? memBufsIt->second : srcInfo.buffers;

    int64_t linkDepth = srcInfo.depth > 0 ? srcInfo.depth : 1;
    int64_t perBufLen =
        srcInfo.capacity > 0 ? srcInfo.capacity / linkDepth : 1;

    // Per-destination independent lock pairs on the MemTile (distribute).
    bool isDistribute = (mode == "distribute");
    unsigned numDsts = static_cast<unsigned>(dsts.size());

    llvm::SmallVector<AIE::LockOp> sliceProdLocks;
    llvm::SmallVector<AIE::LockOp> sliceConsLocks;

    mlir::Value memtileVal = memtile.getResult();

    if (isDistribute && numDsts > 0) {
      builder.setInsertionPoint(state.deviceBody->getTerminator());
      for (unsigned sliceIdx = 0; sliceIdx < numDsts; ++sliceIdx) {
        if (isAIE2) {
          {
            int lockIdx = state.lockIdCounter[memtileVal]++;
            std::string symName = srcName + "_link_prod_lock_" +
                                  std::to_string(sliceIdx);
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                state.deviceOp.getLoc(), memtileVal, lockIdx,
                static_cast<int>(linkDepth));
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            sliceProdLocks.push_back(lk);
          }
          {
            int lockIdx = state.lockIdCounter[memtileVal]++;
            std::string symName = srcName + "_link_cons_lock_" +
                                  std::to_string(sliceIdx);
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                state.deviceOp.getLoc(), memtileVal, lockIdx, 0);
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            sliceConsLocks.push_back(lk);
          }
        } else {
          int lockIdx = state.lockIdCounter[memtileVal]++;
          std::string symName = srcName + "_link_lock_" +
                                std::to_string(sliceIdx);
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), memtileVal, lockIdx, 0);
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          sliceProdLocks.push_back(lk);
          sliceConsLocks.push_back(lk);
        }
      }
    }

    // Join intermediate buffers and locks.
    llvm::SmallVector<AIE::BufferOp> joinIntermediateBuffers;
    llvm::SmallVector<AIE::LockOp> joinSrcProdLocks;
    llvm::SmallVector<AIE::LockOp> joinSrcConsLocks;
    int64_t joinDstPerBufForLen = 1;

    if (!isDistribute && !dsts.empty()) {
      std::string jDstName = mlir::cast<mlir::StringAttr>(dsts[0]).getValue().str();
      ConduitInfo *jDstInfo = state.lookupConduit(jDstName);
      if (!jDstInfo) {
        linkOp.emitWarning("conduit-to-dma: join destination conduit '")
            << jDstName << "' not found — BD lengths defaulting to 1";
      } else {
        int64_t jDstDepth = jDstInfo->depth > 0 ? jDstInfo->depth : 1;
        joinDstPerBufForLen = jDstInfo->capacity > 0 ? jDstInfo->capacity / jDstDepth : 1;

        mlir::Type intBufTy = jDstInfo->elemType;
        if (!intBufTy)
          intBufTy = mlir::MemRefType::get({joinDstPerBufForLen}, mlir::IntegerType::get(ctx, 32));

        builder.setInsertionPoint(state.deviceBody->getTerminator());
        unsigned numJoinSrcs = static_cast<unsigned>(srcs.size());

        for (int64_t i = 0; i < jDstDepth; ++i) {
          std::string symName = jDstName + "_join_buff_" + std::to_string(i);
          auto buf = builder.create<AIE::BufferOp>(state.deviceOp.getLoc(), intBufTy, memtileVal,
              mlir::StringAttr::get(ctx, symName), mlir::IntegerAttr{},
              mlir::ElementsAttr{}, mlir::IntegerAttr{});
          joinIntermediateBuffers.push_back(buf);
        }

        for (unsigned srcIdx = 0; srcIdx < numJoinSrcs; ++srcIdx) {
          if (isAIE2) {
            { int lockIdx = state.lockIdCounter[memtileVal]++;
              std::string symName = jDstName + "_join_prod_lock_" + std::to_string(srcIdx);
              AIE::LockOp lk = builder.create<AIE::LockOp>(
                  state.deviceOp.getLoc(), memtileVal, lockIdx, static_cast<int>(jDstDepth));
              lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
              joinSrcProdLocks.push_back(lk); }
            { int lockIdx = state.lockIdCounter[memtileVal]++;
              std::string symName = jDstName + "_join_cons_lock_" + std::to_string(srcIdx);
              AIE::LockOp lk = builder.create<AIE::LockOp>(
                  state.deviceOp.getLoc(), memtileVal, lockIdx, 0);
              lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
              joinSrcConsLocks.push_back(lk); }
          } else {
            int lockIdx = state.lockIdCounter[memtileVal]++;
            std::string symName = jDstName + "_join_lock_" + std::to_string(srcIdx);
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                state.deviceOp.getLoc(), memtileVal, lockIdx, 0);
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            joinSrcProdLocks.push_back(lk);
            joinSrcConsLocks.push_back(lk);
          }
        }
      }
    }

    // -----------------------------------------------------------------------
    // Emit aie.flow ops for distribute/join.
    // -----------------------------------------------------------------------
    if (isDistribute) {
      builder.setInsertionPoint(state.deviceBody->getTerminator());
      for (unsigned dstIdx = 0; dstIdx < numDsts; ++dstIdx) {
        std::string dstName =
            mlir::cast<mlir::StringAttr>(dsts[dstIdx]).getValue().str();
        ConduitInfo *dstInfo = state.lookupConduit(dstName);
        if (!dstInfo || dstInfo->consumerTileCoords.empty())
          continue;

        auto [dstConsCol, dstConsRow] = dstInfo->consumerTileCoords[0];
        AIE::TileOp dstConsTile =
            state.lookupTileByCoord(dstConsCol, dstConsRow);
        if (!dstConsTile)
          continue;

        builder.create<AIE::FlowOp>(
            state.deviceOp.getLoc(), memtileVal, AIE::WireBundle::DMA,
            static_cast<int32_t>(dstIdx), dstConsTile.getResult(),
            AIE::WireBundle::DMA, static_cast<int32_t>(0));
      }

      // Source→MemTile flow for compute-tile producers.
      {
        auto [srcProdCol, srcProdRow] = srcInfo.producerTileCoord;
        if (srcProdCol >= 0 && srcProdRow >= 2) {
          AIE::TileOp srcProdTile =
              state.lookupTileByCoord(srcProdCol, srcProdRow);
          if (srcProdTile) {
            builder.create<AIE::FlowOp>(
                state.deviceOp.getLoc(), srcProdTile.getResult(),
                AIE::WireBundle::DMA, static_cast<int32_t>(0),
                memtileVal, AIE::WireBundle::DMA,
                static_cast<int32_t>(0));
          }
        }
      }
    } else {
      // Join: per-source flows + destination flow.
      builder.setInsertionPoint(state.deviceBody->getTerminator());

      for (unsigned srcIdx = 0; srcIdx < srcs.size(); ++srcIdx) {
        std::string sName =
            mlir::cast<mlir::StringAttr>(srcs[srcIdx]).getValue().str();
        ConduitInfo *sInfo = state.lookupConduit(sName);
        if (!sInfo)
          continue;
        auto [srcProdCol, srcProdRow] = sInfo->producerTileCoord;
        if (srcProdCol < 0 || srcProdRow == 0)
          continue;
        AIE::TileOp srcProdTile = state.lookupTileByCoord(srcProdCol, srcProdRow);
        if (!srcProdTile)
          continue;
        builder.create<AIE::FlowOp>(
            state.deviceOp.getLoc(), srcProdTile.getResult(),
            AIE::WireBundle::DMA, static_cast<int32_t>(0),
            memtileVal, AIE::WireBundle::DMA,
            static_cast<int32_t>(srcIdx));
      }

      // Destination flow: memtile MM2S 0 → dst consumer.
      if (!dsts.empty()) {
        std::string dstName = mlir::cast<mlir::StringAttr>(dsts[0]).getValue().str();
        if (ConduitInfo *dstFlowInfo = state.lookupConduit(dstName)) {
          for (unsigned ci = 0; ci < dstFlowInfo->consumerTileCoords.size(); ++ci) {
            auto [consCol, consRow] = dstFlowInfo->consumerTileCoords[ci];
            AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
            if (consTile)
              builder.create<AIE::FlowOp>(state.deviceOp.getLoc(), memtileVal,
                  AIE::WireBundle::DMA, 0, consTile.getResult(),
                  AIE::WireBundle::DMA, 0);
          }
          for (auto [shimCol, shimRow] : dstFlowInfo->shimConsumerTileCoords) {
            AIE::TileOp shimTile = state.lookupTileByCoord(shimCol, shimRow);
            if (shimTile)
              builder.create<AIE::FlowOp>(state.deviceOp.getLoc(), memtileVal,
                  AIE::WireBundle::DMA, 0, shimTile.getResult(),
                  AIE::WireBundle::DMA, 0);
          }
        }
      }
    }

    // -----------------------------------------------------------------------
    // Create memtile_dma DMA block.
    // -----------------------------------------------------------------------
    builder.setInsertionPoint(state.deviceBody->getTerminator());
    auto memtileDMA =
        builder.create<AIE::MemTileDMAOp>(loc, memtileVal);
    mlir::Region &dmaRegion = memtileDMA.getBody();

    auto addBlock = [&]() -> mlir::Block * {
      return builder.createBlock(&dmaRegion);
    };

    // Build S2MM (ingest) path.
    mlir::Block *mm2sChainStartBlock = nullptr;

    if (isDistribute) {
      // Single S2MM entry, depth*numDsts BD ring.
      mlir::Block *entryBlock = addBlock();

      llvm::SmallVector<mlir::Block *> ingestBlocks;
      for (int64_t bufIdx = 0; bufIdx < linkDepth; ++bufIdx)
        for (unsigned sliceIdx = 0; sliceIdx < numDsts; ++sliceIdx)
          ingestBlocks.push_back(addBlock());

      mm2sChainStartBlock = addBlock();

      builder.setInsertionPointToEnd(entryBlock);
      builder.create<AIE::DMAStartOp>(
          loc, AIE::DMAChannelDir::S2MM,
          static_cast<int32_t>(0), static_cast<int32_t>(0),
          ingestBlocks[0], mm2sChainStartBlock);

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
            acqAction,
            state.lockAcqValue(Port::Produce, 1));
        builder.create<AIE::DMABDOp>(
            loc, linkBufs[bufIdx].getResult(),
            static_cast<int>(dstOffset), static_cast<int>(dstLen));
        builder.create<AIE::UseLockOp>(
            loc, sliceConsLocks[sliceIdx].getResult(),
            AIE::LockAction::Release,
            state.lockRelValue(Port::Produce));
        mlir::Block *nextIngest = ingestBlocks[(blkIdx + 1) % totalIngest];
        builder.create<AIE::NextBDOp>(loc, nextIngest);
      }

    } else {
      // Join: N independent S2MM channels.
      unsigned numSrcs = static_cast<unsigned>(srcs.size());
      int64_t jDepth = static_cast<int64_t>(joinIntermediateBuffers.size());
      if (jDepth == 0) jDepth = 1;

      llvm::SmallVector<int64_t> srcOffsets(numSrcs, 0);
      llvm::SmallVector<int64_t> srcLens(numSrcs, 1);
      for (unsigned srcIdx = 0; srcIdx < numSrcs; ++srcIdx) {
        if (offsets.has_value() && !offsets->empty()) {
          if (srcIdx < static_cast<unsigned>(offsets->size()))
            srcOffsets[srcIdx] = (*offsets)[srcIdx];
          if (srcIdx + 1 < static_cast<unsigned>(offsets->size()))
            srcLens[srcIdx] = (*offsets)[srcIdx + 1] - srcOffsets[srcIdx];
          else
            srcLens[srcIdx] = joinDstPerBufForLen - srcOffsets[srcIdx];
        } else {
          srcLens[srcIdx] = joinDstPerBufForLen;
        }
      }

      llvm::SmallVector<mlir::Block *> s2mmEntries(numSrcs);
      llvm::SmallVector<llvm::SmallVector<mlir::Block *>> srcIngestBlocks(numSrcs);
      for (unsigned srcIdx = 0; srcIdx < numSrcs; ++srcIdx) {
        s2mmEntries[srcIdx] = addBlock();
        for (int64_t i = 0; i < jDepth; ++i)
          srcIngestBlocks[srcIdx].push_back(addBlock());
      }
      mm2sChainStartBlock = addBlock();

      for (unsigned srcIdx = 0; srcIdx < numSrcs; ++srcIdx) {
        int64_t srcOffset = srcOffsets[srcIdx];
        int64_t srcLen = srcLens[srcIdx];
        mlir::Block *nextBlock = (srcIdx + 1 < numSrcs) ?
            s2mmEntries[srcIdx + 1] : mm2sChainStartBlock;

        builder.setInsertionPointToEnd(s2mmEntries[srcIdx]);
        builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::S2MM,
            static_cast<int32_t>(srcIdx), 0,
            srcIngestBlocks[srcIdx][0], nextBlock);

        for (int64_t i = 0; i < jDepth; ++i) {
          builder.setInsertionPointToEnd(srcIngestBlocks[srcIdx][i]);
          if (srcIdx < joinSrcProdLocks.size())
            builder.create<AIE::UseLockOp>(loc, joinSrcProdLocks[srcIdx].getResult(),
                acqAction, state.lockAcqValue(Port::Produce, 1));
          if (!joinIntermediateBuffers.empty())
            builder.create<AIE::DMABDOp>(loc,
                joinIntermediateBuffers[i % joinIntermediateBuffers.size()].getResult(),
                static_cast<int>(srcOffset), static_cast<int>(srcLen));
          if (srcIdx < joinSrcConsLocks.size())
            builder.create<AIE::UseLockOp>(loc, joinSrcConsLocks[srcIdx].getResult(),
                AIE::LockAction::Release, state.lockRelValue(Port::Produce));
          builder.create<AIE::NextBDOp>(loc, srcIngestBlocks[srcIdx][(i + 1) % jDepth]);
        }
      }
    }

    // Build MM2S send chains.
    mlir::Block *prevChainBlock = mm2sChainStartBlock ? mm2sChainStartBlock : addBlock();
    mlir::Block *endBlock = nullptr;

    if (!isDistribute && !joinIntermediateBuffers.empty()) {
      // Join MM2S: single channel, depth×N interleaved BDs.
      unsigned numJoinSrcs = static_cast<unsigned>(srcs.size());
      int64_t jDepth = static_cast<int64_t>(joinIntermediateBuffers.size());
      unsigned totalBDs = static_cast<unsigned>(jDepth) * numJoinSrcs;

      llvm::SmallVector<int64_t> mm2sOffsets(numJoinSrcs, 0);
      llvm::SmallVector<int64_t> mm2sLens(numJoinSrcs, 1);
      for (unsigned s = 0; s < numJoinSrcs; ++s) {
        if (offsets.has_value() && !offsets->empty()) {
          if (s < static_cast<unsigned>(offsets->size()))
            mm2sOffsets[s] = (*offsets)[s];
          if (s + 1 < static_cast<unsigned>(offsets->size()))
            mm2sLens[s] = (*offsets)[s + 1] - mm2sOffsets[s];
          else
            mm2sLens[s] = joinDstPerBufForLen - mm2sOffsets[s];
        } else {
          mm2sLens[s] = joinDstPerBufForLen;
        }
      }

      llvm::SmallVector<mlir::Block *> sendBDBlocks;
      for (unsigned i = 0; i < totalBDs; ++i)
        sendBDBlocks.push_back(addBlock());
      endBlock = addBlock();

      builder.setInsertionPointToEnd(prevChainBlock);
      builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::MM2S,
          0, 0, sendBDBlocks[0], endBlock);

      for (unsigned bdIdx = 0; bdIdx < totalBDs; ++bdIdx) {
        unsigned bufIdx = bdIdx / numJoinSrcs;
        unsigned srcIdx = bdIdx % numJoinSrcs;
        builder.setInsertionPointToEnd(sendBDBlocks[bdIdx]);
        if (srcIdx < joinSrcConsLocks.size())
          builder.create<AIE::UseLockOp>(loc, joinSrcConsLocks[srcIdx].getResult(),
              acqAction, state.lockAcqValue(Port::Consume, 1));
        builder.create<AIE::DMABDOp>(loc,
            joinIntermediateBuffers[bufIdx % joinIntermediateBuffers.size()].getResult(),
            static_cast<int>(mm2sOffsets[srcIdx]), static_cast<int>(mm2sLens[srcIdx]));
        if (srcIdx < joinSrcProdLocks.size())
          builder.create<AIE::UseLockOp>(loc, joinSrcProdLocks[srcIdx].getResult(),
              AIE::LockAction::Release, state.lockRelValue(Port::Consume));
        builder.create<AIE::NextBDOp>(loc, sendBDBlocks[(bdIdx + 1) % totalBDs]);
      }
    } else {
      // Distribute MM2S: per-destination chains.
      for (unsigned dstIdx = 0; dstIdx < dsts.size(); ++dstIdx) {
        int64_t thisDstDepth = linkDepth;
        int64_t dstOffset = 0, dstLen = perBufLen;
        if (offsets.has_value() && !offsets->empty()) {
          if (dstIdx < static_cast<unsigned>(offsets->size()))
            dstOffset = (*offsets)[dstIdx];
          if (dstIdx + 1 < static_cast<unsigned>(offsets->size()))
            dstLen = (*offsets)[dstIdx + 1] - dstOffset;
          else
            dstLen = perBufLen - dstOffset;
        }

        AIE::LockOp mm2sAcqLock, mm2sRelLock;
        if (dstIdx < sliceConsLocks.size()) {
          mm2sAcqLock = sliceConsLocks[dstIdx];
          mm2sRelLock = sliceProdLocks[dstIdx];
        }

        llvm::SmallVector<mlir::Block *> sendBDBlocks;
        for (int64_t i = 0; i < thisDstDepth; ++i)
          sendBDBlocks.push_back(addBlock());

        mlir::Block *nextChainBlock;
        if (dstIdx + 1 == dsts.size()) {
          endBlock = addBlock();
          nextChainBlock = endBlock;
        } else {
          nextChainBlock = addBlock();
        }

        builder.setInsertionPointToEnd(prevChainBlock);
        builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::MM2S,
            static_cast<int32_t>(dstIdx), 0, sendBDBlocks[0], nextChainBlock);

        for (int64_t i = 0; i < thisDstDepth; ++i) {
          builder.setInsertionPointToEnd(sendBDBlocks[i]);
          if (mm2sAcqLock)
            builder.create<AIE::UseLockOp>(loc, mm2sAcqLock.getResult(), acqAction,
                state.lockAcqValue(Port::Consume, 1));
          if (!linkBufs.empty())
            builder.create<AIE::DMABDOp>(loc,
                linkBufs[i % linkBufs.size()].getResult(),
                static_cast<int>(dstOffset), static_cast<int>(dstLen));
          if (mm2sRelLock)
            builder.create<AIE::UseLockOp>(loc, mm2sRelLock.getResult(),
                AIE::LockAction::Release,
                state.lockRelValue(Port::Consume));
          builder.create<AIE::NextBDOp>(loc, sendBDBlocks[(i + 1) % thisDstDepth]);
        }
        prevChainBlock = nextChainBlock;
      }
    }

    if (!endBlock)
      endBlock = addBlock();

    builder.setInsertionPointToEnd(endBlock);
    builder.create<AIE::EndOp>(loc);

    linkOpsToErase.push_back(linkOp);
  });

  // Erase link ops after processing (collect-then-erase pattern).
  for (auto op : llvm::reverse(linkOpsToErase))
    op.erase();

  if (state.passFailed)
    return;

  // -----------------------------------------------------------------------
  // Phase 5.5: aie.mem BD chains for simple (non-link) conduits.
  // -----------------------------------------------------------------------

  // Pre-compute used DMA channels per tile (avoids O(n²) scan).
  state.deviceOp.walk([&](AIE::DMAStartOp dmaStart) {
    mlir::Value parentTile;
    if (auto memOp =
            mlir::dyn_cast<AIE::MemOp>(dmaStart->getParentOp()))
      parentTile = memOp.getTile();
    else if (auto mtOp =
                 mlir::dyn_cast<AIE::MemTileDMAOp>(dmaStart->getParentOp()))
      parentTile = mtOp.getTile();
    if (!parentTile)
      return;
    if (dmaStart.getChannelDir() == AIE::DMAChannelDir::MM2S)
      state.preUsedMM2SChannels[parentTile].insert(
          static_cast<int32_t>(dmaStart.getChannelIndex()));
    else
      state.preUsedS2MMChannels[parentTile].insert(
          static_cast<int32_t>(dmaStart.getChannelIndex()));
  });

  // Pre-compute tile → DMA region map to avoid O(n²) walks inside the
  // conduitMap loop.  Updated when new DMA ops are created below.
  llvm::DenseMap<mlir::Value, mlir::Region *> tileToDMARegion;
  state.deviceOp.walk([&](AIE::MemOp memOp) {
    tileToDMARegion[memOp.getTile()] = &memOp.getBody();
  });
  state.deviceOp.walk([&](AIE::MemTileDMAOp mtOp) {
    tileToDMARegion[mtOp.getTile()] = &mtOp.getBody();
  });

  for (auto &[name, info] : state.conduitMap) {
    if (info.buffers.empty() || !info.prodLock || !info.consLock)
      continue;

    // Handle link source conduits: emit aie.mem MM2S on producer compute tile.
    if (state.linkSrcNames.count(name)) {
      if (state.linkSrcNamesEarly.count(name)) {
        // Distribute source with compute producer.
        auto [prodCol, prodRow] = info.producerTileCoord;
        if (prodRow < 2)
          continue;

        AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
        if (!prodTile)
          continue;

        mlir::Value prodTileVal = prodTile.getResult();

        auto bufIt = info.consumerTileBuffers.find(prodTileVal);
        auto lockIt = info.consumerTileLocks.find(prodTileVal);
        if (bufIt == info.consumerTileBuffers.end() ||
            bufIt->second.empty() ||
            lockIt == info.consumerTileLocks.end())
          continue;

        llvm::SmallVector<AIE::BufferOp> &prodBuffers = bufIt->second;
        AIE::LockOp pProdLock = lockIt->second.first;
        AIE::LockOp pConsLock = lockIt->second.second;

        int64_t depth = info.depth > 0 ? info.depth : 1;
        int64_t perBufLen = info.capacity > 0 ? info.capacity / depth : 1;

        builder.setInsertionPoint(state.deviceBody->getTerminator());
        auto memOp =
            builder.create<AIE::MemOp>(state.deviceOp.getLoc(), prodTileVal);
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
        builder.create<AIE::DMAStartOp>(state.deviceOp.getLoc(),
                                        AIE::DMAChannelDir::MM2S,
                                        static_cast<int32_t>(0),
                                        static_cast<int32_t>(0),
                                        bdBlocks[0], endMemBlock);

        llvm::SmallVector<AIE::LockOp> *pA1Locks = nullptr;
        {
          auto a1It = info.consumerTileAIE1Locks.find(prodTileVal);
          if (a1It != info.consumerTileAIE1Locks.end() && !a1It->second.empty())
            pA1Locks = &a1It->second;
        }
        for (int64_t i = 0; i < depth; ++i) {
          builder.setInsertionPointToEnd(bdBlocks[i]);
          mlir::Value blockAcq =
              isAIE2 ? pConsLock.getResult()
                     : (pA1Locks && !pA1Locks->empty()
                            ? (*pA1Locks)[i % pA1Locks->size()].getResult()
                            : pConsLock.getResult());
          mlir::Value blockRel =
              isAIE2 ? pProdLock.getResult() : blockAcq;
          builder.create<AIE::UseLockOp>(
              state.deviceOp.getLoc(), blockAcq, acqAction,
              state.lockAcqValue(Port::Consume, 1));
          builder.create<AIE::DMABDOp>(
              state.deviceOp.getLoc(),
              prodBuffers[i % prodBuffers.size()].getResult(),
              0, static_cast<int>(perBufLen));
          builder.create<AIE::UseLockOp>(
              state.deviceOp.getLoc(), blockRel,
              AIE::LockAction::Release,
              state.lockRelValue(Port::Consume));
          builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                        bdBlocks[(i + 1) % depth]);
        }
        builder.setInsertionPointToEnd(endMemBlock);
        builder.create<AIE::EndOp>(state.deviceOp.getLoc());
        continue;
      }

      // Join sources: compute producer sending to memtile.
      if (!state.linkJoinSrcNames.count(name))
        continue;

      auto [prodCol, prodRow] = info.producerTileCoord;
      if (prodRow < 2)
        continue;

      AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
      if (!prodTile)
        continue;

      int64_t depth = info.depth > 0 ? info.depth : 1;
      int64_t perBufLen = info.capacity > 0 ? info.capacity / depth : 1;

      builder.setInsertionPoint(state.deviceBody->getTerminator());
      auto memOp =
          builder.create<AIE::MemOp>(state.deviceOp.getLoc(), prodTile.getResult());
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
      builder.create<AIE::DMAStartOp>(state.deviceOp.getLoc(),
                                      AIE::DMAChannelDir::MM2S,
                                      static_cast<int32_t>(0),
                                      static_cast<int32_t>(0),
                                      bdBlocks[0], endMemBlock);

      for (int64_t i = 0; i < depth; ++i) {
        builder.setInsertionPointToEnd(bdBlocks[i]);
        mlir::Value blockLock = isAIE2 ? info.consLock.getResult()
            : (info.aie1Locks.empty()
                   ? info.consLock.getResult()
                   : info.aie1Locks[i % info.aie1Locks.size()].getResult());
        mlir::Value blockRelLock = isAIE2 ? info.prodLock.getResult()
            : blockLock;
        builder.create<AIE::UseLockOp>(
            state.deviceOp.getLoc(), blockLock, acqAction,
            state.lockAcqValue(Port::Consume, 1));
        builder.create<AIE::DMABDOp>(
            state.deviceOp.getLoc(),
            info.buffers[i % info.buffers.size()].getResult(),
            0, static_cast<int>(perBufLen));
        builder.create<AIE::UseLockOp>(
            state.deviceOp.getLoc(), blockRelLock,
            AIE::LockAction::Release,
            state.lockRelValue(Port::Consume));
        builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                      bdBlocks[(i + 1) % depth]);
      }
      builder.setInsertionPointToEnd(endMemBlock);
      builder.create<AIE::EndOp>(state.deviceOp.getLoc());
      continue;
    }

    // Skip shared memory conduits.
    if (info.sharedMemory)
      continue;

    // Case C: non-adjacent producer MM2S on producer tile.
    {
      auto [prodCol, prodRow] = info.producerTileCoord;
      if (prodCol >= 0 && prodRow >= 1 &&
          !state.linkSrcNames.count(name) &&
          !info.consumerTileCoords.empty()) {
        AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
        if (prodTile) {
          mlir::Value prodTileVal = prodTile.getResult();
          auto bufIt = info.consumerTileBuffers.find(prodTileVal);
          if (bufIt != info.consumerTileBuffers.end() &&
              !bufIt->second.empty()) {
            llvm::SmallVector<AIE::BufferOp> &prodBuffers = bufIt->second;
            int64_t depth = info.depth > 0 ? info.depth : 1;
            int64_t perBufLen =
                info.capacity > 0 ? info.capacity / depth : 1;

            AIE::LockOp mm2sAcqLock, mm2sRelLock;
            llvm::SmallVector<AIE::LockOp> *prodAIE1Locks = nullptr;
            {
              auto lockIt = info.consumerTileLocks.find(prodTileVal);
              if (lockIt != info.consumerTileLocks.end()) {
                mm2sAcqLock = lockIt->second.second; // consLock
                mm2sRelLock = lockIt->second.first;  // prodLock
              }
              auto aie1It = info.consumerTileAIE1Locks.find(prodTileVal);
              if (aie1It != info.consumerTileAIE1Locks.end() &&
                  !aie1It->second.empty())
                prodAIE1Locks = &aie1It->second;
            }

            if (mm2sAcqLock && mm2sRelLock) {
              bool prodIsMemTile =
                  targetModel.isMemTile(prodCol, prodRow);

              int32_t mm2sChannel = 0;
              {
                auto chIt = state.conduitMM2SChannel.find(name);
                if (chIt != state.conduitMM2SChannel.end()) {
                  mm2sChannel = chIt->second;
                } else {
                  auto &usedCh = state.preUsedMM2SChannels[prodTileVal];
                  while (usedCh.count(mm2sChannel))
                    ++mm2sChannel;
                  usedCh.insert(mm2sChannel);
                }
              }

              // Find existing DMA op for this tile (pre-computed map).
              mlir::Region *existingDMARegion = nullptr;
              {
                auto dmaIt = tileToDMARegion.find(prodTileVal);
                if (dmaIt != tileToDMARegion.end())
                  existingDMARegion = dmaIt->second;
              }

              bool isFusedNonFirst = false;
              if (!info.fuseGroup.empty()) {
                auto &members = state.fuseGroupMembers[info.fuseGroup];
                isFusedNonFirst = (!members.empty() &&
                                   members.front() != name);
              }

              if (existingDMARegion) {
                mlir::Region &memRegion = *existingDMARegion;
                mlir::Block *endBlock = nullptr;
                for (mlir::Block &block : memRegion)
                  for (mlir::Operation &opInBlock : block)
                    if (mlir::isa<AIE::EndOp>(opInBlock))
                      endBlock = &block;

                if (endBlock) {
                  auto addBlock = [&]() -> mlir::Block * {
                    return builder.createBlock(&memRegion);
                  };
                  llvm::SmallVector<mlir::Block *> bdBlocks;
                  for (int64_t i = 0; i < depth; ++i)
                    bdBlocks.push_back(addBlock());
                  mlir::Block *newEndBlock = addBlock();

                  if (isFusedNonFirst) {
                    // Non-first fused member: no new dma_start.
                  } else {
                    mlir::Operation *oldEnd = endBlock->getTerminator();
                    builder.setInsertionPointToEnd(endBlock);
                    oldEnd->erase();
                    builder.create<AIE::DMAStartOp>(
                        state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
                        mm2sChannel, static_cast<int32_t>(0),
                        bdBlocks[0], newEndBlock);
                  }

                  for (int64_t i = 0; i < depth; ++i) {
                    builder.setInsertionPointToEnd(bdBlocks[i]);
                    mlir::Value blockAcq =
                        isAIE2 ? mm2sAcqLock.getResult()
                               : (prodAIE1Locks && !prodAIE1Locks->empty()
                                      ? (*prodAIE1Locks)[i % prodAIE1Locks->size()].getResult()
                                      : mm2sAcqLock.getResult());
                    mlir::Value blockRel =
                        isAIE2 ? mm2sRelLock.getResult() : blockAcq;
                    builder.create<AIE::UseLockOp>(
                        state.deviceOp.getLoc(), blockAcq, acqAction,
                        state.lockAcqValue(Port::Consume, 1));
                    builder.create<AIE::DMABDOp>(
                        state.deviceOp.getLoc(),
                        prodBuffers[i % prodBuffers.size()].getResult(),
                        0, static_cast<int>(perBufLen));
                    builder.create<AIE::UseLockOp>(
                        state.deviceOp.getLoc(), blockRel,
                        AIE::LockAction::Release,
                        state.lockRelValue(Port::Consume));
                    builder.create<AIE::NextBDOp>(
                        state.deviceOp.getLoc(), bdBlocks[(i + 1) % depth]);
                  }
                  builder.setInsertionPointToEnd(newEndBlock);
                  builder.create<AIE::EndOp>(state.deviceOp.getLoc());

                  if (!info.fuseGroup.empty())
                    state.conduitBDRange[name] = {bdBlocks.front(), bdBlocks.back()};
                }
              } else {
                builder.setInsertionPoint(state.deviceBody->getTerminator());
                mlir::Region *dmaRegionPtr;
                if (prodIsMemTile) {
                  auto mtOp = builder.create<AIE::MemTileDMAOp>(
                      state.deviceOp.getLoc(), prodTileVal);
                  dmaRegionPtr = &mtOp.getBody();
                } else {
                  auto memOp = builder.create<AIE::MemOp>(
                      state.deviceOp.getLoc(), prodTileVal);
                  dmaRegionPtr = &memOp.getBody();
                }
                tileToDMARegion[prodTileVal] = dmaRegionPtr;
                mlir::Region &memRegion = *dmaRegionPtr;
                auto addBlock = [&]() -> mlir::Block * {
                  return builder.createBlock(&memRegion);
                };
                mlir::Block *dmaStartBlock = nullptr;
                if (!isFusedNonFirst)
                  dmaStartBlock = addBlock();
                llvm::SmallVector<mlir::Block *> bdBlocks;
                for (int64_t i = 0; i < depth; ++i)
                  bdBlocks.push_back(addBlock());
                mlir::Block *endBlock = addBlock();

                if (!isFusedNonFirst) {
                  builder.setInsertionPointToEnd(dmaStartBlock);
                  builder.create<AIE::DMAStartOp>(
                      state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
                      mm2sChannel, static_cast<int32_t>(0),
                      bdBlocks[0], endBlock);
                }

                for (int64_t i = 0; i < depth; ++i) {
                  builder.setInsertionPointToEnd(bdBlocks[i]);
                  mlir::Value blockAcq =
                      isAIE2 ? mm2sAcqLock.getResult()
                             : (prodAIE1Locks && !prodAIE1Locks->empty()
                                    ? (*prodAIE1Locks)[i % prodAIE1Locks->size()].getResult()
                                    : mm2sAcqLock.getResult());
                  mlir::Value blockRel =
                      isAIE2 ? mm2sRelLock.getResult() : blockAcq;
                  builder.create<AIE::UseLockOp>(
                      state.deviceOp.getLoc(), blockAcq, acqAction,
                      state.lockAcqValue(Port::Consume, 1));
                  builder.create<AIE::DMABDOp>(
                      state.deviceOp.getLoc(),
                      prodBuffers[i % prodBuffers.size()].getResult(),
                      0, static_cast<int>(perBufLen));
                  builder.create<AIE::UseLockOp>(
                      state.deviceOp.getLoc(), blockRel,
                      AIE::LockAction::Release,
                      state.lockRelValue(Port::Consume));
                  builder.create<AIE::NextBDOp>(
                      state.deviceOp.getLoc(), bdBlocks[(i + 1) % depth]);
                }
                builder.setInsertionPointToEnd(endBlock);
                builder.create<AIE::EndOp>(state.deviceOp.getLoc());

                if (!info.fuseGroup.empty())
                  state.conduitBDRange[name] = {bdBlocks.front(), bdBlocks.back()};
              }
            }
          }
        }
      }
    }

    // Consumer S2MM.
    bool isProducerToShim = info.consumerTileCoords.empty() &&
                            !info.shimConsumerTileCoords.empty();

    if (isProducerToShim) {
      // Case B: compute tile sends MM2S to shim.
      auto [prodCol, prodRow] = info.producerTileCoord;
      if (prodCol < 0 || prodRow == 0)
        continue;
      AIE::TileOp dmaHostTile = state.lookupTileByCoord(prodCol, prodRow);
      if (!dmaHostTile)
        continue;
      if (prodRow < 2)
        continue;

      int64_t depth = info.depth > 0 ? info.depth : 1;
      int64_t perBufLen = info.capacity > 0 ? info.capacity / depth : 1;

      builder.setInsertionPoint(state.deviceBody->getTerminator());
      auto memOp = builder.create<AIE::MemOp>(state.deviceOp.getLoc(),
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
      builder.create<AIE::DMAStartOp>(state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
                                      static_cast<int32_t>(0),
                                      static_cast<int32_t>(0),
                                      bdBlocks[0], endMemBlock);

      for (int64_t i = 0; i < depth; ++i) {
        builder.setInsertionPointToEnd(bdBlocks[i]);
        mlir::Value blockAcqVal = isAIE2 ? info.consLock.getResult()
            : (info.aie1Locks.empty()
                   ? info.consLock.getResult()
                   : info.aie1Locks[i % info.aie1Locks.size()].getResult());
        mlir::Value blockRelVal = isAIE2 ? info.prodLock.getResult()
            : (info.aie1Locks.empty()
                   ? info.prodLock.getResult()
                   : info.aie1Locks[i % info.aie1Locks.size()].getResult());
        builder.create<AIE::UseLockOp>(state.deviceOp.getLoc(), blockAcqVal,
                                       acqAction,
                                       state.lockAcqValue(Port::Consume, 1));
        builder.create<AIE::DMABDOp>(state.deviceOp.getLoc(),
                                     info.buffers[i].getResult(), 0,
                                     static_cast<int>(perBufLen));
        builder.create<AIE::UseLockOp>(state.deviceOp.getLoc(), blockRelVal,
                                       AIE::LockAction::Release,
                                       state.lockRelValue(Port::Consume));
        builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                      bdBlocks[(i + 1) % depth]);
      }
      builder.setInsertionPointToEnd(endMemBlock);
      builder.create<AIE::EndOp>(state.deviceOp.getLoc());

    } else {
      // Case A: S2MM on consumer tile(s).
      if (info.consumerTileCoords.empty())
        continue;

      int64_t depth = info.depth > 0 ? info.depth : 1;
      int64_t perBufLen = info.capacity > 0 ? info.capacity / depth : 1;

      for (unsigned consIdx = 0; consIdx < info.consumerTileCoords.size();
           ++consIdx) {
        auto [consCol, consRow] = info.consumerTileCoords[consIdx];
        AIE::TileOp dmaHostTile = state.lookupTileByCoord(consCol, consRow);
        if (!dmaHostTile)
          continue;
        if (consRow == 0)
          continue;

        bool consIsMemTile = targetModel.isMemTile(consCol, consRow);

        // Resolve per-consumer-tile resources.
        llvm::SmallVector<AIE::BufferOp> *tileBuffers = &info.buffers;
        AIE::LockOp tileProdLock = info.prodLock;
        AIE::LockOp tileConsLock = info.consLock;
        llvm::SmallVector<AIE::LockOp> *tileAIE1Locks = &info.aie1Locks;
        {
          mlir::Value consTileVal = dmaHostTile.getResult();
          auto bufIt = info.consumerTileBuffers.find(consTileVal);
          if (bufIt != info.consumerTileBuffers.end() && !bufIt->second.empty())
            tileBuffers = &bufIt->second;
          auto lockIt = info.consumerTileLocks.find(consTileVal);
          if (lockIt != info.consumerTileLocks.end()) {
            tileProdLock = lockIt->second.first;
            tileConsLock = lockIt->second.second;
          }
          auto aie1It = info.consumerTileAIE1Locks.find(consTileVal);
          if (aie1It != info.consumerTileAIE1Locks.end() && !aie1It->second.empty())
            tileAIE1Locks = &aie1It->second;
        }

        if (!tileProdLock || !tileConsLock || tileBuffers->empty())
          continue;

        int32_t s2mmChannel = 0;
        {
          auto chIt = state.conduitConsS2MMChannel.find({name, consIdx});
          if (chIt != state.conduitConsS2MMChannel.end()) {
            s2mmChannel = chIt->second;
          } else {
            auto &usedCh = state.preUsedS2MMChannels[dmaHostTile.getResult()];
            while (usedCh.count(s2mmChannel))
              ++s2mmChannel;
            usedCh.insert(s2mmChannel);
          }
        }

        // Find or create DMA op for this consumer tile (pre-computed map).
        mlir::Value consTileVal2 = dmaHostTile.getResult();
        mlir::Region *dmaRegionPtr = nullptr;
        {
          auto dmaIt = tileToDMARegion.find(consTileVal2);
          if (dmaIt != tileToDMARegion.end())
            dmaRegionPtr = dmaIt->second;
        }
        if (!dmaRegionPtr) {
          builder.setInsertionPoint(state.deviceBody->getTerminator());
          if (consIsMemTile) {
            auto mtOp = builder.create<AIE::MemTileDMAOp>(
                state.deviceOp.getLoc(), consTileVal2);
            dmaRegionPtr = &mtOp.getBody();
          } else {
            auto memOp = builder.create<AIE::MemOp>(
                state.deviceOp.getLoc(), consTileVal2);
            dmaRegionPtr = &memOp.getBody();
          }
          tileToDMARegion[consTileVal2] = dmaRegionPtr;
        }
        mlir::Region &memRegion = *dmaRegionPtr;
        auto addMemBlock = [&]() -> mlir::Block * {
          return builder.createBlock(&memRegion);
        };

        mlir::Block *existingEndBlock = nullptr;
        if (!memRegion.empty()) {
          for (mlir::Block &block : memRegion)
            for (mlir::Operation &opInBlock : block)
              if (mlir::isa<AIE::EndOp>(opInBlock))
                existingEndBlock = &block;
        }

        llvm::SmallVector<mlir::Block *> bdBlocks;
        for (int64_t i = 0; i < depth; ++i)
          bdBlocks.push_back(addMemBlock());
        mlir::Block *endMemBlock = addMemBlock();

        if (existingEndBlock) {
          mlir::Operation *oldEnd = existingEndBlock->getTerminator();
          builder.setInsertionPointToEnd(existingEndBlock);
          oldEnd->erase();
          builder.create<AIE::DMAStartOp>(state.deviceOp.getLoc(), AIE::DMAChannelDir::S2MM,
                                          s2mmChannel, static_cast<int32_t>(0),
                                          bdBlocks[0], endMemBlock);
        } else {
          mlir::Block *dmaStartBlock = addMemBlock();
          dmaStartBlock->moveBefore(&memRegion.front());
          builder.setInsertionPointToEnd(dmaStartBlock);
          builder.create<AIE::DMAStartOp>(state.deviceOp.getLoc(), AIE::DMAChannelDir::S2MM,
                                          s2mmChannel, static_cast<int32_t>(0),
                                          bdBlocks[0], endMemBlock);
        }

        for (int64_t i = 0; i < depth; ++i) {
          builder.setInsertionPointToEnd(bdBlocks[i]);
          mlir::Value blockLockAcq = isAIE2 ? tileProdLock.getResult()
              : (tileAIE1Locks->empty()
                     ? tileProdLock.getResult()
                     : (*tileAIE1Locks)[i % tileAIE1Locks->size()].getResult());
          mlir::Value blockLockRel = isAIE2 ? tileConsLock.getResult()
              : (tileAIE1Locks->empty()
                     ? tileConsLock.getResult()
                     : (*tileAIE1Locks)[i % tileAIE1Locks->size()].getResult());
          builder.create<AIE::UseLockOp>(
              state.deviceOp.getLoc(), blockLockAcq,
              acqAction, state.lockAcqValue(Port::Produce, 1));
          builder.create<AIE::DMABDOp>(
              state.deviceOp.getLoc(), (*tileBuffers)[i % tileBuffers->size()].getResult(),
              0, static_cast<int>(perBufLen));
          builder.create<AIE::UseLockOp>(
              state.deviceOp.getLoc(), blockLockRel,
              AIE::LockAction::Release, state.lockRelValue(Port::Produce));
          builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                        bdBlocks[(i + 1) % depth]);
        }
        builder.setInsertionPointToEnd(endMemBlock);
        builder.create<AIE::EndOp>(state.deviceOp.getLoc());
      }
    }
  }

  // -----------------------------------------------------------------------
  // Phase 5.5 post-pass: link fused BD chains.
  // -----------------------------------------------------------------------
  for (auto &[groupLabel, members] : state.fuseGroupMembers) {
    if (members.size() < 2)
      continue;

    llvm::SmallVector<std::pair<mlir::Block *, mlir::Block *>, 4> ranges;
    for (const auto &memberName : members) {
      auto it = state.conduitBDRange.find(memberName);
      if (it == state.conduitBDRange.end())
        continue;
      ranges.push_back(it->second);
    }
    if (ranges.size() < 2)
      continue;

    for (unsigned i = 0; i < ranges.size(); ++i) {
      mlir::Block *lastBD = ranges[i].second;
      mlir::Block *nextFirstBD = ranges[(i + 1) % ranges.size()].first;

      mlir::Operation *term = lastBD->getTerminator();
      if (!term || !mlir::isa<AIE::NextBDOp>(term))
        continue;
      auto nextBDOp = mlir::cast<AIE::NextBDOp>(term);
      builder.setInsertionPoint(nextBDOp);
      builder.create<AIE::NextBDOp>(nextBDOp.getLoc(), nextFirstBD);
      nextBDOp.erase();
    }
  }
}

} // namespace xilinx::conduit
