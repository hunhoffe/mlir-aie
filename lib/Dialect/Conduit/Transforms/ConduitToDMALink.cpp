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
// Phase 5: Lower conduit.distribute/join/forward → MemTile DMA BD chain.
//   Distribute (1 src → N dsts): S2MM ingests full buffer, N MM2S channels.
//   Join (N srcs → 1 dst): N S2MM channels, one MM2S output.
//   Forward (1 src → 1 dst): treated as distribute with 1 destination.
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
  (void)ctx; // suppress unused warning when not used in all paths

  // Collect ALL link source and destination names for Phase 5.5 skip logic.
  state.module.walk([&](Distribute op) {
    for (auto s : op.getSrcs())
      state.linkSrcNames.insert(mlir::cast<mlir::FlatSymbolRefAttr>(s).getValue());
    for (auto d : op.getDsts())
      state.linkDstNames.insert(mlir::cast<mlir::FlatSymbolRefAttr>(d).getValue());
  });
  state.module.walk([&](Join op) {
    for (auto s : op.getSrcs())
      state.linkSrcNames.insert(mlir::cast<mlir::FlatSymbolRefAttr>(s).getValue());
    for (auto d : op.getDsts())
      state.linkDstNames.insert(mlir::cast<mlir::FlatSymbolRefAttr>(d).getValue());
  });
  state.module.walk([&](Forward op) {
    for (auto s : op.getSrcs())
      state.linkSrcNames.insert(mlir::cast<mlir::FlatSymbolRefAttr>(s).getValue());
    for (auto d : op.getDsts())
      state.linkDstNames.insert(mlir::cast<mlir::FlatSymbolRefAttr>(d).getValue());
  });

  // -----------------------------------------------------------------------
  // Phase 5: Lower conduit.distribute / conduit.join / conduit.forward.
  //
  // LinkAdapter unifies the three op types so the lowering body can be shared.
  // -----------------------------------------------------------------------

  struct LinkAdapter {
    mlir::Operation *op;
    mlir::ArrayAttr srcs;
    mlir::ArrayAttr dsts;
    llvm::StringRef memtileStr;
    bool isDistribute; // true for Distribute + Forward; false for Join
    std::optional<llvm::ArrayRef<int64_t>> offsets;

    mlir::Location getLoc() const { return op->getLoc(); }
    mlir::InFlightDiagnostic emitError(const llvm::Twine &msg) const {
      return op->emitError(msg);
    }
    mlir::InFlightDiagnostic emitWarning(const llvm::Twine &msg) const {
      return op->emitWarning(msg);
    }
  };

  llvm::SmallVector<mlir::Operation *> linkOpsToErase;
  llvm::SmallVector<LinkAdapter> linkAdapters;

  state.module.walk([&](Distribute distOp) {
    LinkAdapter a;
    a.op = distOp.getOperation();
    a.srcs = distOp.getSrcs();
    a.dsts = distOp.getDsts();
    a.memtileStr = distOp.getMemtile();
    a.isDistribute = true;
    a.offsets = distOp.getOffsets();
    linkAdapters.push_back(a);
  });
  state.module.walk([&](Join joinOp) {
    LinkAdapter a;
    a.op = joinOp.getOperation();
    a.srcs = joinOp.getSrcs();
    a.dsts = joinOp.getDsts();
    a.memtileStr = joinOp.getMemtile();
    a.isDistribute = false;
    a.offsets = joinOp.getOffsets();
    linkAdapters.push_back(a);
  });
  state.module.walk([&](Forward fwdOp) {
    LinkAdapter a;
    a.op = fwdOp.getOperation();
    a.srcs = fwdOp.getSrcs();
    a.dsts = fwdOp.getDsts();
    a.memtileStr = fwdOp.getMemtile();
    a.isDistribute = true; // forward = distribute with 1 dst
    a.offsets = fwdOp.getOffsets();
    linkAdapters.push_back(a);
  });

  // Map from memtile tile value → existing MemTileDMAOp for merging.
  // When multiple link groups reference the same memtile, we merge them
  // into a single MemTileDMAOp with non-overlapping DMA channel numbers.
  llvm::DenseMap<mlir::Value, AIE::MemTileDMAOp> memtileDMAMap;

  for (auto &linkOp : linkAdapters) {
    builder.setInsertionPoint(state.deviceBody->getTerminator());
    mlir::Location loc = linkOp.getLoc();

    auto srcs = linkOp.srcs;
    auto dsts = linkOp.dsts;
    llvm::StringRef memtileStr = linkOp.memtileStr;
    auto offsets = linkOp.offsets;

    AIE::TileOp memtile = state.lookupTile(memtileStr);
    if (!memtile) {
      linkOp.emitError("conduit-to-dma: relay tile '" + memtileStr.str() +
                       "' not found — cannot lower link op");
      state.passFailed = true;
      continue;
    }

    // Verify relay tile is a MemTile.
    bool relayIsMemTile =
        targetModel.isMemTile(memtile.getCol(), memtile.getRow());

    // CoreTile relay path: when the relay is a compute tile (not a MemTile),
    // lower as a compute-tile DMA relay. The relay tile's aie.mem gets both
    // S2MM (receive from upstream) and MM2S (send to consumer), sharing the
    // source conduit's relay buffers and locks.
    //
    // Two sub-cases:
    //   (a) dst conduit is shared-memory (info.sharedMemory=true): only S2MM
    //       on the relay tile; consumer reads buffers directly via shared mem.
    //   (b) dst conduit uses DMA: S2MM + MM2S on relay, S2MM on consumer,
    //       plus aie.flow from relay MM2S → consumer S2MM.
    if (!relayIsMemTile) {
      std::string coreRelayName =
          mlir::cast<mlir::FlatSymbolRefAttr>(srcs[0]).getValue().str();
      ConduitInfo *coreRelaySrcPtr = state.lookupConduit(coreRelayName);
      if (!coreRelaySrcPtr) {
        linkOp.emitError("conduit-to-dma: CoreTile relay: src conduit '" +
                         coreRelayName + "' not found");
        state.passFailed = true;
        continue;
      }
      ConduitInfo &coreRelaySrc = *coreRelaySrcPtr;

      mlir::Value relayTileVal = memtile.getResult();

      // Retrieve relay buffers and locks from srcInfo.
      auto relayBufIt = coreRelaySrc.consumerTileBuffers.find(relayTileVal);
      if (relayBufIt == coreRelaySrc.consumerTileBuffers.end() ||
          relayBufIt->second.empty()) {
        linkOp.emitError("conduit-to-dma: CoreTile relay: relay buffers for "
                         "src conduit '" + coreRelayName +
                         "' not found on relay tile '" + memtileStr.str() +
                         "'");
        state.passFailed = true;
        continue;
      }
      auto &relayBufs = relayBufIt->second;

      AIE::LockOp relayProdLock, relayConsLock;
      {
        auto lockIt = coreRelaySrc.consumerTileLocks.find(relayTileVal);
        if (lockIt != coreRelaySrc.consumerTileLocks.end()) {
          relayProdLock = lockIt->second.first;
          relayConsLock = lockIt->second.second;
        }
      }

      int64_t relayDepth =
          coreRelaySrc.depth > 0 ? coreRelaySrc.depth : 1;

      // If relay locks are missing (e.g., shim→relay path where Phase 3 doesn't
      // allocate consumer-side locks on the relay tile), allocate them now.
      if (!relayProdLock && !coreRelaySrc.disableSynchronization) {
        builder.setInsertionPoint(state.deviceBody->getTerminator());
        {
          int lockIdx = state.lockIdCounter[relayTileVal]++;
          std::string symName = coreRelayName + "_cons_prod_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              loc, relayTileVal, lockIdx, static_cast<int>(relayDepth));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          relayProdLock = lk;
          coreRelaySrc.consumerTileLocks[relayTileVal].first = lk;
        }
        {
          int lockIdx = state.lockIdCounter[relayTileVal]++;
          std::string symName = coreRelayName + "_cons_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              loc, relayTileVal, lockIdx, 0);
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          relayConsLock = lk;
          coreRelaySrc.consumerTileLocks[relayTileVal].second = lk;
        }
      }

      int64_t relayPerBufLen =
          coreRelaySrc.capacity > 0 ? coreRelaySrc.capacity / relayDepth : 1;

      // Retrieve the S2MM channel pre-assigned by Phase 4a.
      int32_t relaySrcS2MMCh = -1;
      {
        auto chIt = state.conduitConsS2MMChannel.find({coreRelayName, 0u});
        if (chIt != state.conduitConsS2MMChannel.end())
          relaySrcS2MMCh = chIt->second;
        else
          relaySrcS2MMCh = state.tileNextS2MMChannel[relayTileVal]++;
      }

      // Always use DMA relay for CoreTile relay links.
      // Phase 3c now skips linkDstNames conduits, so dst conduits are never
      // marked sharedMemory here; we always generate S2MM + MM2S on the relay.
      const bool allDstSharedMem = false;

      // Helper: create or find the aie.mem region for the relay tile.
      // We reuse the pre-computed tileToDMARegion map built at Phase 5.5 entry,
      // but that hasn't been built yet (Phase 5 runs before 5.5). Walk ops now.
      auto findOrCreateRelayMem = [&]() -> mlir::Region * {
        // Check if an aie.mem already exists for this tile.
        mlir::Region *existing = nullptr;
        state.deviceOp.walk([&](AIE::MemOp m) {
          if (m.getTile() == relayTileVal)
            existing = &m.getBody();
        });
        if (existing)
          return existing;
        builder.setInsertionPoint(state.deviceBody->getTerminator());
        auto memOp =
            builder.create<AIE::MemOp>(loc, relayTileVal);
        return &memOp.getBody();
      };

      // Emit S2MM BD chain on relay tile (receives from upstream).
      // Then, if not all-shared-mem, append MM2S chain using same relay bufs.
      {
        mlir::Region *relayRegion = findOrCreateRelayMem();
        auto addRelayBlock = [&]() -> mlir::Block * {
          return builder.createBlock(relayRegion);
        };

        // Find or create end block.
        mlir::Block *existEnd = nullptr;
        for (mlir::Block &blk : *relayRegion)
          for (mlir::Operation &op : blk)
            if (mlir::isa<AIE::EndOp>(op))
              existEnd = &blk;

        // Build S2MM chain.
        mlir::Block *s2mmEntry = existEnd ? existEnd : addRelayBlock();
        if (existEnd)
          existEnd->back().erase(); // remove old aie.end; this block continues

        llvm::SmallVector<mlir::Block *> s2mmBDs;
        for (int64_t i = 0; i < relayDepth; ++i)
          s2mmBDs.push_back(addRelayBlock());

        // S2MM exits to MM2S start (or to end if shared-mem only).
        mlir::Block *mm2sStart = nullptr;
        mlir::Block *endBlock = nullptr;
        if (!allDstSharedMem) {
          mm2sStart = addRelayBlock();
        } else {
          endBlock = addRelayBlock();
        }
        mlir::Block *s2mmExit = allDstSharedMem ? endBlock : mm2sStart;

        builder.setInsertionPointToEnd(s2mmEntry);
        builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::S2MM,
                                        relaySrcS2MMCh, 0,
                                        s2mmBDs[0], s2mmExit);

        for (int64_t i = 0; i < relayDepth; ++i) {
          mlir::Value acqLock =
              relayProdLock ? relayProdLock.getResult() : mlir::Value{};
          mlir::Value relLock =
              relayConsLock ? relayConsLock.getResult() : mlir::Value{};
          state.emitBDBlock(loc, s2mmBDs[i],
                            acqLock, state.lockAcqValue(Port::Produce, 1),
                            relayBufs[i % relayBufs.size()].getResult(),
                            0, relayPerBufLen,
                            relLock, state.lockRelValue(Port::Produce));
          builder.create<AIE::NextBDOp>(loc, s2mmBDs[(i + 1) % relayDepth]);
        }

        if (allDstSharedMem) {
          // Shared-memory-only case: just add aie.end after S2MM.
          builder.setInsertionPointToEnd(endBlock);
          builder.create<AIE::EndOp>(loc);
        } else {
          // DMA relay case: add MM2S chain for each dst.
          // Use one MM2S channel per dst conduit.
          mlir::Block *prevMM2SBlock = mm2sStart;
          for (unsigned dstIdx = 0; dstIdx < static_cast<unsigned>(dsts.size());
               ++dstIdx) {
            std::string dstName =
                mlir::cast<mlir::FlatSymbolRefAttr>(dsts[dstIdx]).getValue().str();
            ConduitInfo *dstInfo = state.lookupConduit(dstName);
            if (!dstInfo || dstInfo->consumerTileCoords.empty()) {
              continue;
            }

            // Allocate MM2S channel on relay tile.
            int32_t mm2sCh = state.tileNextMM2SChannel[relayTileVal]++;

            // Emit flows and record S2MM channels on consumer tiles.
            for (unsigned consIdx = 0;
                 consIdx < dstInfo->consumerTileCoords.size(); ++consIdx) {
              auto [consCol, consRow] = dstInfo->consumerTileCoords[consIdx];
              AIE::TileOp consTile =
                  state.lookupTileByCoord(consCol, consRow);
              if (!consTile) continue;
              mlir::Value consTileVal = consTile.getResult();
              int32_t s2mmCh = state.tileNextS2MMChannel[consTileVal]++;
              state.conduitConsS2MMChannel[{dstName, consIdx}] = s2mmCh;
              builder.setInsertionPoint(state.deviceBody->getTerminator());
              builder.create<AIE::FlowOp>(loc, relayTileVal,
                                          AIE::WireBundle::DMA, mm2sCh,
                                          consTileVal,
                                          AIE::WireBundle::DMA, s2mmCh);
            }

            // Build MM2S BD chain for this dst.
            llvm::SmallVector<mlir::Block *> mm2sBDs;
            for (int64_t i = 0; i < relayDepth; ++i)
              mm2sBDs.push_back(addRelayBlock());

            bool isLastDst = (dstIdx + 1 == static_cast<unsigned>(dsts.size()));
            mlir::Block *nextChainOrEnd =
                isLastDst ? addRelayBlock() : addRelayBlock();
            if (isLastDst)
              endBlock = nextChainOrEnd;

            builder.setInsertionPointToEnd(prevMM2SBlock);
            builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::MM2S,
                                            mm2sCh, 0,
                                            mm2sBDs[0], nextChainOrEnd);

            for (int64_t i = 0; i < relayDepth; ++i) {
              // MM2S: acq consLock (data ready), send, rel prodLock (space free)
              mlir::Value acqLock =
                  relayConsLock ? relayConsLock.getResult() : mlir::Value{};
              mlir::Value relLock =
                  relayProdLock ? relayProdLock.getResult() : mlir::Value{};
              state.emitBDBlock(loc, mm2sBDs[i],
                                acqLock, state.lockAcqValue(Port::Consume, 1),
                                relayBufs[i % relayBufs.size()].getResult(),
                                0, relayPerBufLen,
                                relLock, state.lockRelValue(Port::Consume));
              builder.create<AIE::NextBDOp>(loc,
                                            mm2sBDs[(i + 1) % relayDepth]);
            }

            prevMM2SBlock = nextChainOrEnd;
          }
          if (endBlock) {
            builder.setInsertionPointToEnd(endBlock);
            builder.create<AIE::EndOp>(loc);
          }
        }
      }

      linkOpsToErase.push_back(linkOp.op);
      continue;
    }

    std::string srcName =
        mlir::cast<mlir::FlatSymbolRefAttr>(srcs[0]).getValue().str();
    ConduitInfo *srcInfoPtr = state.lookupConduit(srcName);

    // Guard: cascade-mode conduits cannot be used with distribute/join/forward.
    if (srcInfoPtr && srcInfoPtr->routingMode == "cascade") {
      linkOp.emitError("conduit distribute/join/forward cannot use cascade-mode "
                       "conduit '" + srcName + "' — cascade is point-to-point "
                       "and has no MemTile relay or DMA BD chain");
      state.passFailed = true;
      continue;
    }

    // For join links, the first source may be a MemTile relay whose buffers
    // were not allocated in Phase 3j (relay producers live on MemTiles,
    // not compute tiles).  The join path uses joinIntermediateBuffers, not
    // srcInfo.buffers, so empty buffers are safe for MemTile relay sources.
    bool isMemTileRelaySrc = false;
    if (srcInfoPtr) {
      auto [sp, sr] = srcInfoPtr->producerTileCoord;
      isMemTileRelaySrc =
          (sp >= 0 && sr >= 0 && targetModel.isMemTile(sp, sr));
    }
    if (!srcInfoPtr ||
        (srcInfoPtr->buffers.empty() && !isMemTileRelaySrc)) {
      linkOp.emitError("conduit-to-dma: src conduit '" + srcName +
                       "' buffers not allocated for link op");
      state.passFailed = true;
      continue;
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

    // Per-destination independent lock pairs on the MemTile (distribute/forward).
    bool isDistribute = linkOp.isDistribute;
    unsigned numDsts = static_cast<unsigned>(dsts.size());

    llvm::SmallVector<AIE::LockOp> sliceProdLocks;
    llvm::SmallVector<AIE::LockOp> sliceConsLocks;

    mlir::Value memtileVal = memtile.getResult();

    // Determine the correct insertion point for locks and buffers.
    // If a memtile_dma already exists for this memtile (from a previous link
    // group), we must insert locks/buffers BEFORE it so they dominate the
    // use_lock ops inside the merged memtile_dma region.
    mlir::Operation *lockInsertionPoint = state.deviceBody->getTerminator();
    {
      auto mtExisting = memtileDMAMap.find(memtileVal);
      if (mtExisting != memtileDMAMap.end())
        lockInsertionPoint = mtExisting->second.getOperation();
    }

    // Distribute MemTile per-destination lock pairs.
    // Skipped when the source conduit has disable_synchronization: the oracle
    // emits no locks and no use_lock for synchronization-disabled conduits on
    // the MemTile side. The BD chains are still emitted (lock values remain
    // null and emitBDBlock skips the use_lock emission).
    if (isDistribute && numDsts > 0 && !srcInfo.disableSynchronization) {
      builder.setInsertionPoint(lockInsertionPoint);
      for (unsigned sliceIdx = 0; sliceIdx < numDsts; ++sliceIdx) {
        // Scale per-slice lock init by the destination fifo's repeat_count.
        // The MemTile MM2S fires linkDepth×repeat times before releasing.
        int64_t dstRepeat = 1;
        {
          std::string dstNameR =
              mlir::cast<mlir::FlatSymbolRefAttr>(dsts[sliceIdx]).getValue().str();
          if (ConduitInfo *dstInfoR = state.lookupConduit(dstNameR))
            if (dstInfoR->bdChainRepeatCount > 1)
              dstRepeat = dstInfoR->bdChainRepeatCount;
        }
        int64_t sliceProdInit = linkDepth * dstRepeat;

        if (isAIE2) {
          {
            int lockIdx = state.lockIdCounter[memtileVal]++;
            std::string symName = srcName + "_link_prod_lock_" +
                                  std::to_string(sliceIdx);
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                state.deviceOp.getLoc(), memtileVal, lockIdx,
                static_cast<int>(sliceProdInit));
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
    //
    // The join destination conduit's buffers were NOT pre-allocated in
    // Phase 3 (linkDstNames skip). Phase 5 allocates them here directly
    // on the memtile, then records them into jDstInfo->buffers so that
    // later phases (Phase 5.5 skip, Phase 6/7 erase) see the correct resources.
    //
    // Per-source lock pairs (one pair per S2MM channel) are also allocated
    // here. Phase 3 no longer allocates a redundant single lock pair for
    // the join destination. This eliminates the former +2 buffer / +2 lock
    // resource surplus.
    llvm::SmallVector<AIE::BufferOp> joinIntermediateBuffers;
    llvm::SmallVector<AIE::LockOp> joinSrcProdLocks;
    llvm::SmallVector<AIE::LockOp> joinSrcConsLocks;
    int64_t joinDstPerBufForLen = 1;

    if (!isDistribute && !dsts.empty()) {
      std::string jDstName = mlir::cast<mlir::FlatSymbolRefAttr>(dsts[0]).getValue().str();
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

        builder.setInsertionPoint(lockInsertionPoint);
        unsigned numJoinSrcs = static_cast<unsigned>(srcs.size());

        // Allocate join intermediate buffers on the memtile (depth-many).
        // Record them in jDstInfo->buffers so Phase 6/7 can look them up.
        for (int64_t i = 0; i < jDstDepth; ++i) {
          std::string symName = jDstName + "_buff_" + std::to_string(i);
          auto buf = builder.create<AIE::BufferOp>(state.deviceOp.getLoc(), intBufTy, memtileVal,
              mlir::StringAttr::get(ctx, symName), mlir::IntegerAttr{},
              mlir::ElementsAttr{}, mlir::IntegerAttr{});
          joinIntermediateBuffers.push_back(buf);
        }
        // Register in jDstInfo so downstream phases see the correct buffers.
        jDstInfo->buffers = joinIntermediateBuffers;

        // Allocate per-source lock pairs on the memtile.
        // Each S2MM channel i acquires joinSrcProdLocks[i] and releases
        // joinSrcConsLocks[i]; the MM2S chain acquires cons and releases prod.
        // Skipped when the join destination has disable_synchronization: oracle
        // emits no locks on the MemTile for synchronization-disabled fifos.
        if (!jDstInfo->disableSynchronization) {
          for (unsigned srcIdx = 0; srcIdx < numJoinSrcs; ++srcIdx) {
            if (isAIE2) {
              { int lockIdx = state.lockIdCounter[memtileVal]++;
                std::string symName = jDstName + "_prod_lock_" + std::to_string(srcIdx);
                AIE::LockOp lk = builder.create<AIE::LockOp>(
                    state.deviceOp.getLoc(), memtileVal, lockIdx, static_cast<int>(jDstDepth));
                lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                joinSrcProdLocks.push_back(lk); }
              { int lockIdx = state.lockIdCounter[memtileVal]++;
                std::string symName = jDstName + "_cons_lock_" + std::to_string(srcIdx);
                AIE::LockOp lk = builder.create<AIE::LockOp>(
                    state.deviceOp.getLoc(), memtileVal, lockIdx, 0);
                lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                joinSrcConsLocks.push_back(lk); }
            } else {
              int lockIdx = state.lockIdCounter[memtileVal]++;
              std::string symName = jDstName + "_lock_" + std::to_string(srcIdx);
              AIE::LockOp lk = builder.create<AIE::LockOp>(
                  state.deviceOp.getLoc(), memtileVal, lockIdx, 0);
              lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
              joinSrcProdLocks.push_back(lk);
              joinSrcConsLocks.push_back(lk);
            }
          }
        }
      }
    }

    // -----------------------------------------------------------------------
    // Pre-compute DMA channel assignments for this link group.
    //
    // Phase 4 (routePhase) may have already assigned channels for the shim
    // endpoints of the link source/destination conduits.  We reuse those
    // channel numbers and allocate new ones from tileNextS2MMChannel /
    // tileNextMM2SChannel for the internal memtile↔compute flows.
    // This avoids channel conflicts when multiple link groups share a
    // memtile (e.g., A-distribute + C-join on the same memtile).
    // -----------------------------------------------------------------------

    // Distribute: S2MM ingest channel on the memtile (matches the flow that
    // delivers data to the memtile from shim or compute producer).
    int32_t ingestS2MMCh = -1;
    if (isDistribute) {
      // If Phase 4 assigned a S2MM channel for the link source's memtile
      // consumer, reuse it.  Otherwise allocate one now (compute producer).
      auto it = state.conduitConsS2MMChannel.find({srcName, 0u});
      if (it != state.conduitConsS2MMChannel.end()) {
        ingestS2MMCh = it->second;
      } else {
        ingestS2MMCh = state.tileNextS2MMChannel[memtileVal]++;
      }
    }

    // Distribute: per-destination MM2S channels on the memtile.
    llvm::SmallVector<int32_t> distMM2SChannels;
    if (isDistribute) {
      for (unsigned i = 0; i < numDsts; ++i)
        distMM2SChannels.push_back(
            state.tileNextMM2SChannel[memtileVal]++);
    }

    // Join: per-source S2MM channels on the memtile.
    // For MemTile relay sources, the upstream distribute link already
    // allocated an S2MM channel on this memtile — reuse it to match
    // the flow that delivers relay data.
    llvm::SmallVector<int32_t> joinS2MMChannels;
    if (!isDistribute) {
      for (unsigned i = 0; i < static_cast<unsigned>(srcs.size()); ++i) {
        std::string sName =
            mlir::cast<mlir::FlatSymbolRefAttr>(srcs[i]).getValue().str();
        ConduitInfo *sInfo = state.lookupConduit(sName);
        bool reused = false;
        if (sInfo) {
          auto [sp, sr] = sInfo->producerTileCoord;
          if (sp >= 0 && sr >= 0 && targetModel.isMemTile(sp, sr)) {
            auto chIt = state.conduitConsS2MMChannel.find({sName, 0u});
            if (chIt != state.conduitConsS2MMChannel.end()) {
              joinS2MMChannels.push_back(chIt->second);
              reused = true;
            }
          }
        }
        if (!reused) {
          int32_t ch = state.tileNextS2MMChannel[memtileVal]++;
          joinS2MMChannels.push_back(ch);
          // B-2 fix: record the newly allocated S2MM channel in
          // conduitConsS2MMChannel so that if this join source conduit also
          // appears as a broadcast consumer (or in a subsequent link group),
          // the lookup in Phase 4a/4.5a finds the same channel instead of
          // allocating a new one and mismatching the upstream flow.
          state.conduitConsS2MMChannel[{sName, 0u}] = ch;
        }
      }
    }

    // Join: MM2S output channel on the memtile for the join destination.
    // If Phase 4b assigned a channel (for memtile→shim egress), reuse it.
    // Otherwise allocate one now (for memtile→compute consumer).
    int32_t joinMM2SCh = -1;
    if (!isDistribute && !dsts.empty()) {
      std::string dstName0 =
          mlir::cast<mlir::FlatSymbolRefAttr>(dsts[0]).getValue().str();
      auto it = state.conduitMM2SChannel.find(dstName0);
      if (it != state.conduitMM2SChannel.end()) {
        joinMM2SCh = it->second;
      } else {
        joinMM2SCh = state.tileNextMM2SChannel[memtileVal]++;
      }
    }

    // -----------------------------------------------------------------------
    // Emit aie.flow ops for distribute/join.
    // -----------------------------------------------------------------------
    if (isDistribute) {
      builder.setInsertionPoint(state.deviceBody->getTerminator());
      for (unsigned dstIdx = 0; dstIdx < numDsts; ++dstIdx) {
        std::string dstName =
            mlir::cast<mlir::FlatSymbolRefAttr>(dsts[dstIdx]).getValue().str();
        ConduitInfo *dstInfo = state.lookupConduit(dstName);
        if (!dstInfo || dstInfo->consumerTileCoords.empty())
          continue;

        // Emit one flow per consumer tile of this dst conduit.
        // For simple distribute (1 consumer per dst), this is one flow.
        // For broadcast distribute (N consumers per dst), this emits N flows
        // from the same MemTile MM2S channel to each consumer — matching the
        // stateful transform which emits one aie.flow per consumer tile.
        int32_t mm2sCh = dstIdx < distMM2SChannels.size()
                             ? distMM2SChannels[dstIdx] : 0;
        for (unsigned consIdx = 0;
             consIdx < dstInfo->consumerTileCoords.size(); ++consIdx) {
          auto [dstConsCol, dstConsRow] =
              dstInfo->consumerTileCoords[consIdx];
          AIE::TileOp dstConsTile =
              state.lookupTileByCoord(dstConsCol, dstConsRow);
          if (!dstConsTile)
            continue;

          // S2MM channel on the consumer tile: assigned independently per
          // consumer tile (broadcast consumers each use their own S2MM).
          mlir::Value consTileVal = dstConsTile.getResult();
          int32_t s2mmCh = state.tileNextS2MMChannel[consTileVal]++;
          state.conduitConsS2MMChannel[{dstName, consIdx}] = s2mmCh;

          builder.create<AIE::FlowOp>(
              state.deviceOp.getLoc(), memtileVal, AIE::WireBundle::DMA,
              mm2sCh, dstConsTile.getResult(),
              AIE::WireBundle::DMA, static_cast<int32_t>(s2mmCh));
        }
      }

      // Source→MemTile flow for compute-tile producers.
      // Stream conduits: use Core:N wire bundle on the producer side.
      // The producer core outputs data directly through its AXI stream port
      // (no DMA engine). The MemTile S2MM DMA receives the stream data.
      {
        auto [srcProdCol, srcProdRow] = srcInfo.producerTileCoord;
        if (srcProdCol >= 0 && srcProdRow >= 2) {
          AIE::TileOp srcProdTile =
              state.lookupTileByCoord(srcProdCol, srcProdRow);
          if (srcProdTile) {
            AIE::WireBundle srcBundle = AIE::WireBundle::DMA;
            int32_t srcPort = 0;
            if (srcInfo.routingMode == "stream") {
              srcBundle = AIE::WireBundle::Core;
              srcPort = srcInfo.aieStreamPort >= 0 ? srcInfo.aieStreamPort : 0;
            }
            builder.create<AIE::FlowOp>(
                state.deviceOp.getLoc(), srcProdTile.getResult(),
                srcBundle, srcPort,
                memtileVal, AIE::WireBundle::DMA,
                static_cast<int32_t>(ingestS2MMCh));
          }
        }
      }
    } else {
      // Join: per-source flows + destination flow.
      builder.setInsertionPoint(state.deviceBody->getTerminator());

      for (unsigned srcIdx = 0; srcIdx < srcs.size(); ++srcIdx) {
        std::string sName =
            mlir::cast<mlir::FlatSymbolRefAttr>(srcs[srcIdx]).getValue().str();
        ConduitInfo *sInfo = state.lookupConduit(sName);
        if (!sInfo)
          continue;
        auto [srcProdCol, srcProdRow] = sInfo->producerTileCoord;
        if (srcProdCol < 0 || srcProdRow == 0)
          continue;
        // Skip MemTile relay join sources: the upstream distribute link
        // already emitted the flow from the relay MemTile to this join
        // MemTile.  Emitting again would create a duplicate flow.
        if (targetModel.isMemTile(srcProdCol, srcProdRow))
          continue;
        AIE::TileOp srcProdTile = state.lookupTileByCoord(srcProdCol, srcProdRow);
        if (!srcProdTile)
          continue;
        int32_t s2mmCh = srcIdx < joinS2MMChannels.size()
                             ? joinS2MMChannels[srcIdx] : 0;
        builder.create<AIE::FlowOp>(
            state.deviceOp.getLoc(), srcProdTile.getResult(),
            AIE::WireBundle::DMA, static_cast<int32_t>(0),
            memtileVal, AIE::WireBundle::DMA, s2mmCh);
      }

      // Destination flow: memtile MM2S → dst compute consumer.
      // NOTE: shim consumer flows are handled by Phase 4b (routePhase) which
      // iterates all conduits with shimConsumerTileCoords. We must NOT emit
      // the shim flow here to avoid duplicating Phase 4b's emission.
      if (!dsts.empty()) {
        std::string dstName = mlir::cast<mlir::FlatSymbolRefAttr>(dsts[0]).getValue().str();
        if (ConduitInfo *dstFlowInfo = state.lookupConduit(dstName)) {
          for (unsigned ci = 0; ci < dstFlowInfo->consumerTileCoords.size(); ++ci) {
            auto [consCol, consRow] = dstFlowInfo->consumerTileCoords[ci];
            AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
            if (consTile) {
              int32_t consS2MM =
                  state.tileNextS2MMChannel[consTile.getResult()]++;
              // Record the allocated S2MM channel so downstream forward links
              // can reuse it instead of double-allocating (join dst → forward
              // src relay chain).  Without this, the forward link's source
              // lookup in conduitConsS2MMChannel fails, causing a new S2MM
              // channel to be allocated — leading to flow/DMAStartOp channel
              // mismatch and potential MemTile S2MM overflow (>6 channels).
              state.conduitConsS2MMChannel[{dstName, ci}] = consS2MM;
              builder.create<AIE::FlowOp>(state.deviceOp.getLoc(), memtileVal,
                  AIE::WireBundle::DMA, joinMM2SCh, consTile.getResult(),
                  AIE::WireBundle::DMA, consS2MM);
            }
          }
          // Shim consumer flows are NOT emitted here. Phase 4b emits
          // prodTile→shimTile for join destination conduits that have
          // shimConsumerTileCoords. Emitting here would produce duplicates.
        }
      }
    }

    // -----------------------------------------------------------------------
    // Create or reuse memtile_dma DMA block.
    //
    // When multiple link groups share the same memtile, we merge their
    // DMA chains into a single aie.memtile_dma op.  The previous group's
    // aie.end terminator is replaced with a DMAStartOp that chains to
    // the new group's channels.
    // -----------------------------------------------------------------------
    mlir::Region *dmaRegionPtr = nullptr;
    mlir::Block *mergeChainBlock = nullptr;

    auto mtIt = memtileDMAMap.find(memtileVal);
    if (mtIt != memtileDMAMap.end()) {
      // Reuse existing MemTileDMAOp — find and remove the aie.end block.
      dmaRegionPtr = &mtIt->second.getBody();
      for (mlir::Block &block : *dmaRegionPtr) {
        if (auto *term = block.getTerminator()) {
          if (mlir::isa<AIE::EndOp>(term)) {
            mergeChainBlock = &block;
            term->erase();
            break;
          }
        }
      }
    } else {
      builder.setInsertionPoint(state.deviceBody->getTerminator());
      auto memtileDMA =
          builder.create<AIE::MemTileDMAOp>(loc, memtileVal);
      dmaRegionPtr = &memtileDMA.getBody();
      memtileDMAMap[memtileVal] = memtileDMA;
    }

    mlir::Region &dmaRegion = *dmaRegionPtr;

    auto addBlock = [&]() -> mlir::Block * {
      return builder.createBlock(&dmaRegion);
    };

    // Build S2MM (ingest) path.
    mlir::Block *mm2sChainStartBlock = nullptr;

    if (isDistribute) {
      // Single S2MM entry, depth*numDsts BD ring.
      // If merging, chain from the previous group's EndOp block.
      mlir::Block *entryBlock = mergeChainBlock ? mergeChainBlock
                                                : addBlock();

      llvm::SmallVector<mlir::Block *> ingestBlocks;
      for (int64_t bufIdx = 0; bufIdx < linkDepth; ++bufIdx)
        for (unsigned sliceIdx = 0; sliceIdx < numDsts; ++sliceIdx)
          ingestBlocks.push_back(addBlock());

      mm2sChainStartBlock = addBlock();

      builder.setInsertionPointToEnd(entryBlock);
      builder.create<AIE::DMAStartOp>(
          loc, AIE::DMAChannelDir::S2MM,
          ingestS2MMCh, static_cast<int32_t>(0),
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

        mlir::Value ingestAcqLock =
            (sliceIdx < sliceProdLocks.size())
                ? sliceProdLocks[sliceIdx].getResult()
                : mlir::Value{};
        mlir::Value ingestRelLock =
            (sliceIdx < sliceConsLocks.size())
                ? sliceConsLocks[sliceIdx].getResult()
                : mlir::Value{};
        state.emitBDBlock(
            loc, ingestBlocks[blkIdx],
            ingestAcqLock,
            state.lockAcqValue(Port::Produce, 1),
            linkBufs[bufIdx].getResult(), dstOffset, dstLen,
            ingestRelLock,
            state.lockRelValue(Port::Produce));
        builder.create<AIE::NextBDOp>(loc,
            ingestBlocks[(blkIdx + 1) % totalIngest]);
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
      // If merging, the first entry reuses the previous EndOp block.
      if (mergeChainBlock) {
        s2mmEntries[0] = mergeChainBlock;
        for (int64_t i = 0; i < jDepth; ++i)
          srcIngestBlocks[0].push_back(addBlock());
      }
      for (unsigned srcIdx = (mergeChainBlock ? 1u : 0u); srcIdx < numSrcs; ++srcIdx) {
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

        int32_t s2mmCh = srcIdx < joinS2MMChannels.size()
                             ? joinS2MMChannels[srcIdx] : 0;
        builder.setInsertionPointToEnd(s2mmEntries[srcIdx]);
        builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::S2MM,
            s2mmCh, 0,
            srcIngestBlocks[srcIdx][0], nextBlock);

        mlir::Value jAcqLock = (srcIdx < joinSrcProdLocks.size())
            ? joinSrcProdLocks[srcIdx].getResult() : mlir::Value{};
        mlir::Value jRelLock = (srcIdx < joinSrcConsLocks.size())
            ? joinSrcConsLocks[srcIdx].getResult() : mlir::Value{};
        for (int64_t i = 0; i < jDepth; ++i) {
          mlir::Value buf = !joinIntermediateBuffers.empty()
              ? joinIntermediateBuffers[i % joinIntermediateBuffers.size()].getResult()
              : mlir::Value{};
          state.emitBDBlock(loc, srcIngestBlocks[srcIdx][i],
              jAcqLock, state.lockAcqValue(Port::Produce, 1),
              buf, srcOffset, srcLen,
              jRelLock, state.lockRelValue(Port::Produce));
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
          joinMM2SCh, 0, sendBDBlocks[0], endBlock);

      for (unsigned bdIdx = 0; bdIdx < totalBDs; ++bdIdx) {
        unsigned bufIdx = bdIdx / numJoinSrcs;
        unsigned srcIdx = bdIdx % numJoinSrcs;
        mlir::Value acqLock = (srcIdx < joinSrcConsLocks.size())
            ? joinSrcConsLocks[srcIdx].getResult() : mlir::Value{};
        mlir::Value relLock = (srcIdx < joinSrcProdLocks.size())
            ? joinSrcProdLocks[srcIdx].getResult() : mlir::Value{};
        state.emitBDBlock(loc, sendBDBlocks[bdIdx],
            acqLock, state.lockAcqValue(Port::Consume, 1),
            joinIntermediateBuffers[bufIdx % joinIntermediateBuffers.size()].getResult(),
            mm2sOffsets[srcIdx], mm2sLens[srcIdx],
            relLock, state.lockRelValue(Port::Consume));
        builder.create<AIE::NextBDOp>(loc, sendBDBlocks[(bdIdx + 1) % totalBDs]);
      }
    } else {
      // Distribute MM2S: per-destination chains.
      // producerDimensions for each destination come from the dst conduit's
      // producerDimensions (from that dst fifo's dimensionsToStream).
      // The MemTile MM2S BD carries the transform that the MemTile applies
      // as it sends data toward the compute tile.
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

        // Look up dst fifo's producerDimensions and repeat_count.
        AIE::BDDimLayoutArrayAttr dstProdDims;
        int64_t mm2sDstRepeat = 1;
        {
          std::string dstName2 =
              mlir::cast<mlir::FlatSymbolRefAttr>(dsts[dstIdx]).getValue().str();
          if (ConduitInfo *dstInfo = state.lookupConduit(dstName2)) {
            dstProdDims = dstInfo->producerDimensions;
            if (dstInfo->bdChainRepeatCount > 1)
              mm2sDstRepeat = dstInfo->bdChainRepeatCount;
          }
        }
        // Unroll by repeat_count: each source buffer is sent repeat times.
        int64_t thisDstEffective = thisDstDepth * mm2sDstRepeat;

        llvm::SmallVector<mlir::Block *> sendBDBlocks;
        for (int64_t i = 0; i < thisDstEffective; ++i)
          sendBDBlocks.push_back(addBlock());

        mlir::Block *nextChainBlock;
        if (dstIdx + 1 == dsts.size()) {
          endBlock = addBlock();
          nextChainBlock = endBlock;
        } else {
          nextChainBlock = addBlock();
        }

        int32_t mm2sCh = dstIdx < distMM2SChannels.size()
                             ? distMM2SChannels[dstIdx] : 0;
        builder.setInsertionPointToEnd(prevChainBlock);
        builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::MM2S,
            mm2sCh, 0, sendBDBlocks[0], nextChainBlock);

        for (int64_t i = 0; i < thisDstEffective; ++i) {
          // Each source buffer (linkBufs[j]) is repeated mm2sDstRepeat times.
          mlir::Value buf = !linkBufs.empty()
              ? linkBufs[(i / mm2sDstRepeat) % linkBufs.size()].getResult()
              : mlir::Value{};
          state.emitBDBlock(loc, sendBDBlocks[i],
              mm2sAcqLock ? mm2sAcqLock.getResult() : mlir::Value{},
              state.lockAcqValue(Port::Consume, 1),
              buf, dstOffset, dstLen,
              mm2sRelLock ? mm2sRelLock.getResult() : mlir::Value{},
              state.lockRelValue(Port::Consume), dstProdDims);
          builder.create<AIE::NextBDOp>(loc, sendBDBlocks[(i + 1) % thisDstEffective]);
        }
        prevChainBlock = nextChainBlock;
      }
    }

    if (!endBlock)
      endBlock = addBlock();

    builder.setInsertionPointToEnd(endBlock);
    builder.create<AIE::EndOp>(loc);

    linkOpsToErase.push_back(linkOp.op);
  } // end for (auto &linkOp : linkAdapters)

  // Erase distribute/join/forward ops after processing (collect-then-erase).
  for (auto *op : llvm::reverse(linkOpsToErase))
    op->erase();

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

  // -----------------------------------------------------------------------
  // Phase 5.5e: Build aie.shim_dma BD chains for external-buffer conduits.
  //
  // When conduit.register_external_buffers was present, the shim DMA must
  // use the registered external buffer(s) in its BD chain instead of
  // allocated tile-memory buffers.  Phase 4a already emitted the shim
  // locks (shimProdLock / shimConsLock) and shim_dma_allocation.
  // -----------------------------------------------------------------------
  for (auto &[name, info] : state.conduitMap) {
    if (info.externalBuffers.empty())
      continue;
    auto [prodCol, prodRow] = info.producerTileCoord;
    if (prodCol < 0 || prodRow != 0)
      continue; // only shim producers

    AIE::TileOp shimTile = state.lookupTileByCoord(prodCol, prodRow);
    if (!shimTile)
      continue;

    mlir::Value shimTileVal = shimTile.getResult();
    builder.setInsertionPoint(state.deviceBody->getTerminator());

    // Build aie.shim_dma with one BD per external buffer.
    auto shimDMAOp =
        builder.create<AIE::ShimDMAOp>(state.deviceOp.getLoc(), shimTileVal);
    mlir::Region &shimRegion = shimDMAOp.getBody();
    auto addShimBlock = [&]() -> mlir::Block * {
      return builder.createBlock(&shimRegion);
    };

    mlir::Block *entryBlock = addShimBlock();
    unsigned numExtBufs = info.externalBuffers.size();
    llvm::SmallVector<mlir::Block *> bdBlocks;
    for (unsigned i = 0; i < numExtBufs; ++i)
      bdBlocks.push_back(addShimBlock());
    mlir::Block *endBlock = addShimBlock();

    builder.setInsertionPointToEnd(entryBlock);
    builder.create<AIE::DMAStartOp>(
        state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
        static_cast<int32_t>(0), static_cast<int32_t>(0),
        bdBlocks[0], endBlock);

    for (unsigned i = 0; i < numExtBufs; ++i) {
      mlir::Value extBuf = info.externalBuffers[i];
      // Determine length from the external buffer memref type.
      int64_t bufLen = 1;
      auto mref = mlir::dyn_cast<mlir::MemRefType>(extBuf.getType());
      if (mref && !mref.getShape().empty()) {
        bufLen = 1;
        for (int64_t d : mref.getShape())
          if (!mlir::ShapedType::isDynamic(d))
            bufLen *= d;
      }

      mlir::Value acqLock =
          info.shimProdLock ? info.shimProdLock.getResult() : mlir::Value{};
      mlir::Value relLock =
          info.shimConsLock ? info.shimConsLock.getResult() : mlir::Value{};

      state.emitBDBlock(
          state.deviceOp.getLoc(), bdBlocks[i],
          acqLock, state.lockAcqValue(Port::Consume, 1),
          extBuf, 0, bufLen,
          relLock, state.lockRelValue(Port::Consume));
      // Circular ring.
      builder.create<AIE::NextBDOp>(
          state.deviceOp.getLoc(), bdBlocks[(i + 1) % numExtBufs]);
    }

    builder.setInsertionPointToEnd(endBlock);
    builder.create<AIE::EndOp>(state.deviceOp.getLoc());
  }

  for (auto &[name, info] : state.conduitMap) {
    // For disable_synchronization conduits, locks are null by design — skip the
    // lock check. Still skip if buffers are empty (no allocation happened).
    if (info.buffers.empty())
      continue;
    if (!info.disableSynchronization && (!info.prodLock || !info.consLock))
      continue;

    // Handle link source conduits: emit aie.mem MM2S on producer compute tile.
    // Stream conduits: skip entirely — the producer uses a Core stream port,
    // not a DMA engine. No aie.mem or BD chain on the producer tile.
    // Distribute sources (linkSrcNamesEarly) are unreachable here: Phase 3
    // skips top-level prodLock/consLock for distribute sources (they use
    // per-tile consumerTileLocks instead), so line 913 guards above fires and
    // the loop body is never reached for distribute link sources.
    if (state.linkSrcNames.count(name)) {
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
      int64_t perBufLen = info.numElems > 0
                             ? info.numElems
                             : (info.capacity > 0 ? info.capacity / depth : 1);
      mlir::Value prodTileVal = prodTile.getResult();

      // Check for an existing aie.mem for this tile (e.g. created by Phase 5.5
      // for a broadcast consumer S2MM on the same tile).  If one exists, append
      // the MM2S chain into it rather than creating a duplicate aie.mem op.
      // Without this check, a tile that is both a broadcast consumer and a join
      // source receives two separate aie.mem blocks, and aiecc silently discards
      // the second one, causing a hardware deadlock.
      mlir::Region *existingRegion = nullptr;
      {
        auto it = tileToDMARegion.find(prodTileVal);
        if (it != tileToDMARegion.end())
          existingRegion = it->second;
      }

      // Acquire the MM2S channel index (channel 0 unless pre-used).
      int32_t joinMM2SChannel = 0;
      {
        auto &usedCh = state.preUsedMM2SChannels[prodTileVal];
        while (usedCh.count(joinMM2SChannel))
          ++joinMM2SChannel;
        usedCh.insert(joinMM2SChannel);
      }

      mlir::Region *memRegion = nullptr;
      if (existingRegion) {
        // Append MM2S into existing aie.mem: find the aie.end block and insert
        // new DMA start + BD blocks before it.
        memRegion = existingRegion;
        mlir::Block *endBlock = nullptr;
        for (mlir::Block &blk : *memRegion)
          for (mlir::Operation &op : blk)
            if (mlir::isa<AIE::EndOp>(op))
              endBlock = &blk;

        if (!endBlock) {
          state.deviceOp.emitError(
              "conduit-to-dma: join-source append: existing aie.mem has no "
              "aie.end block — region is malformed");
          state.passFailed = true;
          return;
        }
        {
          int64_t nBufs = info.nConsumerBuffers();
          auto addBlock = [&]() -> mlir::Block * {
            return builder.createBlock(memRegion);
          };
          llvm::SmallVector<mlir::Block *> bdBlocks;
          for (int64_t i = 0; i < nBufs; ++i)
            bdBlocks.push_back(addBlock());
          mlir::Block *newEndBlock = addBlock();

          // Remove aie.end from old end block; it becomes a fallthrough.
          endBlock->back().erase();

          builder.setInsertionPointToEnd(endBlock);
          builder.create<AIE::DMAStartOp>(
              state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
              joinMM2SChannel, static_cast<int32_t>(0),
              bdBlocks[0], newEndBlock);

          for (int64_t i = 0; i < nBufs; ++i) {
            mlir::Value acqLock = isAIE2 ? info.consLock.getResult()
                : (info.aie1Locks.empty() ? info.consLock.getResult()
                       : info.aie1Locks[i % info.aie1Locks.size()].getResult());
            mlir::Value relLock = isAIE2 ? info.prodLock.getResult() : acqLock;
            state.emitBDBlock(
                state.deviceOp.getLoc(), bdBlocks[i],
                acqLock, state.lockAcqValue(Port::Consume, 1),
                info.buffers[i % info.buffers.size()].getResult(), 0, perBufLen,
                relLock, state.lockRelValue(Port::Consume));
            builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                          bdBlocks[(i + 1) % nBufs]);
          }
          builder.setInsertionPointToEnd(newEndBlock);
          builder.create<AIE::EndOp>(state.deviceOp.getLoc());
        }
      } else {
        // No existing aie.mem for this tile — create one and register it.
        builder.setInsertionPoint(state.deviceBody->getTerminator());
        auto memOp = builder.create<AIE::MemOp>(
            state.deviceOp.getLoc(), prodTileVal);
        memRegion = &memOp.getBody();
        tileToDMARegion[prodTileVal] = memRegion;   // register for later phases

        int64_t nBufs = info.nConsumerBuffers();
        auto addMemBlock = [&]() -> mlir::Block * {
          return builder.createBlock(memRegion);
        };
        mlir::Block *dmaStartBlock = addMemBlock();
        llvm::SmallVector<mlir::Block *> bdBlocks;
        for (int64_t i = 0; i < nBufs; ++i)
          bdBlocks.push_back(addMemBlock());
        mlir::Block *endMemBlock = addMemBlock();

        builder.setInsertionPointToEnd(dmaStartBlock);
        builder.create<AIE::DMAStartOp>(state.deviceOp.getLoc(),
                                        AIE::DMAChannelDir::MM2S,
                                        joinMM2SChannel,
                                        static_cast<int32_t>(0),
                                        bdBlocks[0], endMemBlock);

        for (int64_t i = 0; i < nBufs; ++i) {
          mlir::Value acqLock = isAIE2 ? info.consLock.getResult()
              : (info.aie1Locks.empty() ? info.consLock.getResult()
                     : info.aie1Locks[i % info.aie1Locks.size()].getResult());
          mlir::Value relLock = isAIE2 ? info.prodLock.getResult() : acqLock;
          state.emitBDBlock(
              state.deviceOp.getLoc(), bdBlocks[i],
              acqLock, state.lockAcqValue(Port::Consume, 1),
              info.buffers[i % info.buffers.size()].getResult(), 0, perBufLen,
              relLock, state.lockRelValue(Port::Consume));
          builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                        bdBlocks[(i + 1) % nBufs]);
        }
        builder.setInsertionPointToEnd(endMemBlock);
        builder.create<AIE::EndOp>(state.deviceOp.getLoc());
      }
      continue;
    }

    // Skip shared memory conduits.
    if (info.sharedMemory)
      continue;

    // Case C: non-adjacent producer MM2S on producer tile.
    // Skip link destinations: their MemTile MM2S BDs are emitted in Phase 5.
    {
      auto [prodCol, prodRow] = info.producerTileCoord;
      if (prodCol >= 0 && prodRow >= 1 &&
          !state.linkSrcNames.count(name) &&
          !state.linkDstNames.count(name) &&
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

            // For disable_synchronization conduits, locks are null by design
            // but we still emit BD chains (without lock ops via emitBDBlock).
            if (mm2sAcqLock || info.disableSynchronization) {
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

              // Compute DMAStartOp repeat_count from iter_count.
              int32_t dmaRepeatCount = (info.iterCount > 0) ?
                  static_cast<int32_t>(info.iterCount - 1) : 0;
              // BD chain repeat factor for objectfifo repeat_count.
              int64_t bdRepeat = info.bdChainRepeatCount > 1 ?
                  info.bdChainRepeatCount : 1;
              int64_t effectiveBDs = depth * bdRepeat;

              if (existingDMARegion) {
                mlir::Region &memRegion = *existingDMARegion;
                mlir::Block *endBlock = nullptr;
                for (mlir::Block &block : memRegion)
                  for (mlir::Operation &opInBlock : block)
                    if (mlir::isa<AIE::EndOp>(opInBlock))
                      endBlock = &block;

                if (!endBlock) {
                  state.deviceOp.emitError(
                      "conduit-to-dma: Case C append: existing DMA region "
                      "has no aie.end block — region is malformed");
                  state.passFailed = true;
                  return;
                }
                {
                  auto addBlock = [&]() -> mlir::Block * {
                    return builder.createBlock(&memRegion);
                  };
                  llvm::SmallVector<mlir::Block *> bdBlocks;
                  for (int64_t i = 0; i < effectiveBDs; ++i)
                    bdBlocks.push_back(addBlock());

                  // For finite chains (iter_count > 0), create a dedicated BD
                  // terminal block BEFORE newEndBlock.  The scan for the next
                  // channel's "endBlock" iterates blocks in insertion order and
                  // returns the LAST block with aie.end; since bdTermBlock is
                  // inserted first, newEndBlock (last) is selected by the next
                  // channel and its aie.end is replaced.  bdTermBlock keeps its
                  // aie.end permanently, satisfying the AIEAssignBufferDescriptorIDs
                  // assertion: "bb that's not in blockMap can only have aie.end".
                  mlir::Block *bdTermBlock =
                      (info.iterCount > 0) ? addBlock() : nullptr;
                  mlir::Block *newEndBlock = addBlock();

                  if (isFusedNonFirst) {
                    // Non-first fused member: no new dma_start.
                  } else {
                    mlir::Operation *oldEnd = endBlock->getTerminator();
                    builder.setInsertionPointToEnd(endBlock);
                    oldEnd->erase();
                    builder.create<AIE::DMAStartOp>(
                        state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
                        mm2sChannel, dmaRepeatCount,
                        bdBlocks[0], newEndBlock);
                  }

                  for (int64_t i = 0; i < effectiveBDs; ++i) {
                    // Null locks for disable_synchronization — emitBDBlock skips them.
                    mlir::Value blockAcq =
                        mm2sAcqLock
                            ? (isAIE2 ? mm2sAcqLock.getResult()
                                      : (prodAIE1Locks && !prodAIE1Locks->empty()
                                             ? (*prodAIE1Locks)[i % prodAIE1Locks->size()].getResult()
                                             : mm2sAcqLock.getResult()))
                            : mlir::Value{};
                    mlir::Value blockRel =
                        mm2sRelLock
                            ? (isAIE2 ? mm2sRelLock.getResult() : blockAcq)
                            : mlir::Value{};
                    state.emitBDBlock(
                        state.deviceOp.getLoc(), bdBlocks[i],
                        blockAcq, state.lockAcqValue(Port::Consume, 1),
                        prodBuffers[(i / bdRepeat) % prodBuffers.size()].getResult(),
                        0, perBufLen,
                        blockRel, state.lockRelValue(Port::Consume),
                        info.producerDimensions);
                    // Non-circular when iter_count > 0: last BD → bdTermBlock (aie.end).
                    bool isLast = (i == effectiveBDs - 1) && (info.iterCount > 0);
                    if (isLast)
                      builder.create<AIE::NextBDOp>(
                          state.deviceOp.getLoc(), bdTermBlock);
                    else
                      builder.create<AIE::NextBDOp>(
                          state.deviceOp.getLoc(), bdBlocks[(i + 1) % effectiveBDs]);
                  }
                  if (bdTermBlock) {
                    builder.setInsertionPointToEnd(bdTermBlock);
                    builder.create<AIE::EndOp>(state.deviceOp.getLoc());
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
                for (int64_t i = 0; i < effectiveBDs; ++i)
                  bdBlocks.push_back(addBlock());
                mlir::Block *endBlock = addBlock();

                if (!isFusedNonFirst) {
                  builder.setInsertionPointToEnd(dmaStartBlock);
                  builder.create<AIE::DMAStartOp>(
                      state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
                      mm2sChannel, dmaRepeatCount,
                      bdBlocks[0], endBlock);
                }

                for (int64_t i = 0; i < effectiveBDs; ++i) {
                  // Null locks for disable_synchronization — emitBDBlock skips.
                  mlir::Value blockAcq =
                      mm2sAcqLock
                          ? (isAIE2 ? mm2sAcqLock.getResult()
                                    : (prodAIE1Locks && !prodAIE1Locks->empty()
                                           ? (*prodAIE1Locks)[i % prodAIE1Locks->size()].getResult()
                                           : mm2sAcqLock.getResult()))
                          : mlir::Value{};
                  mlir::Value blockRel =
                      mm2sRelLock
                          ? (isAIE2 ? mm2sRelLock.getResult() : blockAcq)
                          : mlir::Value{};
                  state.emitBDBlock(
                      state.deviceOp.getLoc(), bdBlocks[i],
                      blockAcq, state.lockAcqValue(Port::Consume, 1),
                      prodBuffers[(i / bdRepeat) % prodBuffers.size()].getResult(),
                      0, perBufLen,
                      blockRel, state.lockRelValue(Port::Consume),
                      info.producerDimensions);
                  // Non-circular when iter_count > 0.
                  bool isLast = (i == effectiveBDs - 1) && (info.iterCount > 0);
                  if (isLast)
                    builder.create<AIE::NextBDOp>(
                        state.deviceOp.getLoc(), endBlock);
                  else
                    builder.create<AIE::NextBDOp>(
                        state.deviceOp.getLoc(), bdBlocks[(i + 1) % effectiveBDs]);
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
      int64_t perBufLen = info.numElems > 0
                             ? info.numElems
                             : (info.capacity > 0 ? info.capacity / depth : 1);

      mlir::Value prodTileVal = dmaHostTile.getResult();

      // Check for an existing aie.mem / aie.memtile_dma region for this
      // tile (created earlier by Case A's S2MM path).  If one exists,
      // append the MM2S chain into it instead of creating a duplicate.
      mlir::Region *existingDMARegion = nullptr;
      {
        auto dmaIt = tileToDMARegion.find(prodTileVal);
        if (dmaIt != tileToDMARegion.end())
          existingDMARegion = dmaIt->second;
      }

      // Case B: compute tile sends MM2S to shim.
      // producer_dimensions are NOT applied here: for shim consumers, the
      // DMA descriptor on the shim side (runtime-programmed) carries dims.
      // The compute tile MM2S BD uses the raw buffer without transforms.
      int32_t caseBDmaRepeatCount = (info.iterCount > 0) ?
          static_cast<int32_t>(info.iterCount - 1) : 0;
      int64_t caseBBdRepeat = info.bdChainRepeatCount > 1 ?
          info.bdChainRepeatCount : 1;
      int64_t caseBEffectiveBDs = info.nConsumerBuffers() * caseBBdRepeat;

      if (existingDMARegion) {
        mlir::Region &memRegion = *existingDMARegion;
        mlir::Block *endBlock = nullptr;
        for (mlir::Block &block : memRegion)
          for (mlir::Operation &opInBlock : block)
            if (mlir::isa<AIE::EndOp>(opInBlock))
              endBlock = &block;

        if (!endBlock) {
          state.deviceOp.emitError(
              "conduit-to-dma: Case B append: existing aie.mem has no "
              "aie.end block — region is malformed");
          state.passFailed = true;
          return;
        }
        {
          auto addMemBlock = [&]() -> mlir::Block * {
            return builder.createBlock(&memRegion);
          };
          llvm::SmallVector<mlir::Block *> bdBlocks;
          for (int64_t i = 0; i < caseBEffectiveBDs; ++i)
            bdBlocks.push_back(addMemBlock());
          mlir::Block *newEndBlock = addMemBlock();

          mlir::Operation *oldEnd = endBlock->getTerminator();
          builder.setInsertionPointToEnd(endBlock);
          oldEnd->erase();
          int32_t caseBMM2SCh;
          {
            // Reuse MM2S channel allocated by routePhase for shim consumers,
            // if present. Otherwise allocate a new one.
            auto chIt = state.conduitMM2SChannel.find(name);
            if (chIt != state.conduitMM2SChannel.end()) {
              caseBMM2SCh = chIt->second;
            } else {
              caseBMM2SCh = state.tileNextMM2SChannel[prodTileVal]++;
              int32_t caseBMM2SLimit = static_cast<int32_t>(
                  targetModel.getNumSourceSwitchboxConnections(
                      prodCol, prodRow, AIE::WireBundle::DMA));
              if (caseBMM2SCh >= caseBMM2SLimit) {
                state.deviceOp.emitError(
                    "conduit-to-dma: MM2S channel limit exceeded on tile (")
                    << prodCol << "," << prodRow << "): needed channel "
                    << caseBMM2SCh << " but max is " << caseBMM2SLimit;
                state.passFailed = true;
                return;
              }
              state.conduitMM2SChannel[name] = caseBMM2SCh;
            }
          }
          builder.create<AIE::DMAStartOp>(
              state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
              caseBMM2SCh, caseBDmaRepeatCount,
              bdBlocks[0], newEndBlock);

          for (int64_t i = 0; i < caseBEffectiveBDs; ++i) {
            // Null locks for disable_synchronization — emitBDBlock skips them.
            mlir::Value blockAcqVal =
                info.consLock
                    ? (isAIE2 ? info.consLock.getResult()
                               : (info.aie1Locks.empty()
                                      ? info.consLock.getResult()
                                      : info.aie1Locks[i % info.aie1Locks.size()].getResult()))
                    : mlir::Value{};
            mlir::Value blockRelVal =
                info.prodLock
                    ? (isAIE2 ? info.prodLock.getResult()
                               : (info.aie1Locks.empty()
                                      ? info.prodLock.getResult()
                                      : info.aie1Locks[i % info.aie1Locks.size()].getResult()))
                    : mlir::Value{};
            // Case B: compute MM2S to shim — no BDDimLayout on this BD
            // (shim-side descriptor is runtime-programmed, not emitted here).
            state.emitBDBlock(
                state.deviceOp.getLoc(), bdBlocks[i],
                blockAcqVal, state.lockAcqValue(Port::Consume, 1),
                info.buffers[(i / caseBBdRepeat) % info.buffers.size()].getResult(),
                0, perBufLen,
                blockRelVal, state.lockRelValue(Port::Consume));
            bool caseBIsLast = (i == caseBEffectiveBDs - 1) && (info.iterCount > 0);
            if (caseBIsLast)
              builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(), newEndBlock);
            else
              builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                            bdBlocks[(i + 1) % caseBEffectiveBDs]);
          }
          builder.setInsertionPointToEnd(newEndBlock);
          builder.create<AIE::EndOp>(state.deviceOp.getLoc());
        }
      } else {
        builder.setInsertionPoint(state.deviceBody->getTerminator());
        auto memOp = builder.create<AIE::MemOp>(state.deviceOp.getLoc(),
                                                prodTileVal);
        mlir::Region &memRegion = memOp.getBody();
        tileToDMARegion[prodTileVal] = &memRegion;
        auto addMemBlock = [&]() -> mlir::Block * {
          return builder.createBlock(&memRegion);
        };
        mlir::Block *dmaStartBlock = addMemBlock();
        llvm::SmallVector<mlir::Block *> bdBlocks;
        for (int64_t i = 0; i < caseBEffectiveBDs; ++i)
          bdBlocks.push_back(addMemBlock());
        mlir::Block *endMemBlock = addMemBlock();
        builder.setInsertionPointToEnd(dmaStartBlock);
        int32_t caseBMM2SCh2;
        {
          // Reuse MM2S channel allocated by routePhase for shim consumers,
          // if present. Otherwise allocate a new one.
          auto chIt = state.conduitMM2SChannel.find(name);
          if (chIt != state.conduitMM2SChannel.end()) {
            caseBMM2SCh2 = chIt->second;
          } else {
            caseBMM2SCh2 = state.tileNextMM2SChannel[prodTileVal]++;
            int32_t caseBMM2SLimit2 = static_cast<int32_t>(
                targetModel.getNumSourceSwitchboxConnections(
                    prodCol, prodRow, AIE::WireBundle::DMA));
            if (caseBMM2SCh2 >= caseBMM2SLimit2) {
              state.deviceOp.emitError(
                  "conduit-to-dma: MM2S channel limit exceeded on tile (")
                  << prodCol << "," << prodRow << "): needed channel "
                  << caseBMM2SCh2 << " but max is " << caseBMM2SLimit2;
              state.passFailed = true;
              return;
            }
            state.conduitMM2SChannel[name] = caseBMM2SCh2;
          }
        }
        builder.create<AIE::DMAStartOp>(state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
                                        caseBMM2SCh2, caseBDmaRepeatCount,
                                        bdBlocks[0], endMemBlock);

        for (int64_t i = 0; i < caseBEffectiveBDs; ++i) {
          // Null locks for disable_synchronization — emitBDBlock skips them.
          mlir::Value blockAcqVal =
              info.consLock
                  ? (isAIE2 ? info.consLock.getResult()
                             : (info.aie1Locks.empty()
                                    ? info.consLock.getResult()
                                    : info.aie1Locks[i % info.aie1Locks.size()].getResult()))
                  : mlir::Value{};
          mlir::Value blockRelVal =
              info.prodLock
                  ? (isAIE2 ? info.prodLock.getResult()
                             : (info.aie1Locks.empty()
                                    ? info.prodLock.getResult()
                                    : info.aie1Locks[i % info.aie1Locks.size()].getResult()))
                  : mlir::Value{};
          // Case B: compute MM2S to shim — no BDDimLayout.
          state.emitBDBlock(
              state.deviceOp.getLoc(), bdBlocks[i],
              blockAcqVal, state.lockAcqValue(Port::Consume, 1),
              info.buffers[(i / caseBBdRepeat) % info.buffers.size()].getResult(),
              0, perBufLen,
              blockRelVal, state.lockRelValue(Port::Consume));
          bool caseBIsLast = (i == caseBEffectiveBDs - 1) && (info.iterCount > 0);
          if (caseBIsLast)
            builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(), endMemBlock);
          else
            builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                          bdBlocks[(i + 1) % caseBEffectiveBDs]);
        }
        builder.setInsertionPointToEnd(endMemBlock);
        builder.create<AIE::EndOp>(state.deviceOp.getLoc());
      }

    } else {
      // Case A: S2MM on consumer tile(s).
      if (info.consumerTileCoords.empty())
        continue;

      int64_t depth = info.depth > 0 ? info.depth : 1;
      int64_t perBufLen = info.numElems > 0
                             ? info.numElems
                             : (info.capacity > 0 ? info.capacity / depth : 1);
      // nConsumerBuffers() >= depth; extra slots support sliding-window patterns.
      int64_t nBufs = info.nConsumerBuffers();

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

        // For disable_synchronization, locks are null by design — still emit BDs.
        if ((!tileProdLock || !tileConsLock) &&
            !info.disableSynchronization)
          continue;
        if (tileBuffers->empty())
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

        // Compute DMAStartOp repeat_count from iter_count.
        int32_t dmaRepeatCount = (info.iterCount > 0) ?
            static_cast<int32_t>(info.iterCount - 1) : 0;

        llvm::SmallVector<mlir::Block *> bdBlocks;
        for (int64_t i = 0; i < nBufs; ++i)
          bdBlocks.push_back(addMemBlock());

        // bdTermBlock strategy: when adding a channel to an existing DMA region
        // (existingEndBlock != null), the "next-channel" scan finds the LAST
        // aie.end block and replaces it with a DMAStartOp.  For finite chains
        // (iter_count > 0), we need a dedicated bdTermBlock (placed BEFORE
        // endMemBlock) whose aie.end stays permanent so the scan correctly
        // targets only endMemBlock.  When creating a fresh DMA region (no
        // existing channels), no scan occurs, so the last BD can point directly
        // to endMemBlock — no extra terminal block needed.
        mlir::Block *bdTermBlock =
            (info.iterCount > 0 && existingEndBlock) ? addMemBlock() : nullptr;
        mlir::Block *endMemBlock = addMemBlock();

        if (existingEndBlock) {
          mlir::Operation *oldEnd = existingEndBlock->getTerminator();
          builder.setInsertionPointToEnd(existingEndBlock);
          oldEnd->erase();
          builder.create<AIE::DMAStartOp>(state.deviceOp.getLoc(), AIE::DMAChannelDir::S2MM,
                                          s2mmChannel, dmaRepeatCount,
                                          bdBlocks[0], endMemBlock);
        } else {
          mlir::Block *dmaStartBlock = addMemBlock();
          dmaStartBlock->moveBefore(&memRegion.front());
          builder.setInsertionPointToEnd(dmaStartBlock);
          builder.create<AIE::DMAStartOp>(state.deviceOp.getLoc(), AIE::DMAChannelDir::S2MM,
                                          s2mmChannel, dmaRepeatCount,
                                          bdBlocks[0], endMemBlock);
        }

        // Pick consumer BDDimLayout for this consumer index.
        AIE::BDDimLayoutArrayAttr consDims;
        if (!info.consumerDimensions.empty())
          consDims = info.consumerDimensions[consIdx % info.consumerDimensions.size()];

        for (int64_t i = 0; i < nBufs; ++i) {
          // Null locks for disable_synchronization — emitBDBlock skips them.
          mlir::Value blockLockAcq =
              tileProdLock
                  ? (isAIE2 ? tileProdLock.getResult()
                             : (tileAIE1Locks->empty()
                                    ? tileProdLock.getResult()
                                    : (*tileAIE1Locks)[i % tileAIE1Locks->size()].getResult()))
                  : mlir::Value{};
          mlir::Value blockLockRel =
              tileConsLock
                  ? (isAIE2 ? tileConsLock.getResult()
                             : (tileAIE1Locks->empty()
                                    ? tileConsLock.getResult()
                                    : (*tileAIE1Locks)[i % tileAIE1Locks->size()].getResult()))
                  : mlir::Value{};
          state.emitBDBlock(
              state.deviceOp.getLoc(), bdBlocks[i],
              blockLockAcq, state.lockAcqValue(Port::Produce, 1),
              (*tileBuffers)[i % tileBuffers->size()].getResult(), 0, perBufLen,
              blockLockRel, state.lockRelValue(Port::Produce), consDims);
          // Non-circular chain when iter_count > 0: last BD → bdTermBlock (if
          // existingEndBlock) or endMemBlock (fresh region, no extra block needed).
          bool isLast = (i == nBufs - 1) && (info.iterCount > 0);
          if (isLast)
            builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                          bdTermBlock ? bdTermBlock : endMemBlock);
          else
            builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                          bdBlocks[(i + 1) % nBufs]);
        }
        if (bdTermBlock) {
          builder.setInsertionPointToEnd(bdTermBlock);
          builder.create<AIE::EndOp>(state.deviceOp.getLoc());
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
