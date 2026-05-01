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
// Phase 5: Lower conduit.scatter/gather/transpose → MemTile DMA BD chain.
//   Scatter (1 src → N dsts): S2MM ingests full buffer, N MM2S channels.
//   Gather (N srcs → 1 dst): N S2MM channels, one MM2S output.
//   Transpose: static N:M redistribution via MemTile.
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

// Derive the per-buffer BD transfer length for a Pass C BD descriptor.
//
// MUST agree with the actual buffer the BD writes into (intBufTy). The legacy
// `numElems` value is the SHIM aggregated per-dispatch window from
// --dma-task-to-conduit, which can exceed the per-tile buffer when an inner
// core loop reuses the same FIFO slot multiple times. Using numElems for BD
// length causes overflow at the memtile / compute tile (see commits
// 7569b3e8e6 and follow-up — JOIN-overflow + Llama RMSNorm distribution).
//
// Single source of truth: prefer intBufTy.getNumElements(), fall back to
// numElems only when intBufTy is not a MemRefType from which we can read
// the element count. Default 1 if neither yields a usable value.
static int64_t deriveBdLength(mlir::Type intBufTy, int64_t numElemsFallback) {
  if (auto mref = mlir::dyn_cast_or_null<mlir::MemRefType>(intBufTy))
    return mref.getNumElements();
  if (numElemsFallback > 0)
    return numElemsFallback;
  return 1;
}

void linkPhase(ConduitToDMAState &state) {
  if (!state.deviceOp)
    return;

  mlir::OpBuilder &builder = *state.builder;
  mlir::MLIRContext *ctx = state.ctx;
  const bool isAIE2 = state.isAIE2Plus();
  const AIE::AIETargetModel &targetModel = *state.targetModel;
  (void)ctx; // suppress unused warning when not used in all paths

  // Collect ALL link source and destination names for Phase 5.5 skip logic.
  // Use device-qualified keys so they match conduitMap iteration names.
  state.module.walk([&](ScatterOp op) {
    state.linkSrcNames.insert(state.makeConduitKey(op.getSrc(), op));
    for (auto d : op.getDsts())
      state.linkDstNames.insert(state.makeConduitKey(
          mlir::cast<mlir::FlatSymbolRefAttr>(d).getValue(), op));
  });
  state.module.walk([&](GatherOp op) {
    for (auto s : op.getSrcs())
      state.linkSrcNames.insert(state.makeConduitKey(
          mlir::cast<mlir::FlatSymbolRefAttr>(s).getValue(), op));
    state.linkDstNames.insert(state.makeConduitKey(op.getDst(), op));
  });

  // -----------------------------------------------------------------------
  // Phase 5: Lower conduit.scatter / conduit.gather / conduit.transpose.
  //
  // LinkAdapter unifies the relay op types so the lowering body can be shared.
  // -----------------------------------------------------------------------

  struct LinkAdapter {
    mlir::Operation *op;
    mlir::ArrayAttr srcs;
    mlir::ArrayAttr dsts;
    mlir::Value memtile;
    bool isDistribute; // true for Scatter; false for Gather
    std::optional<llvm::ArrayRef<int64_t>> offsets;
    int deviceIndex = -1; // Multi-device: owning device index.

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

  state.module.walk([&](ScatterOp scatterOp) {
    LinkAdapter a;
    a.op = scatterOp.getOperation();
    a.srcs = builder.getArrayAttr({scatterOp.getSrcAttr()});
    a.dsts = scatterOp.getDsts();
    a.memtile = scatterOp.getMemtile();
    a.isDistribute = true; // scatter = 1→N distribute
    a.offsets = scatterOp.getOffsets();
    // Multi-device: determine owning device index.
    if (state.isMultiDevice()) {
      for (int i = 0; i < static_cast<int>(state.deviceOps.size()); ++i) {
        if (scatterOp->getParentOfType<AIE::DeviceOp>() == state.deviceOps[i]) {
          a.deviceIndex = i;
          break;
        }
      }
    }
    linkAdapters.push_back(a);
  });
  state.module.walk([&](GatherOp gatherOp) {
    LinkAdapter a;
    a.op = gatherOp.getOperation();
    a.srcs = gatherOp.getSrcs();
    a.dsts = builder.getArrayAttr({gatherOp.getDstAttr()});
    a.memtile = gatherOp.getMemtile();
    a.isDistribute = false; // gather = N→1 join
    a.offsets = gatherOp.getOffsets();
    // Multi-device: determine owning device index.
    if (state.isMultiDevice()) {
      for (int i = 0; i < static_cast<int>(state.deviceOps.size()); ++i) {
        if (gatherOp->getParentOfType<AIE::DeviceOp>() == state.deviceOps[i]) {
          a.deviceIndex = i;
          break;
        }
      }
    }
    linkAdapters.push_back(a);
  });

  // Map from memtile tile value → existing MemTileDMAOp for merging.
  // When multiple link groups reference the same memtile, we merge them
  // into a single MemTileDMAOp with non-overlapping DMA channel numbers.
  llvm::DenseMap<mlir::Value, AIE::MemTileDMAOp> memtileDMAMap;

  for (auto &linkOp : linkAdapters) {
    // Multi-device: ensure tile lookups target the correct device.
    if (state.isMultiDevice() && linkOp.deviceIndex >= 0)
      state.switchToDeviceIndex(linkOp.deviceIndex);

    builder.setInsertionPoint(state.deviceBody->getTerminator());
    mlir::Location loc = linkOp.getLoc();

    auto srcs = linkOp.srcs;
    auto dsts = linkOp.dsts;
    auto offsets = linkOp.offsets;

    // memtile is now an SSA operand of the relay op (Index type, defining op
    // is aie.tile).  By construction (Pass A uses TileOp::getOrCreate), this
    // must be an AIE::TileOp result; the SSA edge keeps the tile from being
    // DCE'd.
    AIE::TileOp memtile = linkOp.memtile.getDefiningOp<AIE::TileOp>();
    if (!memtile) {
      linkOp.emitError("conduit-to-dma: relay-op memtile operand is not the "
                       "result of an aie.tile op");
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
      ConduitInfo *coreRelaySrcPtr =
          state.lookupConduit(coreRelayName, linkOp.op);
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
        std::string memtileStr = "tile(" + std::to_string(memtile.getCol()) +
                                 "," + std::to_string(memtile.getRow()) + ")";
        linkOp.emitError("conduit-to-dma: CoreTile relay: relay buffers for "
                         "src conduit '" +
                         coreRelayName + "' not found on relay tile '" +
                         memtileStr + "'");
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

      int64_t relayDepth = coreRelaySrc.depth > 0 ? coreRelaySrc.depth : 1;

      // If relay locks are missing (e.g., shim→relay path where Phase 3 doesn't
      // allocate consumer-side locks on the relay tile), allocate them now.
      if (!relayProdLock && !coreRelaySrc.noLocks) {
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
          AIE::LockOp lk =
              builder.create<AIE::LockOp>(loc, relayTileVal, lockIdx, 0);
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          relayConsLock = lk;
          coreRelaySrc.consumerTileLocks[relayTileVal].second = lk;
        }
      }

      int64_t relayPerBufLen =
          deriveBdLength(coreRelaySrc.elemType, coreRelaySrc.numElems);

      // Retrieve the S2MM channel pre-assigned by Phase 4a.
      // Use the device-qualified-then-unqualified helper so multi-device
      // modules find the channel that Phase 4 wrote under the qualified key.
      int32_t relaySrcS2MMCh =
          state.lookupS2MMChannel(coreRelayName, 0u, linkOp.op);
      if (relaySrcS2MMCh < 0)
        relaySrcS2MMCh = state.tileNextS2MMChannel[relayTileVal]++;

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
        auto memOp = builder.create<AIE::MemOp>(loc, relayTileVal);
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
                                        relaySrcS2MMCh, 0, s2mmBDs[0],
                                        s2mmExit);

        for (int64_t i = 0; i < relayDepth; ++i) {
          mlir::Value acqLock =
              relayProdLock ? relayProdLock.getResult() : mlir::Value{};
          mlir::Value relLock =
              relayConsLock ? relayConsLock.getResult() : mlir::Value{};
          AIE::BDDimLayoutArrayAttr relayConsDims;
          if (!coreRelaySrc.consumerDimensions.empty())
            relayConsDims = coreRelaySrc.consumerDimensions[0];
          state.emitBDBlock(
              loc, s2mmBDs[i], acqLock, state.lockAcqValue(Port::Produce, 1),
              relayBufs[i % relayBufs.size()].getResult(), 0, relayPerBufLen,
              relLock, state.lockRelValue(Port::Produce), relayConsDims);
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
                mlir::cast<mlir::FlatSymbolRefAttr>(dsts[dstIdx])
                    .getValue()
                    .str();
            ConduitInfo *dstInfo = state.lookupConduit(dstName, linkOp.op);
            if (!dstInfo || dstInfo->consumerTileCoords.empty()) {
              continue;
            }

            // Allocate MM2S channel on relay tile.
            int32_t mm2sCh = state.tileNextMM2SChannel[relayTileVal]++;

            // Emit flows and record S2MM channels on consumer tiles.
            for (unsigned consIdx = 0;
                 consIdx < dstInfo->consumerTileCoords.size(); ++consIdx) {
              auto [consCol, consRow] = dstInfo->consumerTileCoords[consIdx];
              AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
              if (!consTile)
                continue;
              mlir::Value consTileVal = consTile.getResult();
              int32_t s2mmCh = state.tileNextS2MMChannel[consTileVal]++;
              state.insertS2MMChannel(dstName, consIdx, s2mmCh, linkOp.op);
              builder.setInsertionPoint(state.deviceBody->getTerminator());
              builder.create<AIE::FlowOp>(
                  loc, relayTileVal, AIE::WireBundle::DMA, mm2sCh, consTileVal,
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
                                            mm2sCh, 0, mm2sBDs[0],
                                            nextChainOrEnd);

            for (int64_t i = 0; i < relayDepth; ++i) {
              // MM2S: acq consLock (data ready), send, rel prodLock (space
              // free)
              mlir::Value acqLock =
                  relayConsLock ? relayConsLock.getResult() : mlir::Value{};
              mlir::Value relLock =
                  relayProdLock ? relayProdLock.getResult() : mlir::Value{};
              state.emitBDBlock(loc, mm2sBDs[i], acqLock,
                                state.lockAcqValue(Port::Consume, 1),
                                relayBufs[i % relayBufs.size()].getResult(), 0,
                                relayPerBufLen, relLock,
                                state.lockRelValue(Port::Consume),
                                dstInfo->producerDimensions);
              builder.create<AIE::NextBDOp>(loc, mm2sBDs[(i + 1) % relayDepth]);
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
    ConduitInfo *srcInfoPtr = state.lookupConduit(srcName, linkOp.op);

    // Guard: cascade-mode conduits cannot be used with distribute/join/forward.
    if (srcInfoPtr && srcInfoPtr->routingMode == RoutingMode::Cascade) {
      linkOp.emitError(
          "conduit distribute/join/forward cannot use cascade-mode "
          "conduit '" +
          srcName +
          "' — cascade is point-to-point "
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
      isMemTileRelaySrc = (sp >= 0 && sr >= 0 && targetModel.isMemTile(sp, sr));
    }
    if (!srcInfoPtr || (srcInfoPtr->buffers.empty() && !isMemTileRelaySrc)) {
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
                         ? memBufsIt->second
                         : srcInfo.buffers;

    int64_t linkDepth = srcInfo.depth > 0 ? srcInfo.depth : 1;
    // perBufLen: number of elements per physical buffer for this source
    // conduit at the memtile. See deriveBdLength comment for the
    // single-source-of-truth rationale (avoid SHIM-window overflow at the
    // memtile in DISTRIBUTE last-slice fall-through arms; mirrors the JOIN
    // path fix below).
    int64_t perBufLen = deriveBdLength(srcInfo.elemType, srcInfo.numElems);

    // Per-destination independent lock pairs on the MemTile
    // (distribute/forward).
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
    if (isDistribute && numDsts > 0 && !srcInfo.noLocks) {
      builder.setInsertionPoint(lockInsertionPoint);
      for (unsigned sliceIdx = 0; sliceIdx < numDsts; ++sliceIdx) {
        // Scale per-slice lock init by the destination fifo's bd_repeat.
        // The MemTile MM2S fires linkDepth×repeat times before releasing.
        int64_t dstRepeat = 1;
        {
          std::string dstNameR =
              mlir::cast<mlir::FlatSymbolRefAttr>(dsts[sliceIdx])
                  .getValue()
                  .str();
          if (ConduitInfo *dstInfoR = state.lookupConduit(dstNameR, linkOp.op))
            if (dstInfoR->bdRepeat > 1)
              dstRepeat = dstInfoR->bdRepeat;
        }
        int64_t sliceProdInit = linkDepth * dstRepeat;

        if (isAIE2) {
          {
            int lockIdx = state.lockIdCounter[memtileVal]++;
            std::string symName =
                srcName + "_link_prod_lock_" + std::to_string(sliceIdx);
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                state.deviceOp.getLoc(), memtileVal, lockIdx,
                static_cast<int>(sliceProdInit));
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            sliceProdLocks.push_back(lk);
          }
          {
            int lockIdx = state.lockIdCounter[memtileVal]++;
            std::string symName =
                srcName + "_link_cons_lock_" + std::to_string(sliceIdx);
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                state.deviceOp.getLoc(), memtileVal, lockIdx, 0);
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            sliceConsLocks.push_back(lk);
          }
        } else {
          int lockIdx = state.lockIdCounter[memtileVal]++;
          std::string symName =
              srcName + "_link_lock_" + std::to_string(sliceIdx);
          AIE::LockOp lk = builder.create<AIE::LockOp>(state.deviceOp.getLoc(),
                                                       memtileVal, lockIdx, 0);
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
      std::string jDstName =
          mlir::cast<mlir::FlatSymbolRefAttr>(dsts[0]).getValue().str();
      ConduitInfo *jDstInfo = state.lookupConduit(jDstName, linkOp.op);
      if (!jDstInfo) {
        linkOp.emitWarning("conduit-to-dma: join destination conduit '")
            << jDstName << "' not found — BD lengths defaulting to 1";
      } else {
        int64_t jDstDepth = jDstInfo->depth > 0 ? jDstInfo->depth : 1;

        // Derive memtile JOIN buffer length via deriveBdLength to avoid the
        // SHIM aggregated per-dispatch window overflowing the per-buffer
        // size in the last-slice fall-through arm.
        mlir::Type intBufTy = jDstInfo->elemType;
        joinDstPerBufForLen = deriveBdLength(intBufTy, jDstInfo->numElems);

        if (!intBufTy)
          intBufTy = mlir::MemRefType::get({joinDstPerBufForLen},
                                           mlir::IntegerType::get(ctx, 32));

        builder.setInsertionPoint(lockInsertionPoint);
        unsigned numJoinSrcs = static_cast<unsigned>(srcs.size());

        // Allocate join intermediate buffers on the memtile (depth-many).
        // Record them in jDstInfo->buffers so Phase 6/7 can look them up.
        for (int64_t i = 0; i < jDstDepth; ++i) {
          std::string symName = jDstName + "_buff_" + std::to_string(i);
          auto buf = builder.create<AIE::BufferOp>(
              state.deviceOp.getLoc(), intBufTy, memtileVal,
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
        if (!jDstInfo->noLocks) {
          for (unsigned srcIdx = 0; srcIdx < numJoinSrcs; ++srcIdx) {
            if (isAIE2) {
              {
                int lockIdx = state.lockIdCounter[memtileVal]++;
                std::string symName =
                    jDstName + "_prod_lock_" + std::to_string(srcIdx);
                AIE::LockOp lk = builder.create<AIE::LockOp>(
                    state.deviceOp.getLoc(), memtileVal, lockIdx,
                    static_cast<int>(jDstDepth));
                lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                joinSrcProdLocks.push_back(lk);
              }
              {
                int lockIdx = state.lockIdCounter[memtileVal]++;
                std::string symName =
                    jDstName + "_cons_lock_" + std::to_string(srcIdx);
                AIE::LockOp lk = builder.create<AIE::LockOp>(
                    state.deviceOp.getLoc(), memtileVal, lockIdx, 0);
                lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                joinSrcConsLocks.push_back(lk);
              }
            } else {
              int lockIdx = state.lockIdCounter[memtileVal]++;
              std::string symName =
                  jDstName + "_lock_" + std::to_string(srcIdx);
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
      // Device-qualified-then-unqualified lookup so multi-device modules
      // find the channel that Phase 4 wrote under the qualified key.
      ingestS2MMCh = state.lookupS2MMChannel(srcName, 0u, linkOp.op);
      if (ingestS2MMCh < 0)
        ingestS2MMCh = state.tileNextS2MMChannel[memtileVal]++;
    }

    // Distribute: per-destination MM2S channels on the memtile.
    // If Phase 4b already assigned an MM2S channel for a destination (e.g.,
    // a shim consumer flow), reuse it instead of allocating a new one.
    // Without this, the DMA start uses a freshly allocated channel that
    // does not match the flow emitted by Phase 4b.
    llvm::SmallVector<int32_t> distMM2SChannels;
    if (isDistribute) {
      for (unsigned i = 0; i < numDsts; ++i) {
        std::string dstName =
            mlir::cast<mlir::FlatSymbolRefAttr>(dsts[i]).getValue().str();
        // Device-qualified-then-unqualified lookup so multi-device modules
        // find the channel that Phase 4b wrote under the qualified key.
        int32_t ch = state.lookupMM2SChannel(dstName, linkOp.op);
        if (ch < 0)
          ch = state.tileNextMM2SChannel[memtileVal]++;
        distMM2SChannels.push_back(ch);
      }
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
        ConduitInfo *sInfo = state.lookupConduit(sName, linkOp.op);
        bool reused = false;
        if (sInfo) {
          auto [sp, sr] = sInfo->producerTileCoord;
          if (sp >= 0 && sr >= 0 && targetModel.isMemTile(sp, sr)) {
            // Device-qualified-then-unqualified lookup so multi-device
            // modules find the channel that Phase 4 wrote under the
            // qualified key.
            int32_t ch = state.lookupS2MMChannel(sName, 0u, linkOp.op);
            if (ch >= 0) {
              joinS2MMChannels.push_back(ch);
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
          // Multi-device fix: write under the device-qualified key so the
          // qualified-then-unqualified helper used by other read sites finds
          // the entry (matches the Phase 4 writer convention that iterates
          // `conduitMap` whose keys are produced by `makeConduitKey`).
          state.insertS2MMChannel(sName, 0u, ch, linkOp.op);
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
      // Device-qualified-then-unqualified lookup so multi-device modules
      // find the channel that Phase 4b wrote under the qualified key.
      joinMM2SCh = state.lookupMM2SChannel(dstName0, linkOp.op);
      if (joinMM2SCh < 0)
        joinMM2SCh = state.tileNextMM2SChannel[memtileVal]++;
    }

    // -----------------------------------------------------------------------
    // Emit aie.flow ops for distribute/join.
    // -----------------------------------------------------------------------
    if (isDistribute) {
      builder.setInsertionPoint(state.deviceBody->getTerminator());
      for (unsigned dstIdx = 0; dstIdx < numDsts; ++dstIdx) {
        std::string dstName =
            mlir::cast<mlir::FlatSymbolRefAttr>(dsts[dstIdx]).getValue().str();
        ConduitInfo *dstInfo = state.lookupConduit(dstName, linkOp.op);
        if (!dstInfo || dstInfo->consumerTileCoords.empty())
          continue;

        int32_t mm2sCh =
            dstIdx < distMM2SChannels.size() ? distMM2SChannels[dstIdx] : 0;

        // Determine routing mode for this dst conduit.
        std::optional<RoutingMode> dstRoutingMode =
            dstInfo ? dstInfo->routingMode : RoutingMode::Circuit;

        if (dstRoutingMode == RoutingMode::Packet) {
          // Packet-mode broadcast: emit ONE multi-dest aie.packet_flow.
          // Allocate S2MM channels per consumer first, then build the flow.
          if (!state.packetIDAllocator) {
            state.module.emitError(
                "internal error: packetIDAllocator not initialized");
            state.passFailed = true;
            return;
          }
          mlir::Value pktDomain = state.getMemTileDomain(memtileVal);
          std::optional<uint8_t> pktID =
              state.packetIDAllocator->allocate(pktDomain);
          if (!pktID) {
            state.passFailed = true;
            return;
          }
          state.insertPacketID(dstName, *pktID, linkOp.op);

          auto pktFlow = builder.create<AIE::PacketFlowOp>(
              state.deviceOp.getLoc(), static_cast<int8_t>(*pktID),
              /*keep_pkt_header=*/mlir::BoolAttr{},
              /*priority_route=*/mlir::BoolAttr{});
          mlir::Region &region = pktFlow.getPorts();
          mlir::Block *pktBlock = builder.createBlock(&region);
          builder.setInsertionPointToStart(pktBlock);
          builder.create<AIE::PacketSourceOp>(state.deviceOp.getLoc(),
                                              memtileVal, AIE::WireBundle::DMA,
                                              static_cast<int32_t>(mm2sCh));

          for (unsigned consIdx = 0;
               consIdx < dstInfo->consumerTileCoords.size(); ++consIdx) {
            auto [dstConsCol, dstConsRow] =
                dstInfo->consumerTileCoords[consIdx];
            AIE::TileOp dstConsTile =
                state.lookupTileByCoord(dstConsCol, dstConsRow);
            if (!dstConsTile)
              continue;
            mlir::Value consTileVal = dstConsTile.getResult();
            // S2MM port assignment: only share S2MM ports between packet
            // channels that have the same dma_channel_group.  Independent
            // packet channels (no group) get separate S2MM ports to prevent
            // data crossover.
            // Effective group key: dma_channel_group_s2mm if set, else
            // dma_channel_group.
            std::string s2mmGrp = state.qualifyFuseGroup(
                !dstInfo->fuseGroupS2MM.empty() ? dstInfo->fuseGroupS2MM
                                                : dstInfo->fuseGroup,
                linkOp.deviceIndex);
            int32_t s2mmCh;
            bool s2mmShared = false;
            if (!s2mmGrp.empty()) {
              auto it = state.fuseGroupS2MMChannel.find(s2mmGrp);
              if (it != state.fuseGroupS2MMChannel.end()) {
                s2mmCh = it->second;
                s2mmShared = true;
              }
            }
            if (!s2mmShared) {
              s2mmCh = state.tileNextS2MMChannel[consTileVal]++;
              if (!s2mmGrp.empty())
                state.fuseGroupS2MMChannel[s2mmGrp] = s2mmCh;
            }
            state.insertS2MMChannel(dstName, consIdx, s2mmCh, linkOp.op);

            // Lock sharing: only share locks when channels share an S2MM
            // port (same dma_channel_group).
            if (s2mmShared) {
              auto lockIt = state.pktTileS2MMLock.find(consTileVal);
              if (lockIt != state.pktTileS2MMLock.end()) {
                dstInfo->consumerTileLocks[consTileVal] = {
                    lockIt->second.first.getDefiningOp<AIE::LockOp>(),
                    lockIt->second.second.getDefiningOp<AIE::LockOp>()};
              }
            } else if (!s2mmGrp.empty()) {
              auto &locks = dstInfo->consumerTileLocks[consTileVal];
              if (locks.first && locks.second) {
                state.pktTileS2MMLock[consTileVal] = {locks.first.getResult(),
                                                      locks.second.getResult()};
              }
            }

            builder.create<AIE::PacketDestOp>(
                state.deviceOp.getLoc(), dstConsTile.getResult(),
                AIE::WireBundle::DMA, static_cast<int32_t>(s2mmCh));
          }
          builder.create<AIE::EndOp>(state.deviceOp.getLoc());
          builder.setInsertionPointAfter(pktFlow);
        } else {
          // Circuit or other routing: emit one flow per consumer tile.
          for (unsigned consIdx = 0;
               consIdx < dstInfo->consumerTileCoords.size(); ++consIdx) {
            auto [dstConsCol, dstConsRow] =
                dstInfo->consumerTileCoords[consIdx];
            AIE::TileOp dstConsTile =
                state.lookupTileByCoord(dstConsCol, dstConsRow);
            if (!dstConsTile)
              continue;

            mlir::Value consTileVal = dstConsTile.getResult();
            int32_t s2mmCh = state.tileNextS2MMChannel[consTileVal]++;
            state.insertS2MMChannel(dstName, consIdx, s2mmCh, linkOp.op);

            state.emitFlow(dstRoutingMode, memtileVal, AIE::WireBundle::DMA,
                           mm2sCh, dstConsTile.getResult(),
                           AIE::WireBundle::DMA, static_cast<int32_t>(s2mmCh));
          }
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
            if (srcInfo.routingMode == RoutingMode::Stream) {
              srcBundle = AIE::WireBundle::Core;
              srcPort = srcInfo.aieStreamPort >= 0 ? srcInfo.aieStreamPort : 0;
            } else {
              // Allocate MM2S channel dynamically instead of hardcoding
              // channel 0.  Record in conduitMM2SChannel so Phase 5.5a BD
              // chain generation uses the same channel, and in
              // preUsedMM2SChannels so other phases avoid conflicts.
              srcPort = state.tileNextMM2SChannel[srcProdTile.getResult()]++;
              state.insertMM2SChannel(srcName, srcPort, linkOp.op);
              state.preUsedMM2SChannels[srcProdTile.getResult()].insert(
                  srcPort);
            }
            builder.create<AIE::FlowOp>(
                state.deviceOp.getLoc(), srcProdTile.getResult(), srcBundle,
                srcPort, memtileVal, AIE::WireBundle::DMA,
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
        ConduitInfo *sInfo = state.lookupConduit(sName, linkOp.op);
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
        AIE::TileOp srcProdTile =
            state.lookupTileByCoord(srcProdCol, srcProdRow);
        if (!srcProdTile)
          continue;
        int32_t s2mmCh =
            srcIdx < joinS2MMChannels.size() ? joinS2MMChannels[srcIdx] : 0;
        // A-1 fix: Allocate MM2S channel on the source producer tile
        // dynamically instead of using hardcoded channel 0.  Record in
        // conduitMM2SChannel so Phase 5.5 BD chain generation uses the
        // same channel, and in preUsedMM2SChannels so Phase 5.5a
        // (distribute source) avoids conflicts on the same tile.
        int32_t srcMM2SCh =
            state.tileNextMM2SChannel[srcProdTile.getResult()]++;
        state.insertMM2SChannel(sName, srcMM2SCh, linkOp.op);
        state.preUsedMM2SChannels[srcProdTile.getResult()].insert(srcMM2SCh);
        builder.create<AIE::FlowOp>(state.deviceOp.getLoc(),
                                    srcProdTile.getResult(),
                                    AIE::WireBundle::DMA, srcMM2SCh, memtileVal,
                                    AIE::WireBundle::DMA, s2mmCh);
      }

      // Destination flow: memtile MM2S → dst compute consumer.
      // NOTE: shim consumer flows are handled by Phase 4b (routePhase) which
      // iterates all conduits with shimConsumerTileCoords. We must NOT emit
      // the shim flow here to avoid duplicating Phase 4b's emission.
      if (!dsts.empty()) {
        std::string dstName =
            mlir::cast<mlir::FlatSymbolRefAttr>(dsts[0]).getValue().str();
        if (ConduitInfo *dstFlowInfo =
                state.lookupConduit(dstName, linkOp.op)) {
          for (unsigned ci = 0; ci < dstFlowInfo->consumerTileCoords.size();
               ++ci) {
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
              state.insertS2MMChannel(dstName, ci, consS2MM, linkOp.op);
              builder.create<AIE::FlowOp>(state.deviceOp.getLoc(), memtileVal,
                                          AIE::WireBundle::DMA, joinMM2SCh,
                                          consTile.getResult(),
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
      // Reuse existing MemTileDMAOp — find the LAST aie.end block.
      // Scanning for the first could pick up orphan aie.end blocks from
      // packet-muxed distribute sub-chains, disconnecting the DMA chain.
      dmaRegionPtr = &mtIt->second.getBody();
      for (mlir::Block &block : *dmaRegionPtr) {
        if (auto *term = block.getTerminator()) {
          if (mlir::isa<AIE::EndOp>(term)) {
            mergeChainBlock = &block;
          }
        }
      }
      if (mergeChainBlock)
        mergeChainBlock->getTerminator()->erase();
    } else {
      builder.setInsertionPoint(state.deviceBody->getTerminator());
      auto memtileDMA = builder.create<AIE::MemTileDMAOp>(loc, memtileVal);
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
      mlir::Block *entryBlock = mergeChainBlock ? mergeChainBlock : addBlock();

      llvm::SmallVector<mlir::Block *> ingestBlocks;
      for (int64_t bufIdx = 0; bufIdx < linkDepth; ++bufIdx)
        for (unsigned sliceIdx = 0; sliceIdx < numDsts; ++sliceIdx)
          ingestBlocks.push_back(addBlock());

      mm2sChainStartBlock = addBlock();

      builder.setInsertionPointToEnd(entryBlock);
      builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::S2MM,
                                      ingestS2MMCh, static_cast<int32_t>(0),
                                      ingestBlocks[0], mm2sChainStartBlock);

      unsigned totalIngest = static_cast<unsigned>(linkDepth) * numDsts;
      for (unsigned blkIdx = 0; blkIdx < totalIngest; ++blkIdx) {
        unsigned bufIdx = blkIdx / numDsts;
        unsigned sliceIdx = blkIdx % numDsts;

        int64_t dstOffset = 0;
        int64_t dstLen = perBufLen;
        if (offsets.has_value() && !offsets->empty()) {
          if (sliceIdx < static_cast<unsigned>(offsets->size()))
            dstOffset = (*offsets)[sliceIdx];
          if (sliceIdx + 1 < static_cast<unsigned>(offsets->size()))
            dstLen = (*offsets)[sliceIdx + 1] - dstOffset;
          else
            dstLen = perBufLen - dstOffset;
        }

        mlir::Value ingestAcqLock = (sliceIdx < sliceProdLocks.size())
                                        ? sliceProdLocks[sliceIdx].getResult()
                                        : mlir::Value{};
        mlir::Value ingestRelLock = (sliceIdx < sliceConsLocks.size())
                                        ? sliceConsLocks[sliceIdx].getResult()
                                        : mlir::Value{};
        AIE::BDDimLayoutArrayAttr ingestDims;
        if (!srcInfo.consumerDimensions.empty())
          ingestDims = srcInfo.consumerDimensions[0];
        state.emitBDBlock(loc, ingestBlocks[blkIdx], ingestAcqLock,
                          state.lockAcqValue(Port::Produce, 1),
                          linkBufs[bufIdx].getResult(), dstOffset, dstLen,
                          ingestRelLock, state.lockRelValue(Port::Produce),
                          ingestDims);
        builder.create<AIE::NextBDOp>(loc,
                                      ingestBlocks[(blkIdx + 1) % totalIngest]);
      }

    } else {
      // Join: N independent S2MM channels.
      unsigned numSrcs = static_cast<unsigned>(srcs.size());
      int64_t jDepth = static_cast<int64_t>(joinIntermediateBuffers.size());
      if (jDepth == 0)
        jDepth = 1;

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
      llvm::SmallVector<llvm::SmallVector<mlir::Block *>> srcIngestBlocks(
          numSrcs);
      // If merging, the first entry reuses the previous EndOp block.
      if (mergeChainBlock) {
        s2mmEntries[0] = mergeChainBlock;
        for (int64_t i = 0; i < jDepth; ++i)
          srcIngestBlocks[0].push_back(addBlock());
      }
      for (unsigned srcIdx = (mergeChainBlock ? 1u : 0u); srcIdx < numSrcs;
           ++srcIdx) {
        s2mmEntries[srcIdx] = addBlock();
        for (int64_t i = 0; i < jDepth; ++i)
          srcIngestBlocks[srcIdx].push_back(addBlock());
      }
      mm2sChainStartBlock = addBlock();

      for (unsigned srcIdx = 0; srcIdx < numSrcs; ++srcIdx) {
        int64_t srcOffset = srcOffsets[srcIdx];
        int64_t srcLen = srcLens[srcIdx];
        mlir::Block *nextBlock = (srcIdx + 1 < numSrcs)
                                     ? s2mmEntries[srcIdx + 1]
                                     : mm2sChainStartBlock;

        int32_t s2mmCh =
            srcIdx < joinS2MMChannels.size() ? joinS2MMChannels[srcIdx] : 0;
        builder.setInsertionPointToEnd(s2mmEntries[srcIdx]);
        builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::S2MM, s2mmCh,
                                        0, srcIngestBlocks[srcIdx][0],
                                        nextBlock);

        mlir::Value jAcqLock = (srcIdx < joinSrcProdLocks.size())
                                   ? joinSrcProdLocks[srcIdx].getResult()
                                   : mlir::Value{};
        mlir::Value jRelLock = (srcIdx < joinSrcConsLocks.size())
                                   ? joinSrcConsLocks[srcIdx].getResult()
                                   : mlir::Value{};
        for (int64_t i = 0; i < jDepth; ++i) {
          mlir::Value buf =
              !joinIntermediateBuffers.empty()
                  ? joinIntermediateBuffers[i % joinIntermediateBuffers.size()]
                        .getResult()
                  : mlir::Value{};
          state.emitBDBlock(loc, srcIngestBlocks[srcIdx][i], jAcqLock,
                            state.lockAcqValue(Port::Produce, 1), buf,
                            srcOffset, srcLen, jRelLock,
                            state.lockRelValue(Port::Produce));
          builder.create<AIE::NextBDOp>(
              loc, srcIngestBlocks[srcIdx][(i + 1) % jDepth]);
        }
      }
    }

    // Build MM2S send chains.
    mlir::Block *prevChainBlock =
        mm2sChainStartBlock ? mm2sChainStartBlock : addBlock();
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

      // Look up dst conduit's producerDimensions (dims_to_stream) for the
      // join MM2S BDs, matching how the distribute path passes dstProdDims.
      AIE::BDDimLayoutArrayAttr joinDstProdDims;
      if (!dsts.empty()) {
        std::string joinDstName =
            mlir::cast<mlir::FlatSymbolRefAttr>(dsts[0]).getValue().str();
        ConduitInfo *dstInfo = state.lookupConduit(joinDstName, linkOp.op);
        if (dstInfo)
          joinDstProdDims = dstInfo->producerDimensions;
      }

      llvm::SmallVector<mlir::Block *> sendBDBlocks;
      for (unsigned i = 0; i < totalBDs; ++i)
        sendBDBlocks.push_back(addBlock());
      endBlock = addBlock();

      builder.setInsertionPointToEnd(prevChainBlock);
      builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::MM2S, joinMM2SCh,
                                      0, sendBDBlocks[0], endBlock);

      for (unsigned bdIdx = 0; bdIdx < totalBDs; ++bdIdx) {
        unsigned bufIdx = bdIdx / numJoinSrcs;
        unsigned srcIdx = bdIdx % numJoinSrcs;
        mlir::Value acqLock = (srcIdx < joinSrcConsLocks.size())
                                  ? joinSrcConsLocks[srcIdx].getResult()
                                  : mlir::Value{};
        mlir::Value relLock = (srcIdx < joinSrcProdLocks.size())
                                  ? joinSrcProdLocks[srcIdx].getResult()
                                  : mlir::Value{};
        state.emitBDBlock(
            loc, sendBDBlocks[bdIdx], acqLock,
            state.lockAcqValue(Port::Consume, 1),
            joinIntermediateBuffers[bufIdx % joinIntermediateBuffers.size()]
                .getResult(),
            mm2sOffsets[srcIdx], mm2sLens[srcIdx], relLock,
            state.lockRelValue(Port::Consume), joinDstProdDims);
        builder.create<AIE::NextBDOp>(loc,
                                      sendBDBlocks[(bdIdx + 1) % totalBDs]);
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

        // Look up dst fifo's producerDimensions, bd_repeat, and packet ID.
        AIE::BDDimLayoutArrayAttr dstProdDims;
        int64_t mm2sDstRepeat = 1;
        int dstPktID = -1;
        {
          std::string dstName2 =
              mlir::cast<mlir::FlatSymbolRefAttr>(dsts[dstIdx])
                  .getValue()
                  .str();
          if (ConduitInfo *dstInfo = state.lookupConduit(dstName2, linkOp.op)) {
            dstProdDims = dstInfo->producerDimensions;
            if (dstInfo->bdRepeat > 1)
              mm2sDstRepeat = dstInfo->bdRepeat;
          }
          int32_t pktLookup = state.lookupPacketID(dstName2, linkOp.op);
          if (pktLookup >= 0)
            dstPktID = pktLookup;
        }
        // Unroll by bd_repeat: each source buffer is sent repeat times.
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

        int32_t mm2sCh =
            dstIdx < distMM2SChannels.size() ? distMM2SChannels[dstIdx] : 0;
        builder.setInsertionPointToEnd(prevChainBlock);
        builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::MM2S, mm2sCh,
                                        0, sendBDBlocks[0], nextChainBlock);

        for (int64_t i = 0; i < thisDstEffective; ++i) {
          // Each source buffer (linkBufs[j]) is repeated mm2sDstRepeat times.
          mlir::Value buf =
              !linkBufs.empty()
                  ? linkBufs[(i / mm2sDstRepeat) % linkBufs.size()].getResult()
                  : mlir::Value{};
          state.emitBDBlock(
              loc, sendBDBlocks[i],
              mm2sAcqLock ? mm2sAcqLock.getResult() : mlir::Value{},
              state.lockAcqValue(Port::Consume, 1), buf, dstOffset, dstLen,
              mm2sRelLock ? mm2sRelLock.getResult() : mlir::Value{},
              state.lockRelValue(Port::Consume), dstProdDims, dstPktID);
          builder.create<AIE::NextBDOp>(
              loc, sendBDBlocks[(i + 1) % thisDstEffective]);
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

  // Erase distribute/join/forward/scatter/gather ops after processing
  // (collect-then-erase).
  for (auto *op : llvm::reverse(linkOpsToErase))
    op->erase();

  // Erase any remaining scatter/gather ops not consumed by the adapter loop
  // (e.g., those that hit a continue due to missing memtile inference).
  {
    llvm::SmallVector<mlir::Operation *> remainingScatterGather;
    state.module.walk([&](ScatterOp op) {
      remainingScatterGather.push_back(op.getOperation());
    });
    state.module.walk([&](GatherOp op) {
      remainingScatterGather.push_back(op.getOperation());
    });
    for (auto *op : llvm::reverse(remainingScatterGather))
      op->erase();
  }

  if (state.passFailed)
    return;

  // -----------------------------------------------------------------------
  // Phase 5.5: aie.mem BD chains for simple (non-link) conduits.
  // -----------------------------------------------------------------------

  // Pre-compute used DMA channels per tile (avoids O(n²) scan).
  // Walk ALL devices (not just state.deviceOp) to handle multi-device modules.
  for (auto &devOp : state.deviceOps) {
    devOp.walk([&](AIE::DMAStartOp dmaStart) {
      mlir::Value parentTile;
      if (auto memOp = mlir::dyn_cast<AIE::MemOp>(dmaStart->getParentOp()))
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
  }

  // Pre-compute tile → DMA region map to avoid O(n²) walks inside the
  // conduitMap loop.  Updated when new DMA ops are created below.
  // Walk ALL devices for multi-device support.
  llvm::DenseMap<mlir::Value, mlir::Region *> tileToDMARegion;
  for (auto &devOp : state.deviceOps) {
    devOp.walk([&](AIE::MemOp memOp) {
      tileToDMARegion[memOp.getTile()] = &memOp.getBody();
    });
    devOp.walk([&](AIE::MemTileDMAOp mtOp) {
      tileToDMARegion[mtOp.getTile()] = &mtOp.getBody();
    });
  }

  // -----------------------------------------------------------------------
  // Phase 5.5a: Emit aie.mem MM2S BD chains for distribute/forward link
  // source conduits whose producer is a compute tile.
  //
  // Phase 3 allocates producer-side buffers and locks for these conduits
  // in consumerTileBuffers/consumerTileLocks (keyed by the producer tile
  // value), but does NOT set the top-level info.prodLock/consLock.  The
  // generic Phase 5.5 loop below guards on info.prodLock/consLock being
  // non-null, so distribute/forward sources are filtered out.  Handle
  // them in a dedicated loop here.
  // -----------------------------------------------------------------------
  for (auto &[name, info] : state.conduitMap) {
    // Multi-device: ensure tile lookups target the correct device.
    if (state.isMultiDevice())
      state.switchToDeviceIndex(info.deviceIndex);

    if (!state.linkSrcNamesEarly.count(name))
      continue;
    // Stream conduits: producer uses Core AXI stream port, no DMA needed.
    if (info.routingMode == RoutingMode::Stream)
      continue;
    auto [prodCol, prodRow] = info.producerTileCoord;
    if (prodCol < 0 || prodRow < 2)
      continue; // only compute tiles (row >= 2) need aie.mem MM2S

    AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
    if (!prodTile)
      continue;
    mlir::Value prodTileVal = prodTile.getResult();

    // Look up producer-side buffers and locks from consumerTileBuffers/Locks.
    auto bufIt = info.consumerTileBuffers.find(prodTileVal);
    if (bufIt == info.consumerTileBuffers.end() || bufIt->second.empty())
      continue;
    auto &prodBufs = bufIt->second;

    AIE::LockOp pProdLock, pConsLock;
    if (!info.noLocks) {
      auto lockIt = info.consumerTileLocks.find(prodTileVal);
      if (lockIt == info.consumerTileLocks.end())
        continue; // no locks allocated — cannot emit BD chain
      pProdLock = lockIt->second.first;
      pConsLock = lockIt->second.second;
    }

    int64_t perBufLen = deriveBdLength(info.elemType, info.numElems);
    int64_t nBufs = static_cast<int64_t>(prodBufs.size());

    // Reuse MM2S channel allocated by Phase 5 flow emission, if present.
    // Otherwise allocate a new one (channel 0 unless pre-used).
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

    // Check for an existing aie.mem for this tile.
    mlir::Region *existingRegion = nullptr;
    {
      auto it = tileToDMARegion.find(prodTileVal);
      if (it != tileToDMARegion.end())
        existingRegion = it->second;
    }

    if (existingRegion) {
      // Append MM2S into existing aie.mem.
      mlir::Region &memRegion = *existingRegion;
      mlir::Block *endBlock = nullptr;
      for (mlir::Block &blk : memRegion)
        for (mlir::Operation &op : blk)
          if (mlir::isa<AIE::EndOp>(op))
            endBlock = &blk;

      if (!endBlock) {
        state.deviceOp.emitError(
            "conduit-to-dma: distribute-source append: existing aie.mem has no "
            "aie.end block — region is malformed");
        state.passFailed = true;
        return;
      }

      if (!checkBDChainCap(state, prodTileVal, nBufs, name))
        return;
      auto addBlock = [&]() -> mlir::Block * {
        return builder.createBlock(&memRegion);
      };
      llvm::SmallVector<mlir::Block *> bdBlocks;
      for (int64_t i = 0; i < nBufs; ++i)
        bdBlocks.push_back(addBlock());
      mlir::Block *newEndBlock = addBlock();

      endBlock->back().erase(); // remove old aie.end
      builder.setInsertionPointToEnd(endBlock);
      builder.create<AIE::DMAStartOp>(
          state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S, mm2sChannel,
          static_cast<int32_t>(0), bdBlocks[0], newEndBlock);

      for (int64_t i = 0; i < nBufs; ++i) {
        mlir::Value acqLock =
            isAIE2 ? (pConsLock ? pConsLock.getResult() : mlir::Value{})
                   : (pConsLock ? pConsLock.getResult() : mlir::Value{});
        mlir::Value relLock =
            isAIE2 ? (pProdLock ? pProdLock.getResult() : mlir::Value{})
                   : acqLock;
        state.emitBDBlock(state.deviceOp.getLoc(), bdBlocks[i], acqLock,
                          state.lockAcqValue(Port::Consume, 1),
                          prodBufs[i % prodBufs.size()].getResult(), 0,
                          perBufLen, relLock, state.lockRelValue(Port::Consume),
                          info.producerDimensions);
        builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                      bdBlocks[(i + 1) % nBufs]);
      }
      builder.setInsertionPointToEnd(newEndBlock);
      builder.create<AIE::EndOp>(state.deviceOp.getLoc());
    } else {
      // No existing aie.mem — create one and register it.
      builder.setInsertionPoint(state.deviceBody->getTerminator());
      auto memOp =
          builder.create<AIE::MemOp>(state.deviceOp.getLoc(), prodTileVal);
      mlir::Region *memRegion = &memOp.getBody();
      tileToDMARegion[prodTileVal] = memRegion;

      if (!checkBDChainCap(state, prodTileVal, nBufs, name))
        return;
      auto addMemBlock = [&]() -> mlir::Block * {
        return builder.createBlock(memRegion);
      };
      mlir::Block *dmaStartBlock = addMemBlock();
      llvm::SmallVector<mlir::Block *> bdBlocks;
      for (int64_t i = 0; i < nBufs; ++i)
        bdBlocks.push_back(addMemBlock());
      mlir::Block *endMemBlock = addMemBlock();

      builder.setInsertionPointToEnd(dmaStartBlock);
      builder.create<AIE::DMAStartOp>(
          state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S, mm2sChannel,
          static_cast<int32_t>(0), bdBlocks[0], endMemBlock);

      for (int64_t i = 0; i < nBufs; ++i) {
        mlir::Value acqLock =
            isAIE2 ? (pConsLock ? pConsLock.getResult() : mlir::Value{})
                   : (pConsLock ? pConsLock.getResult() : mlir::Value{});
        mlir::Value relLock =
            isAIE2 ? (pProdLock ? pProdLock.getResult() : mlir::Value{})
                   : acqLock;
        state.emitBDBlock(state.deviceOp.getLoc(), bdBlocks[i], acqLock,
                          state.lockAcqValue(Port::Consume, 1),
                          prodBufs[i % prodBufs.size()].getResult(), 0,
                          perBufLen, relLock, state.lockRelValue(Port::Consume),
                          info.producerDimensions);
        builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                      bdBlocks[(i + 1) % nBufs]);
      }
      builder.setInsertionPointToEnd(endMemBlock);
      builder.create<AIE::EndOp>(state.deviceOp.getLoc());
    }
  }

  // Track packet-muxed S2MM channels that already have a dma_start emitted
  // in Phase 5.5 Case A.  When multiple conduits share the same S2MM port on
  // a consumer tile (via pktTileS2MMChannel), only the FIRST conduit emits a
  // dma_start; subsequent conduits append BD blocks and the post-pass links
  // them into a combined circular ring.
  // Maps consumer tile → (s2mmChannel → synthetic fuse group label).
  llvm::DenseMap<mlir::Value, std::map<int32_t, std::string>> s2mmPktStarted;

  // Track S2MM fuse-group channels that already have a dma_start emitted.
  // Symmetric to s2mmPktStarted but for circuit-mode S2MM fusion groups
  // annotated by --conduit-fuse-channels (dma_channel_group_s2mm attribute).
  // Maps consumer tile → (fuse group label → bool).
  llvm::DenseMap<mlir::Value, llvm::StringMap<bool>> s2mmFuseStarted;

  for (auto &[name, info] : state.conduitMap) {
    // Multi-device: ensure tile lookups target the correct device.
    if (state.isMultiDevice())
      state.switchToDeviceIndex(info.deviceIndex);

    // For disable_synchronization conduits, locks are null by design — skip the
    // lock check. Still skip if buffers are empty (no allocation happened).
    if (info.buffers.empty())
      continue;
    if (!info.noLocks && (!info.prodLock || !info.consLock))
      continue;

    // Handle link source conduits: emit aie.mem MM2S on producer compute tile.
    // Stream conduits: skip entirely — the producer uses a Core stream port,
    // not a DMA engine. No aie.mem or BD chain on the producer tile.
    // Distribute/forward sources (linkSrcNamesEarly) are handled by Phase 5.5a
    // above — they have null info.prodLock/consLock (locks are stored in
    // consumerTileLocks keyed by the producer tile value).
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

      int64_t perBufLen = deriveBdLength(info.elemType, info.numElems);
      mlir::Value prodTileVal = prodTile.getResult();

      // Check for an existing aie.mem for this tile (e.g. created by Phase 5.5
      // for a broadcast consumer S2MM on the same tile).  If one exists, append
      // the MM2S chain into it rather than creating a duplicate aie.mem op.
      // Without this check, a tile that is both a broadcast consumer and a join
      // source receives two separate aie.mem blocks, and aiecc silently
      // discards the second one, causing a hardware deadlock.
      mlir::Region *existingRegion = nullptr;
      {
        auto it = tileToDMARegion.find(prodTileVal);
        if (it != tileToDMARegion.end())
          existingRegion = it->second;
      }

      // A-1 fix: Look up MM2S channel pre-allocated during Phase 5 flow
      // emission.  Falls back to dynamic allocation for gather sources
      // not processed by Phase 5 (e.g., MemTile relay sources).
      int32_t joinMM2SChannel = 0;
      {
        auto chIt = state.conduitMM2SChannel.find(name);
        if (chIt != state.conduitMM2SChannel.end()) {
          joinMM2SChannel = chIt->second;
        } else {
          auto &usedCh = state.preUsedMM2SChannels[prodTileVal];
          while (usedCh.count(joinMM2SChannel))
            ++joinMM2SChannel;
          usedCh.insert(joinMM2SChannel);
        }
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
          if (!checkBDChainCap(state, prodTileVal, nBufs, name))
            return;
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
              joinMM2SChannel, static_cast<int32_t>(0), bdBlocks[0],
              newEndBlock);

          for (int64_t i = 0; i < nBufs; ++i) {
            mlir::Value acqLock =
                isAIE2 ? info.consLock.getResult()
                       : (info.aie1Locks.empty()
                              ? info.consLock.getResult()
                              : info.aie1Locks[i % info.aie1Locks.size()]
                                    .getResult());
            mlir::Value relLock = isAIE2 ? info.prodLock.getResult() : acqLock;
            state.emitBDBlock(state.deviceOp.getLoc(), bdBlocks[i], acqLock,
                              state.lockAcqValue(Port::Consume, 1),
                              info.buffers[i % info.buffers.size()].getResult(),
                              0, perBufLen, relLock,
                              state.lockRelValue(Port::Consume),
                              info.producerDimensions);
            builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                          bdBlocks[(i + 1) % nBufs]);
          }
          builder.setInsertionPointToEnd(newEndBlock);
          builder.create<AIE::EndOp>(state.deviceOp.getLoc());
        }
      } else {
        // No existing aie.mem for this tile — create one and register it.
        builder.setInsertionPoint(state.deviceBody->getTerminator());
        auto memOp =
            builder.create<AIE::MemOp>(state.deviceOp.getLoc(), prodTileVal);
        memRegion = &memOp.getBody();
        tileToDMARegion[prodTileVal] = memRegion; // register for later phases

        int64_t nBufs = info.nConsumerBuffers();
        if (!checkBDChainCap(state, prodTileVal, nBufs, name))
          return;
        auto addMemBlock = [&]() -> mlir::Block * {
          return builder.createBlock(memRegion);
        };
        mlir::Block *dmaStartBlock = addMemBlock();
        llvm::SmallVector<mlir::Block *> bdBlocks;
        for (int64_t i = 0; i < nBufs; ++i)
          bdBlocks.push_back(addMemBlock());
        mlir::Block *endMemBlock = addMemBlock();

        builder.setInsertionPointToEnd(dmaStartBlock);
        builder.create<AIE::DMAStartOp>(
            state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S, joinMM2SChannel,
            static_cast<int32_t>(0), bdBlocks[0], endMemBlock);

        for (int64_t i = 0; i < nBufs; ++i) {
          mlir::Value acqLock =
              isAIE2 ? info.consLock.getResult()
                     : (info.aie1Locks.empty()
                            ? info.consLock.getResult()
                            : info.aie1Locks[i % info.aie1Locks.size()]
                                  .getResult());
          mlir::Value relLock = isAIE2 ? info.prodLock.getResult() : acqLock;
          state.emitBDBlock(state.deviceOp.getLoc(), bdBlocks[i], acqLock,
                            state.lockAcqValue(Port::Consume, 1),
                            info.buffers[i % info.buffers.size()].getResult(),
                            0, perBufLen, relLock,
                            state.lockRelValue(Port::Consume),
                            info.producerDimensions);
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
      if (prodCol >= 0 && prodRow >= 1 && !state.linkSrcNames.count(name) &&
          !state.linkDstNames.count(name) && !info.consumerTileCoords.empty()) {
        AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
        if (prodTile) {
          mlir::Value prodTileVal = prodTile.getResult();
          auto bufIt = info.consumerTileBuffers.find(prodTileVal);
          if (bufIt != info.consumerTileBuffers.end() &&
              !bufIt->second.empty()) {
            llvm::SmallVector<AIE::BufferOp> &prodBuffers = bufIt->second;
            int64_t depth = info.depth > 0 ? info.depth : 1;
            int64_t perBufLen = deriveBdLength(info.elemType, info.numElems);

            // Look up packet flow ID for packet-mode channels.
            // When set, each MM2S BD emits aie.dma_bd_packet so the switchbox
            // routes the data to the correct packet_flow destination(s).
            int pktID = -1;
            {
              auto pktIt = state.conduitPacketID.find(name);
              if (pktIt != state.conduitPacketID.end())
                pktID = static_cast<int>(pktIt->second);
            }

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
            if (mm2sAcqLock || info.noLocks) {
              bool prodIsMemTile = targetModel.isMemTile(prodCol, prodRow);

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
                std::string qFG =
                    state.qualifyFuseGroup(info.fuseGroup, info.deviceIndex);
                auto &members = state.fuseGroupMembers[qFG];
                isFusedNonFirst = (!members.empty() && members.front() != name);
              }

              // Compute DMAStartOp repeat_count from dma_repeat.
              // AIE compute tile DMAs must cycle infinitely (repeat_count=0)
              // — the core controls lifetime via its main() function.
              // Only MemTile and Shim DMAs use finite repeat_count.
              int32_t dmaRepeatCount =
                  (info.dmaRepeat > 0 && prodIsMemTile)
                      ? static_cast<int32_t>(info.dmaRepeat - 1)
                      : 0;
              // BD chain repeat factor for objectfifo bd_repeat.
              int64_t bdRepeat = info.bdRepeat > 1 ? info.bdRepeat : 1;
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
                  if (!checkBDChainCap(state, prodTileVal, effectiveBDs, name))
                    return;
                  auto addBlock = [&]() -> mlir::Block * {
                    return builder.createBlock(&memRegion);
                  };
                  llvm::SmallVector<mlir::Block *> bdBlocks;
                  for (int64_t i = 0; i < effectiveBDs; ++i)
                    bdBlocks.push_back(addBlock());

                  // For finite chains (dma_repeat > 0), create a dedicated BD
                  // terminal block BEFORE newEndBlock.  The scan for the next
                  // channel's "endBlock" iterates blocks in insertion order and
                  // returns the LAST block with aie.end; since bdTermBlock is
                  // inserted first, newEndBlock (last) is selected by the next
                  // channel and its aie.end is replaced.  bdTermBlock keeps its
                  // aie.end permanently, satisfying the
                  // AIEAssignBufferDescriptorIDs assertion: "bb that's not in
                  // blockMap can only have aie.end".
                  // Only create terminal blocks for non-fused or first-in-group
                  // members.  Fused non-first members must NOT create orphan
                  // aie.end blocks — those would be found by subsequent
                  // channels' "find LAST aie.end" scan and cause new
                  // dma_start ops to be attached to unreachable blocks.
                  mlir::Block *bdTermBlock = nullptr;
                  mlir::Block *newEndBlock = nullptr;

                  if (isFusedNonFirst) {
                    // Non-first fused member: no new dma_start, no terminal
                    // blocks.  BD blocks are linked into the combined ring by
                    // the fuse post-pass.
                  } else {
                    bdTermBlock = (info.dmaRepeat > 0) ? addBlock() : nullptr;
                    newEndBlock = addBlock();
                    mlir::Operation *oldEnd = endBlock->getTerminator();
                    builder.setInsertionPointToEnd(endBlock);
                    oldEnd->erase();
                    builder.create<AIE::DMAStartOp>(
                        state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
                        mm2sChannel, dmaRepeatCount, bdBlocks[0], newEndBlock);
                  }

                  for (int64_t i = 0; i < effectiveBDs; ++i) {
                    // Null locks for disable_synchronization — emitBDBlock
                    // skips them.
                    mlir::Value blockAcq =
                        mm2sAcqLock
                            ? (isAIE2
                                   ? mm2sAcqLock.getResult()
                                   : (prodAIE1Locks && !prodAIE1Locks->empty()
                                          ? (*prodAIE1Locks)[i % prodAIE1Locks
                                                                     ->size()]
                                                .getResult()
                                          : mm2sAcqLock.getResult()))
                            : mlir::Value{};
                    mlir::Value blockRel =
                        mm2sRelLock
                            ? (isAIE2 ? mm2sRelLock.getResult() : blockAcq)
                            : mlir::Value{};
                    state.emitBDBlock(
                        state.deviceOp.getLoc(), bdBlocks[i], blockAcq,
                        state.lockAcqValue(Port::Consume, 1),
                        prodBuffers[(i / bdRepeat) % prodBuffers.size()]
                            .getResult(),
                        0, perBufLen, blockRel,
                        state.lockRelValue(Port::Consume),
                        info.producerDimensions, pktID);
                    // When dma_repeat > 0, the DMA engine uses repeat_count
                    // to replay the BD chain — loop back to the first BD.
                    bool isLast =
                        (i == effectiveBDs - 1) && (info.dmaRepeat > 0);
                    if (isLast)
                      builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                                    bdBlocks[0]);
                    else
                      builder.create<AIE::NextBDOp>(
                          state.deviceOp.getLoc(),
                          bdBlocks[(i + 1) % effectiveBDs]);
                  }
                  if (bdTermBlock) {
                    builder.setInsertionPointToEnd(bdTermBlock);
                    builder.create<AIE::EndOp>(state.deviceOp.getLoc());
                  }
                  if (newEndBlock) {
                    builder.setInsertionPointToEnd(newEndBlock);
                    builder.create<AIE::EndOp>(state.deviceOp.getLoc());
                  }

                  if (!info.fuseGroup.empty())
                    state.conduitBDRange[name] = {bdBlocks.front(),
                                                  bdBlocks.back()};
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
                if (!checkBDChainCap(state, prodTileVal, effectiveBDs, name))
                  return;
                auto addBlock = [&]() -> mlir::Block * {
                  return builder.createBlock(&memRegion);
                };
                mlir::Block *dmaStartBlock = nullptr;
                if (!isFusedNonFirst)
                  dmaStartBlock = addBlock();
                llvm::SmallVector<mlir::Block *> bdBlocks;
                for (int64_t i = 0; i < effectiveBDs; ++i)
                  bdBlocks.push_back(addBlock());
                // Only create endBlock for non-fused or first-in-group
                // members.  See orphan aie.end comment in existing-region
                // path above.
                mlir::Block *endBlock = nullptr;
                if (!isFusedNonFirst)
                  endBlock = addBlock();

                if (!isFusedNonFirst) {
                  builder.setInsertionPointToEnd(dmaStartBlock);
                  builder.create<AIE::DMAStartOp>(
                      state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
                      mm2sChannel, dmaRepeatCount, bdBlocks[0], endBlock);
                }

                for (int64_t i = 0; i < effectiveBDs; ++i) {
                  // Null locks for disable_synchronization — emitBDBlock skips.
                  mlir::Value blockAcq =
                      mm2sAcqLock
                          ? (isAIE2 ? mm2sAcqLock.getResult()
                                    : (prodAIE1Locks && !prodAIE1Locks->empty()
                                           ? (*prodAIE1Locks)[i % prodAIE1Locks
                                                                      ->size()]
                                                 .getResult()
                                           : mm2sAcqLock.getResult()))
                          : mlir::Value{};
                  mlir::Value blockRel =
                      mm2sRelLock
                          ? (isAIE2 ? mm2sRelLock.getResult() : blockAcq)
                          : mlir::Value{};
                  state.emitBDBlock(
                      state.deviceOp.getLoc(), bdBlocks[i], blockAcq,
                      state.lockAcqValue(Port::Consume, 1),
                      prodBuffers[(i / bdRepeat) % prodBuffers.size()]
                          .getResult(),
                      0, perBufLen, blockRel, state.lockRelValue(Port::Consume),
                      info.producerDimensions, pktID);
                  // When dma_repeat > 0, loop back to the first BD so the
                  // DMA engine can replay the chain via repeat_count.
                  bool isLast = (i == effectiveBDs - 1) && (info.dmaRepeat > 0);
                  if (isLast)
                    builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                                  bdBlocks[0]);
                  else
                    builder.create<AIE::NextBDOp>(
                        state.deviceOp.getLoc(),
                        bdBlocks[(i + 1) % effectiveBDs]);
                }
                if (endBlock) {
                  builder.setInsertionPointToEnd(endBlock);
                  builder.create<AIE::EndOp>(state.deviceOp.getLoc());
                }

                if (!info.fuseGroup.empty())
                  state.conduitBDRange[name] = {bdBlocks.front(),
                                                bdBlocks.back()};
              }
            }
          }
        }
      }
    }

    // Consumer S2MM.
    bool isProducerToShim =
        info.consumerTileCoords.empty() && !info.shimConsumerTileCoords.empty();

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

      int64_t perBufLen = deriveBdLength(info.elemType, info.numElems);

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
      // Case B is always a compute tile — never set finite repeat_count.
      // Compute tile DMAs must cycle infinitely; the core controls lifetime.
      int32_t caseBDmaRepeatCount = 0;
      int64_t caseBBdRepeat = info.bdRepeat > 1 ? info.bdRepeat : 1;
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
          if (!checkBDChainCap(state, prodTileVal, caseBEffectiveBDs, name))
            return;
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
              state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S, caseBMM2SCh,
              caseBDmaRepeatCount, bdBlocks[0], newEndBlock);

          for (int64_t i = 0; i < caseBEffectiveBDs; ++i) {
            // Null locks for disable_synchronization — emitBDBlock skips them.
            mlir::Value blockAcqVal =
                info.consLock
                    ? (isAIE2 ? info.consLock.getResult()
                              : (info.aie1Locks.empty()
                                     ? info.consLock.getResult()
                                     : info.aie1Locks[i % info.aie1Locks.size()]
                                           .getResult()))
                    : mlir::Value{};
            mlir::Value blockRelVal =
                info.prodLock
                    ? (isAIE2 ? info.prodLock.getResult()
                              : (info.aie1Locks.empty()
                                     ? info.prodLock.getResult()
                                     : info.aie1Locks[i % info.aie1Locks.size()]
                                           .getResult()))
                    : mlir::Value{};
            // Case B: compute MM2S to shim — no BDDimLayout on this BD
            // (shim-side descriptor is runtime-programmed, not emitted here).
            state.emitBDBlock(
                state.deviceOp.getLoc(), bdBlocks[i], blockAcqVal,
                state.lockAcqValue(Port::Consume, 1),
                info.buffers[(i / caseBBdRepeat) % info.buffers.size()]
                    .getResult(),
                0, perBufLen, blockRelVal, state.lockRelValue(Port::Consume));
            // Case B is always a compute tile — chain must be circular
            // (infinite cycling with repeat_count=0).
            builder.create<AIE::NextBDOp>(
                state.deviceOp.getLoc(), bdBlocks[(i + 1) % caseBEffectiveBDs]);
          }
          builder.setInsertionPointToEnd(newEndBlock);
          builder.create<AIE::EndOp>(state.deviceOp.getLoc());
        }
      } else {
        builder.setInsertionPoint(state.deviceBody->getTerminator());
        auto memOp =
            builder.create<AIE::MemOp>(state.deviceOp.getLoc(), prodTileVal);
        mlir::Region &memRegion = memOp.getBody();
        tileToDMARegion[prodTileVal] = &memRegion;
        if (!checkBDChainCap(state, prodTileVal, caseBEffectiveBDs, name))
          return;
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
        builder.create<AIE::DMAStartOp>(
            state.deviceOp.getLoc(), AIE::DMAChannelDir::MM2S, caseBMM2SCh2,
            caseBDmaRepeatCount, bdBlocks[0], endMemBlock);

        for (int64_t i = 0; i < caseBEffectiveBDs; ++i) {
          // Null locks for disable_synchronization — emitBDBlock skips them.
          mlir::Value blockAcqVal =
              info.consLock
                  ? (isAIE2 ? info.consLock.getResult()
                            : (info.aie1Locks.empty()
                                   ? info.consLock.getResult()
                                   : info.aie1Locks[i % info.aie1Locks.size()]
                                         .getResult()))
                  : mlir::Value{};
          mlir::Value blockRelVal =
              info.prodLock
                  ? (isAIE2 ? info.prodLock.getResult()
                            : (info.aie1Locks.empty()
                                   ? info.prodLock.getResult()
                                   : info.aie1Locks[i % info.aie1Locks.size()]
                                         .getResult()))
                  : mlir::Value{};
          // Case B: compute MM2S to shim — no BDDimLayout.
          state.emitBDBlock(
              state.deviceOp.getLoc(), bdBlocks[i], blockAcqVal,
              state.lockAcqValue(Port::Consume, 1),
              info.buffers[(i / caseBBdRepeat) % info.buffers.size()]
                  .getResult(),
              0, perBufLen, blockRelVal, state.lockRelValue(Port::Consume));
          // Case B is always a compute tile — chain must be circular
          // (infinite cycling with repeat_count=0).
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

      int64_t perBufLen = deriveBdLength(info.elemType, info.numElems);
      // nConsumerBuffers() >= depth; extra slots support sliding-window
      // patterns.
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
          if (aie1It != info.consumerTileAIE1Locks.end() &&
              !aie1It->second.empty())
            tileAIE1Locks = &aie1It->second;
        }

        // For disable_synchronization, locks are null by design — still emit
        // BDs.
        if ((!tileProdLock || !tileConsLock) && !info.noLocks)
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
            auto memOp = builder.create<AIE::MemOp>(state.deviceOp.getLoc(),
                                                    consTileVal2);
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

        // Detect packet-muxed S2MM channels: when multiple conduits share the
        // same S2MM port on a consumer tile (via pktTileS2MMChannel), only the
        // first conduit emits a dma_start.  Subsequent conduits append BD
        // blocks and the post-pass links all BD chains into a single combined
        // ring.
        bool isS2MMPktNonFirst = false;
        std::string s2mmPktGroupLabel;
        {
          auto pktIt = state.pktTileS2MMChannel.find(consTileVal2);
          if (pktIt != state.pktTileS2MMChannel.end() &&
              pktIt->second == s2mmChannel) {
            auto &chanMap = s2mmPktStarted[consTileVal2];
            auto grpIt = chanMap.find(s2mmChannel);
            if (grpIt != chanMap.end()) {
              isS2MMPktNonFirst = true;
              s2mmPktGroupLabel = grpIt->second;
            } else {
              s2mmPktGroupLabel = state.qualifyFuseGroup(
                  "pkt_s2mm__" + std::to_string(consCol) + "_" +
                      std::to_string(consRow) + "_ch" +
                      std::to_string(s2mmChannel),
                  info.deviceIndex);
              chanMap[s2mmChannel] = s2mmPktGroupLabel;
            }
          }
        }

        // Detect S2MM fuse-group channels (circuit-mode fusion annotated by
        // --conduit-fuse-channels).  Symmetric to packet-muxed detection:
        // first conduit in group emits dma_start, subsequent conduits skip it.
        bool isS2MMFuseNonFirst = false;
        if (!info.fuseGroupS2MM.empty()) {
          auto &chanMap = s2mmFuseStarted[consTileVal2];
          if (chanMap.count(info.fuseGroupS2MM)) {
            isS2MMFuseNonFirst = true;
          } else {
            chanMap[info.fuseGroupS2MM] = true;
          }
        }

        // Compute DMAStartOp repeat_count from dma_repeat.
        // AIE compute tile DMAs must cycle infinitely (repeat_count=0)
        // — the core controls lifetime.  Only MemTile DMAs use finite
        // repeat_count from dma_repeat.
        int32_t dmaRepeatCount = (info.dmaRepeat > 0 && consIsMemTile)
                                     ? static_cast<int32_t>(info.dmaRepeat - 1)
                                     : 0;

        if (!checkBDChainCap(state, consTileVal2, nBufs, name))
          return;
        llvm::SmallVector<mlir::Block *> bdBlocks;
        for (int64_t i = 0; i < nBufs; ++i)
          bdBlocks.push_back(addMemBlock());

        // For packet-muxed non-first conduits: skip dma_start and terminal
        // blocks.  The first conduit's dma_start and end block serve the entire
        // combined ring.  BD chain circularity within each member is preserved
        // (last BD → first BD); the post-pass replaces these to form the
        // combined ring: Q0→Q1→K0→K1→V0→V1→Q0.
        bool isLinearChain = false;
        mlir::Block *bdTermBlock = nullptr;
        mlir::Block *endMemBlock = nullptr;

        if (isS2MMPktNonFirst || isS2MMFuseNonFirst) {
          // Non-first fused conduit: no dma_start, no terminal blocks.
          // BD blocks were already created above.
        } else {
          // Linear chain condition: either dma_repeat>0 (finite DMA task
          // queue), or putCount>1 with no dmaRepeat (N sequential puts merged
          // by
          // --conduit-fuse-channels; annotation-free temporal multiplexing).
          isLinearChain = (info.dmaRepeat > 0) ||
                          (info.putCount > 1 && info.dmaRepeat == 0);
          // bdTermBlock: create a dedicated terminal block for putCount>1
          // linear chains (dmaRepeat==0).  In this case the last BD's
          // next_bd targets bdTermBlock (with permanent aie.end) instead of
          // endMemBlock.  Without this, endMemBlock's aie.end can be replaced
          // by a subsequent channel's DMAStartOp, corrupting the BD chain.
          // For dmaRepeat>0 chains, the last BD loops back to bdBlocks[0]
          // (repeat_count controls termination), so no bdTermBlock is needed.
          bdTermBlock = (info.putCount > 1 && info.dmaRepeat == 0)
                            ? addMemBlock()
                            : nullptr;
          endMemBlock = addMemBlock();

          if (existingEndBlock) {
            mlir::Operation *oldEnd = existingEndBlock->getTerminator();
            builder.setInsertionPointToEnd(existingEndBlock);
            oldEnd->erase();
            builder.create<AIE::DMAStartOp>(
                state.deviceOp.getLoc(), AIE::DMAChannelDir::S2MM, s2mmChannel,
                dmaRepeatCount, bdBlocks[0], endMemBlock);
          } else {
            mlir::Block *dmaStartBlock = addMemBlock();
            dmaStartBlock->moveBefore(&memRegion.front());
            builder.setInsertionPointToEnd(dmaStartBlock);
            builder.create<AIE::DMAStartOp>(
                state.deviceOp.getLoc(), AIE::DMAChannelDir::S2MM, s2mmChannel,
                dmaRepeatCount, bdBlocks[0], endMemBlock);
          }
        }

        // Pick consumer BDDimLayout for this consumer index.
        AIE::BDDimLayoutArrayAttr consDims;
        if (!info.consumerDimensions.empty())
          consDims =
              info.consumerDimensions[consIdx % info.consumerDimensions.size()];

        for (int64_t i = 0; i < nBufs; ++i) {
          // Null locks for disable_synchronization — emitBDBlock skips them.
          mlir::Value blockLockAcq =
              tileProdLock
                  ? (isAIE2 ? tileProdLock.getResult()
                            : (tileAIE1Locks->empty()
                                   ? tileProdLock.getResult()
                                   : (*tileAIE1Locks)[i % tileAIE1Locks->size()]
                                         .getResult()))
                  : mlir::Value{};
          mlir::Value blockLockRel =
              tileConsLock
                  ? (isAIE2 ? tileConsLock.getResult()
                            : (tileAIE1Locks->empty()
                                   ? tileConsLock.getResult()
                                   : (*tileAIE1Locks)[i % tileAIE1Locks->size()]
                                         .getResult()))
                  : mlir::Value{};
          state.emitBDBlock(state.deviceOp.getLoc(), bdBlocks[i], blockLockAcq,
                            state.lockAcqValue(Port::Produce, 1),
                            (*tileBuffers)[i % tileBuffers->size()].getResult(),
                            0, perBufLen, blockLockRel,
                            state.lockRelValue(Port::Produce), consDims);
          // Linear chain handling: dma_repeat > 0 loops back to first BD
          // (repeat_count replays); putCount > 1 terminates at bdTermBlock
          // or endMemBlock.
          bool isLast = (i == nBufs - 1) && isLinearChain;
          if (isLast && info.dmaRepeat > 0)
            builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(), bdBlocks[0]);
          else if (isLast)
            builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                          bdTermBlock ? bdTermBlock
                                                      : endMemBlock);
          else
            builder.create<AIE::NextBDOp>(state.deviceOp.getLoc(),
                                          bdBlocks[(i + 1) % nBufs]);
        }
        if (bdTermBlock) {
          builder.setInsertionPointToEnd(bdTermBlock);
          builder.create<AIE::EndOp>(state.deviceOp.getLoc());
        }
        if (endMemBlock) {
          builder.setInsertionPointToEnd(endMemBlock);
          builder.create<AIE::EndOp>(state.deviceOp.getLoc());
        }

        // Record BD range for packet-muxed S2MM chain fusion.
        if (!s2mmPktGroupLabel.empty()) {
          std::string bdKey = name + "__s2mm_" + std::to_string(consIdx);
          state.conduitBDRange[bdKey] = {bdBlocks.front(), bdBlocks.back()};
          state.fuseGroupMembers[s2mmPktGroupLabel].push_back(bdKey);
        }

        // Record BD range for circuit-mode S2MM fuse-group chain fusion.
        if (!info.fuseGroupS2MM.empty()) {
          std::string bdKey = name + "__s2mm_" + std::to_string(consIdx);
          std::string qS2MM =
              state.qualifyFuseGroup(info.fuseGroupS2MM, info.deviceIndex);
          state.conduitBDRange[bdKey] = {bdBlocks.front(), bdBlocks.back()};
          state.fuseGroupMembers[qS2MM].push_back(bdKey);
        }
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
