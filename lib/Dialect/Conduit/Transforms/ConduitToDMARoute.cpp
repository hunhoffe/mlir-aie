//===- ConduitToDMARoute.cpp - Phase 4-4.5a: flow + shim DMA ----*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Phase 4: Handle shim-tile endpoints → aie.shim_dma_allocation + aie.flow.
//   4a: Producer is shim (row==0): shim sends MM2S to compute consumer.
//   4b: Consumer is shim (row==0): compute producer sends MM2S to shim S2MM.
//
// Phase 4.5: Rewrite aiex.npu.dma_wait / dma_memcpy_nd symbol references.
//
// Phase 4.5a: Emit aie.flow for non-adjacent conduits (compute→compute,
//             compute→MemTile, MemTile→compute).
//   Step 3.5 (mode=any fallback): When a conduit uses routing_mode="any" and
//   all circuit DMA MM2S channels on the producer tile are allocated, attempt
//   packet-switched DMA fallback (tryPacketFallback).
//
//===----------------------------------------------------------------------===//

#include "ConduitToDMACommon.h"

#include <tuple>

namespace xilinx::conduit {

// ---------------------------------------------------------------------------
// Step 3.5: tryPacketFallback
//
// Called from Phase 4.5a when a mode=any conduit finds no free circuit-mode
// MM2S channels on the producer tile.  Attempts to use packet-switched DMA
// as a fallback.  Returns true if a packet flow was successfully emitted,
// false if packet mode is also ineligible (caller should proceed to Step 4).
//
// Steps (from the expert panel spec):
//   3.5a: Check global packet ID budget.
//   3.5b: Check BD and lock budget on producer tile P and consumer tile D.
//   3.5c: Find or designate a packet-mode MM2S channel on P.
//   3.5d: Convergence hazard check against portOccupancy.
//   3.5e: Allocate flow ID, emit aie.packet_flow, record occupancy.
//
// On success: writes the allocated MM2S and S2MM channels to conduitMM2SChannel
// and conduitConsS2MMChannel, and updates pktChannelState.
// On failure: returns false; caller emits Step 4 error or skips.
// ---------------------------------------------------------------------------
static bool tryPacketFallback(ConduitToDMAState &state,
                              const std::string &conduitName, ConduitInfo &info,
                              mlir::Value prodTileVal, int64_t prodCol,
                              int64_t prodRow, mlir::Value consTileVal,
                              int64_t consCol, int64_t consRow,
                              unsigned consIdx) {
  mlir::Operation *prodTileOp = prodTileVal.getDefiningOp();
  mlir::Operation *consTileOp = consTileVal.getDefiningOp();

  // -----------------------------------------------------------------------
  // Step 3.5a: Check per-MemTile packet ID budget.
  // -----------------------------------------------------------------------
  mlir::Value pktDomain = state.getMemTileDomain(prodTileVal);
  if (!state.packetIDAllocator ||
      state.packetIDAllocator->remaining(pktDomain) == 0) {
    // No packet IDs left in this MemTile domain; Step 4 (error) will handle.
    return false;
  }

  // -----------------------------------------------------------------------
  // Step 3.5b: Check BD and lock budget on P and D.
  //
  // Each new packet flow (logical channel) still requires:
  //   - depth BD slots on the producer tile (MM2S BD chain)
  //   - depth BD slots on the consumer tile (S2MM BD chain)
  //   - 2 lock slots on the producer tile (prod_lock + cons_lock)
  //   - 2 lock slots on the consumer tile (prod_lock + cons_lock)
  //
  // BD budget: queried from AIETargetModel; usage tracked in tileBDUsed.
  // Lock budget: queried from AIETargetModel; usage tracked in lockIdCounter.
  // -----------------------------------------------------------------------
  int64_t depth = info.depth > 0 ? info.depth : 1;

  if (state.targetModel) {
    uint32_t prodBDTotal = state.targetModel->getNumBDs(
        static_cast<int>(prodCol), static_cast<int>(prodRow));
    uint32_t consBDTotal = state.targetModel->getNumBDs(
        static_cast<int>(consCol), static_cast<int>(consRow));
    uint32_t prodLockTotal = state.targetModel->getNumLocks(
        static_cast<int>(prodCol), static_cast<int>(prodRow));
    uint32_t consLockTotal = state.targetModel->getNumLocks(
        static_cast<int>(consCol), static_cast<int>(consRow));

    int32_t prodBDUsed =
        state.tileBDUsed.count(prodTileVal) ? state.tileBDUsed[prodTileVal] : 0;
    int32_t consBDUsed =
        state.tileBDUsed.count(consTileVal) ? state.tileBDUsed[consTileVal] : 0;
    int32_t prodLockUsed = state.lockIdCounter.count(prodTileVal)
                               ? state.lockIdCounter[prodTileVal]
                               : 0;
    int32_t consLockUsed = state.lockIdCounter.count(consTileVal)
                               ? state.lockIdCounter[consTileVal]
                               : 0;

    if (static_cast<int32_t>(prodBDTotal) - prodBDUsed < depth ||
        static_cast<int32_t>(consBDTotal) - consBDUsed < depth) {
      // BD budget insufficient; Step 4 error.
      return false;
    }
    if (static_cast<int32_t>(prodLockTotal) - prodLockUsed < 2 ||
        static_cast<int32_t>(consLockTotal) - consLockUsed < 2) {
      // Lock budget insufficient; Step 4 error.
      return false;
    }
  }

  // -----------------------------------------------------------------------
  // Step 3.5c: Find or designate a packet-mode MM2S channel on P.
  //
  // Priority order:
  //   1. An existing packet-mode channel with BD slots remaining.
  //   2. A free (unused) MM2S channel that can be designated as packet-mode.
  //   3. Failure (all MM2S channels on P are circuit-mode and full).
  // -----------------------------------------------------------------------
  int32_t mm2sChannel = -1;

  // Determine how many MM2S channels the producer tile has.
  uint32_t maxMM2S = 2; // hardware default: 2 MM2S channels per compute tile
  if (state.targetModel)
    maxMM2S = state.targetModel->getNumSourceSwitchboxConnections(
        static_cast<int>(prodCol), static_cast<int>(prodRow),
        AIE::WireBundle::DMA);

  // Search for an existing packet-mode channel.
  for (uint32_t ch = 0; ch < maxMM2S; ++ch) {
    auto key = std::make_pair(prodTileOp, static_cast<int>(ch));
    if (state.pktChannelState.isPacketChannel.count(key) &&
        state.pktChannelState.isPacketChannel[key]) {
      // This channel is packet-mode; check BD availability.
      // (Conservative: we check the global BD budget above; here just pick
      // the first packet-mode channel available on this tile.)
      mm2sChannel = static_cast<int32_t>(ch);
      break;
    }
  }

  if (mm2sChannel < 0) {
    // No existing packet-mode channel; try to designate a free one.
    int32_t nextCh = state.tileNextMM2SChannel.count(prodTileVal)
                         ? state.tileNextMM2SChannel[prodTileVal]
                         : 0;
    if (static_cast<uint32_t>(nextCh) < maxMM2S) {
      mm2sChannel = nextCh;
      state.tileNextMM2SChannel[prodTileVal] = nextCh + 1;
      auto key = std::make_pair(prodTileOp, static_cast<int>(mm2sChannel));
      state.pktChannelState.isPacketChannel[key] = true;
    } else {
      // All MM2S channels on P are allocated (circuit-mode); cannot fall back.
      return false;
    }
  }

  // -----------------------------------------------------------------------
  // Step 3.5d: Convergence hazard check.
  //
  // For each existing packet flow on the same MM2S channel (same output port),
  // check whether any flow routes to the same destination tile D with a
  // different flow ID.  If so, emit a warning (not an error) — the user may
  // not require strict ordering between these flows.
  //
  // Implementation note: we use the MM2S channel on P as a proxy for the
  // "switchbox output port" — all flows on the same physical MM2S channel
  // share the same output port from P's switchbox.  A full pathfinder
  // traversal of intermediate switchboxes is deferred to P2-E.
  // -----------------------------------------------------------------------
  int64_t pKey = PacketChannelState::portKey(prodTileOp, mm2sChannel);
  auto &occupancy = state.pktChannelState.portOccupancy[pKey];
  for (auto &[existFlowID, existDstOp] : occupancy) {
    if (existDstOp == consTileOp) {
      // Two packet flows with different IDs routing to the same consumer
      // through the same output port → ordering hazard.
      state.deviceOp.emitWarning(
          llvm::Twine("conduit-to-dma: packet DMA ordering hazard: conduit '") +
          conduitName +
          "' and an existing flow both route to the same consumer tile "
          "through the same MM2S channel; ordering between them is not "
          "guaranteed under sustained load");
    }
  }

  // -----------------------------------------------------------------------
  // Step 3.5e: Allocate packet flow ID and emit aie.packet_flow.
  // -----------------------------------------------------------------------
  std::optional<uint8_t> pktID = state.packetIDAllocator->allocate(pktDomain);
  if (!pktID) {
    // Allocator emitted the error; signal failure.
    state.passFailed = true;
    return false;
  }

  // Assign S2MM channel on the consumer tile (with bounds check).
  // Only share S2MM ports between packet channels that have the same
  // dma_channel_group.  Independent packet channels (no group) get
  // separate S2MM ports to prevent data crossover.
  // Effective group key: dma_channel_group_s2mm if set, else dma_channel_group.
  std::string s2mmGroupKey =
      !info.fuseGroupS2MM.empty() ? info.fuseGroupS2MM : info.fuseGroup;
  int32_t s2mmChannel;
  bool s2mmShared = false;
  if (!s2mmGroupKey.empty()) {
    auto it = state.fuseGroupS2MMChannel.find(s2mmGroupKey);
    if (it != state.fuseGroupS2MMChannel.end()) {
      s2mmChannel = it->second;
      s2mmShared = true;
    }
  }
  if (!s2mmShared) {
    uint32_t maxS2MM_pkt = 2;
    if (state.targetModel)
      maxS2MM_pkt = state.targetModel->getNumDestSwitchboxConnections(
          static_cast<int>(consCol), static_cast<int>(consRow),
          AIE::WireBundle::DMA);
    int32_t nextS2MM_pkt = state.tileNextS2MMChannel.count(consTileVal)
                               ? state.tileNextS2MMChannel[consTileVal]
                               : 0;
    if (static_cast<uint32_t>(nextS2MM_pkt) >= maxS2MM_pkt) {
      // S2MM channels exhausted on consumer tile; packet fallback ineligible.
      return false;
    }
    s2mmChannel = state.tileNextS2MMChannel[consTileVal]++;
    if (!s2mmGroupKey.empty())
      state.fuseGroupS2MMChannel[s2mmGroupKey] = s2mmChannel;
  }
  state.conduitConsS2MMChannel[{conduitName, consIdx}] = s2mmChannel;

  // Lock sharing: only share locks when channels share an S2MM port
  // (same dma_channel_group).  Independent channels use separate locks.
  if (s2mmShared) {
    auto lockIt = state.pktTileS2MMLock.find(consTileVal);
    if (lockIt != state.pktTileS2MMLock.end()) {
      info.consumerTileLocks[consTileVal] = {
          lockIt->second.first.getDefiningOp<AIE::LockOp>(),
          lockIt->second.second.getDefiningOp<AIE::LockOp>()};
    }
  } else if (!s2mmGroupKey.empty()) {
    // First in group: record locks for future group members.
    auto &locks = info.consumerTileLocks[consTileVal];
    if (locks.first && locks.second) {
      state.pktTileS2MMLock[consTileVal] = {locks.first.getResult(),
                                             locks.second.getResult()};
    }
  }

  // Record occupancy for future convergence checks.
  occupancy.push_back({*pktID, consTileOp});

  // Update BD usage budgets (each packet flow reserves depth BDs on each side).
  state.tileBDUsed[prodTileVal] += static_cast<int32_t>(depth);
  state.tileBDUsed[consTileVal] += static_cast<int32_t>(depth);

  // Record the MM2S channel assignment for Phase 5.5 (BD chain generation).
  state.conduitMM2SChannel[conduitName] = mm2sChannel;
  // Record the packet ID for Phase 5.5 BD chain aie.dma_bd_packet headers.
  state.conduitPacketID[conduitName] = *pktID;

  // Emit the packet flow directly using the pre-allocated ID.
  // We do NOT call state.emitFlow("packet", ...) here because emitFlow would
  // call packetIDAllocator->allocate() a second time (double allocation).
  // Instead, build the aie.PacketFlowOp directly with the ID we already hold.
  mlir::OpBuilder &builder = *state.builder;
  auto pktFlow = builder.create<AIE::PacketFlowOp>(
      state.deviceOp.getLoc(), static_cast<int8_t>(*pktID),
      /*keep_pkt_header=*/mlir::BoolAttr{},
      /*priority_route=*/mlir::BoolAttr{});
  mlir::Region &region = pktFlow.getPorts();
  mlir::Block *block = builder.createBlock(&region);
  builder.setInsertionPointToStart(block);
  builder.create<AIE::PacketSourceOp>(state.deviceOp.getLoc(), prodTileVal,
                                      AIE::WireBundle::DMA,
                                      static_cast<int32_t>(mm2sChannel));
  builder.create<AIE::PacketDestOp>(state.deviceOp.getLoc(), consTileVal,
                                    AIE::WireBundle::DMA,
                                    static_cast<int32_t>(s2mmChannel));
  builder.create<AIE::EndOp>(state.deviceOp.getLoc());
  builder.setInsertionPointAfter(pktFlow);

  // Update routingMode so that Phase 5.5 generates packet BD chains.
  info.routingMode = "packet";

  return true;
}

void routePhase(ConduitToDMAState &state) {
  if (!state.deviceOp)
    return;

  mlir::OpBuilder &builder = *state.builder;
  mlir::MLIRContext *ctx = state.ctx;
  const bool isAIE2 = state.isAIE2Plus();

  // -----------------------------------------------------------------------
  // Phase 4: Shim endpoints.
  // -----------------------------------------------------------------------

  for (auto &[name, info] : state.conduitMap) {
    // Multi-device: ensure tile lookups target the correct device.
    if (state.isMultiDevice())
      state.switchToDeviceIndex(info.deviceIndex);

    // Cascade conduits: no shim DMA, no aie.flow — handled below.
    if (info.routingMode == "cascade")
      continue;

    auto [prodCol, prodRow] = info.producerTileCoord;
    if (prodCol < 0)
      continue;

    // --- Sub-case 4a: producer is a shim tile (row==0) ---
    // Broadcast: emit one aie.flow per consumer tile.
    if (prodRow == 0) {
      if (info.consumerTileCoords.empty())
        continue;

      AIE::TileOp shimTile = state.lookupTileByCoord(prodCol, prodRow);
      if (!shimTile)
        continue;

      // Check MM2S budget on this shim tile; if exhausted, spill to
      // the next available shim tile in the same device.
      bool shimSpilled = false;
      {
        uint32_t maxShimMM2S = 2;
        if (state.targetModel)
          maxShimMM2S = state.targetModel->getNumSourceShimMuxConnections(
              static_cast<int>(prodCol), static_cast<int>(prodRow),
              AIE::WireBundle::DMA);
        int32_t currentMM2S =
            state.tileNextMM2SChannel.count(shimTile.getResult())
                ? state.tileNextMM2SChannel[shimTile.getResult()]
                : 0;
        if (static_cast<uint32_t>(currentMM2S) >= maxShimMM2S &&
            info.routingMode != "packet") {
          bool found = false;
          for (int64_t adjCol = prodCol + 1; adjCol < prodCol + 8; ++adjCol) {
            AIE::TileOp adjTile = state.lookupTileByCoord(adjCol, 0);
            if (!adjTile) {
              // Create the adjacent shim tile op.
              mlir::OpBuilder::InsertionGuard guard(builder);
              if (state.insertAfterTile)
                builder.setInsertionPointAfter(state.insertAfterTile);
              else
                builder.setInsertionPointToStart(state.deviceBody);
              adjTile = builder.create<AIE::TileOp>(
                  state.deviceOp.getLoc(), static_cast<int>(adjCol), 0);
              state.tileCache[{adjCol, 0}] = adjTile;
              if (state.activeDevIdx >= 0 &&
                  state.activeDevIdx <
                      static_cast<int>(state.perDevTileCache.size()))
                state.perDevTileCache[state.activeDevIdx][{adjCol, 0}] =
                    adjTile;
              state.insertAfterTile = adjTile.getOperation();
            }
            int32_t adjMM2S =
                state.tileNextMM2SChannel.count(adjTile.getResult())
                    ? state.tileNextMM2SChannel[adjTile.getResult()]
                    : 0;
            uint32_t adjMax = 2;
            if (state.targetModel)
              adjMax = state.targetModel->getNumSourceShimMuxConnections(
                  static_cast<int>(adjCol), 0, AIE::WireBundle::DMA);
            if (static_cast<uint32_t>(adjMM2S) < adjMax) {
              shimTile = adjTile;
              shimSpilled = true;
              found = true;
              break;
            }
          }
          if (!found) {
            state.deviceOp.emitError(
                llvm::Twine("conduit-to-dma: MM2S DMA channel exhausted on "
                            "shim tile (") +
                llvm::Twine(prodCol) +
                ",0) and no adjacent shim tile has "
                "available MM2S channels");
            state.passFailed = true;
            return;
          }
        }
      }

      builder.setInsertionPoint(state.deviceBody->getTerminator());

      // Shim-side prod/cons locks (AIE2 only).
      // For external-buffer conduits (AIE1), also emit a single shim lock
      // for the aie.shim_dma BD chain.
      // Note: shim-producer conduits (Phase 4a) cannot be link destinations
      // (link dsts are always MemTile/compute tile consumers, not shim
      // producers), so no linkDstNames guard is needed in 4a.
      // Skip lock allocation when disable_synchronization is set: the oracle
      // emits no locks or use_lock for these conduits.
      //
      // Skip shim lock allocation for distribute link sources (shim→MemTile).
      // For these conduits the shim DMA is fire-and-forget (managed by host
      // runtime via aiex.npu.dma_memcpy_nd). The MemTile S2MM synchronization
      // uses per-destination locks allocated on the MemTile by linkPhase
      // (sliceProdLocks/sliceConsLocks). Allocating locks on the shim tile
      // produces dead resources with wrong init values and wrong tile
      // placement.
      bool isDistributeLinkSrc = state.linkSrcNamesEarly.count(name) > 0;
      if (isAIE2 && !info.noLocks && !isDistributeLinkSrc) {
        {
          int lockIdx = state.lockIdCounter[shimTile.getResult()]++;
          std::string symName = name + "_prod_lock_0";
          // prod_lock init=0: shim locks are programmed by the host runtime
          // via aiex.npu.dma_memcpy_nd token signaling; pre-signaling free
          // slots causes over-commitment before the shim DMA is configured.
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), shimTile.getResult(), lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          info.shimProdLock = lk;
        }
        {
          int lockIdx = state.lockIdCounter[shimTile.getResult()]++;
          std::string symName = name + "_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), shimTile.getResult(), lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          info.shimConsLock = lk;
        }
      }
      // For external-buffer conduits on AIE1: allocate a shim lock for the
      // aie.shim_dma BD chain (acquire before DMA, release after).
      if (!isAIE2 && !info.externalBuffers.empty() &&
          !info.noLocks) {
        int lockIdx = state.lockIdCounter[shimTile.getResult()]++;
        std::string symName = name + "_lock_0";
        AIE::LockOp lk = builder.create<AIE::LockOp>(
            state.deviceOp.getLoc(), shimTile.getResult(), lockIdx,
            static_cast<int>(0));
        lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        info.shimProdLock = lk;
        info.shimConsLock = lk;
      }

      // aie.shim_dma_allocation: assign next available MM2S channel on this
      // shim tile.  Multiple shim-producer conduits on the same shim tile
      // must each use a distinct MM2S channel (0, 1, ...).
      int32_t shimMM2SCh = state.tileNextMM2SChannel[shimTile.getResult()]++;
      state.conduitMM2SChannel[name] = shimMM2SCh;

      std::string allocSym = info.origName + "_shim_alloc";
      state.shimConduitNames.insert(info.origName);
      // If the shim tile was spilled to an adjacent tile, the pre-existing
      // ShimDMAAllocationOp (from ObjectFifo/DMATask lowering) references the
      // wrong tile and channel. Erase it so a fresh one is created below.
      if (shimSpilled) {
        if (auto *existingAlloc = mlir::SymbolTable::lookupSymbolIn(
                state.deviceOp, mlir::StringAttr::get(ctx, allocSym)))
          existingAlloc->erase();
      }
      if (!mlir::SymbolTable::lookupSymbolIn(
              state.deviceOp, mlir::StringAttr::get(ctx, allocSym)))
        builder.create<AIE::ShimDMAAllocationOp>(
            state.deviceOp.getLoc(), allocSym, shimTile.getResult(),
            AIE::DMAChannelDir::MM2S,
            /*channel_index=*/static_cast<int64_t>(shimMM2SCh),
            /*plio=*/info.plio,
            /*packet=*/nullptr);

      // One flow per consumer tile.  The shim-side DMA port uses shimMM2SCh
      // so that each conduit routes through its own hardware MM2S channel.
      // The consumer-side port is allocated from tileNextS2MMChannel.
      for (unsigned consIdx = 0; consIdx < info.consumerTileCoords.size();
           ++consIdx) {
        auto [consCol, consRow] = info.consumerTileCoords[consIdx];
        AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
        if (!consTile)
          continue;
        // S2MM fuse group: reuse existing S2MM channel if another conduit
        // in the same fuse group already allocated one on this tile.
        int32_t s2mmCh;
        if (!info.fuseGroupS2MM.empty()) {
          std::string qS2MM = state.qualifyFuseGroup(info.fuseGroupS2MM,
                                                      info.deviceIndex);
          auto it = state.fuseGroupS2MMChannel.find(qS2MM);
          if (it != state.fuseGroupS2MMChannel.end()) {
            s2mmCh = it->second;
          } else {
            // Bounds-check S2MM channels on the consumer tile.
            uint32_t maxS2MM_4a = 2;
            if (state.targetModel)
              maxS2MM_4a = state.targetModel->getNumDestSwitchboxConnections(
                  static_cast<int>(consCol), static_cast<int>(consRow),
                  AIE::WireBundle::DMA);
            int32_t nextS2MM_4a =
                state.tileNextS2MMChannel.count(consTile.getResult())
                    ? state.tileNextS2MMChannel[consTile.getResult()]
                    : 0;
            if (static_cast<uint32_t>(nextS2MM_4a) >= maxS2MM_4a) {
              state.deviceOp.emitError(
                  llvm::Twine("conduit-to-dma: S2MM DMA channel exhausted on "
                              "tile (") +
                  llvm::Twine(consCol) + "," + llvm::Twine(consRow) +
                  "): all " + llvm::Twine(maxS2MM_4a) + " channels in use");
              state.passFailed = true;
              return;
            }
            s2mmCh = state.tileNextS2MMChannel[consTile.getResult()]++;
            state.fuseGroupS2MMChannel[qS2MM] = s2mmCh;
          }
          std::string bdKey = name + "__s2mm_" + std::to_string(consIdx);
          state.fuseGroupMembers[qS2MM].push_back(bdKey);
        } else {
          // Bounds-check S2MM channels on the consumer tile.
          uint32_t maxS2MM_4a = 2;
          if (state.targetModel)
            maxS2MM_4a = state.targetModel->getNumDestSwitchboxConnections(
                static_cast<int>(consCol), static_cast<int>(consRow),
                AIE::WireBundle::DMA);
          int32_t nextS2MM_4a =
              state.tileNextS2MMChannel.count(consTile.getResult())
                  ? state.tileNextS2MMChannel[consTile.getResult()]
                  : 0;
          if (static_cast<uint32_t>(nextS2MM_4a) >= maxS2MM_4a) {
            state.deviceOp.emitError(
                llvm::Twine("conduit-to-dma: S2MM DMA channel exhausted on "
                            "tile (") +
                llvm::Twine(consCol) + "," + llvm::Twine(consRow) +
                "): all " + llvm::Twine(maxS2MM_4a) + " channels in use");
            state.passFailed = true;
            return;
          }
          s2mmCh = state.tileNextS2MMChannel[consTile.getResult()]++;
        }
        state.conduitConsS2MMChannel[{name, consIdx}] = s2mmCh;
        auto shimBundle =
            info.plio ? AIE::WireBundle::PLIO : AIE::WireBundle::DMA;
        state.emitFlow(info.routingMode, shimTile.getResult(), shimBundle,
                       shimMM2SCh, consTile.getResult(), AIE::WireBundle::DMA,
                       s2mmCh);
      }
    }

    // --- Sub-case 4b: consumer is a shim tile (row==0) ---
    for (unsigned shimConsIdx = 0;
         shimConsIdx < info.shimConsumerTileCoords.size(); ++shimConsIdx) {
      auto [shimCol, shimRow] = info.shimConsumerTileCoords[shimConsIdx];
      if (shimRow != 0)
        continue;

      AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
      if (!prodTile)
        continue;

      AIE::TileOp shimTile = state.lookupTileByCoord(shimCol, shimRow);
      if (!shimTile)
        continue;

      builder.setInsertionPoint(state.deviceBody->getTerminator());

      // Determine indexed naming for multi-consumer conduits (compute + shim).
      bool multiConsumer = (info.consumerTileCoords.size() +
                            info.shimConsumerTileCoords.size()) > 1;
      unsigned globalConsIdx =
          static_cast<unsigned>(info.consumerTileCoords.size()) + shimConsIdx;
      std::string consSuffix =
          multiConsumer ? "_cons_" + std::to_string(globalConsIdx) : "_cons";

      // Shim-side consumer locks (AIE2: prod_lock + cons_lock;
      // AIE1: not needed — shim locks managed differently).
      // These are referenced by the host runtime when programming
      // shim S2MM DMA BDs; without them, the runtime has no locks
      // for flow control on the receive path.
      // Skip when disable_synchronization: oracle emits no locks for these.
      if (isAIE2 && !info.noLocks) {
        {
          int lockIdx = state.lockIdCounter[shimTile.getResult()]++;
          // Naming convention: <conduit>_<suffix>_<lock-role>_<idx>
          // consSuffix = "_cons" (single consumer) or "_cons_N" (multi).
          std::string symName = name + consSuffix + "_prod_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), shimTile.getResult(), lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        }
        {
          int lockIdx = state.lockIdCounter[shimTile.getResult()]++;
          std::string symName = name + consSuffix + "_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), shimTile.getResult(), lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        }
      }

      // Allocate the next available S2MM channel on the shim tile (with
      // bounds check).  Use ShimMux (not Switchbox) for shim tiles:
      // getNumDestSwitchboxConnections(col, 0, DMA) returns 0 on AIE2
      // because WireBundle::DMA is not a switchbox port for shim tiles.
      // The correct API is getNumDestShimMuxConnections which returns 2.
      uint32_t maxS2MM_4b = 2;
      if (state.targetModel)
        maxS2MM_4b = state.targetModel->getNumDestShimMuxConnections(
            static_cast<int>(shimCol), static_cast<int>(shimRow),
            AIE::WireBundle::DMA);
      int32_t nextS2MM_4b =
          state.tileNextS2MMChannel.count(shimTile.getResult())
              ? state.tileNextS2MMChannel[shimTile.getResult()]
              : 0;
      if (static_cast<uint32_t>(nextS2MM_4b) >= maxS2MM_4b) {
        state.deviceOp.emitError(
            llvm::Twine("conduit-to-dma: S2MM DMA channel exhausted on "
                        "shim tile (") +
            llvm::Twine(shimCol) + "," + llvm::Twine(shimRow) + "): all " +
            llvm::Twine(maxS2MM_4b) + " channels in use");
        // B-3 fix: return immediately to stop processing with broken state.
        state.passFailed = true;
        return;
      }
      int32_t shimS2MMCh = state.tileNextS2MMChannel[shimTile.getResult()]++;

      std::string allocSym = info.origName + "_shim_alloc";
      state.shimConduitNames.insert(info.origName);
      // Link-dst conduits: this allocation is intentionally kept —
      // linkPhase() emits the flow (memtile MM2S → shim S2MM) but does NOT
      // create a ShimDMAAllocationOp. routePhase owns the allocation for all
      // conduits with a shim consumer, including link-dst conduits.
      if (!mlir::SymbolTable::lookupSymbolIn(
              state.deviceOp, mlir::StringAttr::get(ctx, allocSym)))
        builder.create<AIE::ShimDMAAllocationOp>(
            state.deviceOp.getLoc(), allocSym, shimTile.getResult(),
            AIE::DMAChannelDir::S2MM,
            /*channel_index=*/static_cast<int64_t>(shimS2MMCh),
            /*plio=*/info.plio,
            /*packet=*/nullptr);

      // Emit the producer→shim flow. For join-destination conduits the
      // producer tile is the MemTile; linkPhase() explicitly skips this
      // flow (see ConduitToDMALink.cpp) and expects routePhase to own it.
      // Allocate the next available MM2S channel on the producer tile
      // (typically a MemTile) so that linkPhase() can look up the assigned
      // channel via conduitMM2SChannel and create a matching DMAStartOp.
      int32_t mm2sChForShimCons =
          state.tileNextMM2SChannel[prodTile.getResult()]++;
      state.conduitMM2SChannel[name] = mm2sChForShimCons;
      auto shimBundle =
          info.plio ? AIE::WireBundle::PLIO : AIE::WireBundle::DMA;
      state.emitFlow(info.routingMode, prodTile.getResult(),
                     AIE::WireBundle::DMA, mm2sChForShimCons,
                     shimTile.getResult(), shimBundle, shimS2MMCh);
    }
  }

  // -----------------------------------------------------------------------
  // Phase 4.5: Rewrite symbol references from @<name> to @<name>_shim_alloc.
  // -----------------------------------------------------------------------
  if (!state.shimConduitNames.empty()) {
    state.module.walk([&](mlir::Operation *op) {
      llvm::StringRef opName = op->getName().getStringRef();

      if (opName == "aiex.npu.dma_wait") {
        if (auto symAttr =
                op->getAttrOfType<mlir::FlatSymbolRefAttr>("symbol")) {
          llvm::StringRef ref = symAttr.getValue();
          if (state.shimConduitNames.count(ref)) {
            op->setAttr("symbol", mlir::FlatSymbolRefAttr::get(
                                      ctx, (ref + "_shim_alloc").str()));
          }
        }
        return;
      }

      if (opName == "aiex.npu.dma_memcpy_nd") {
        if (auto symAttr =
                op->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata")) {
          llvm::StringRef ref = symAttr.getValue();
          if (state.shimConduitNames.count(ref)) {
            op->setAttr("metadata", mlir::FlatSymbolRefAttr::get(
                                        ctx, (ref + "_shim_alloc").str()));
          }
        } else if (auto symAttr =
                       op->getAttrOfType<mlir::SymbolRefAttr>("metadata")) {
          llvm::StringRef ref = symAttr.getRootReference().getValue();
          if (!state.shimConduitNames.count(ref))
            return;
          op->setAttr("metadata", mlir::SymbolRefAttr::get(
                                      mlir::StringAttr::get(
                                          ctx, (ref + "_shim_alloc").str()),
                                      symAttr.getNestedReferences()));
        }
        return;
      }
    });
  }

  // -----------------------------------------------------------------------
  // Phase 4.5a: Emit aie.flow for non-adjacent conduits.
  //
  // Fused channel groups: conduits with the same dma_channel_group
  // label share one hardware MM2S channel slot.
  //
  // Step 3.5 (mode=any fallback): When routing_mode="any" and all MM2S
  // channels on the producer tile are allocated for circuit-switched DMA,
  // tryPacketFallback is invoked.  It attempts packet-switched DMA using an
  // existing or newly designated packet-mode physical channel, subject to
  // BD budget, lock budget, and global packet-flow-ID constraints.
  // -----------------------------------------------------------------------

  // -----------------------------------------------------------------------
  // Pre-pass: count packet channels per MM2S fuse group and pre-allocate
  // power-of-2-aligned packet ID blocks.
  //
  // The downstream AIECreatePathFindFlows pass computes mask/value rules for
  // groups of packet flows sharing a source port.  If the IDs are not a
  // power-of-2-aligned contiguous block, the mask can be overly broad and
  // accidentally match IDs from other groups (e.g., IDs {1,2,3,4,5} produce
  // mask=0b11000 value=0 which matches ALL IDs 0-7).
  //
  // This pre-pass ensures correct mask/value by:
  //   1. Counting packet-mode channels per fuse group.
  //   2. Calling allocateBlock() to reserve an aligned ID block per group.
  //   3. Storing the block start; the main loop draws sequential IDs from it.
  // -----------------------------------------------------------------------
  llvm::StringMap<uint8_t> fuseGroupPacketIDBase;  // qFG → start ID
  llvm::StringMap<unsigned> fuseGroupPacketIDNext;  // qFG → next member index
  {
    // Step 1: count packet channels per qualified fuse group.
    llvm::StringMap<unsigned> fuseGroupPacketCount;
    // Also record one MemTile domain per group for block allocation.
    llvm::StringMap<mlir::Value> fuseGroupDomain;
    for (auto &[name, info] : state.conduitMap) {
      if (info.routingMode != "packet" || info.fuseGroup.empty())
        continue;
      if (state.isMultiDevice())
        state.switchToDeviceIndex(info.deviceIndex);
      std::string qFG =
          state.qualifyFuseGroup(info.fuseGroup, info.deviceIndex);
      fuseGroupPacketCount[qFG]++;
      if (fuseGroupDomain.find(qFG) == fuseGroupDomain.end()) {
        auto [prodCol, prodRow] = info.producerTileCoord;
        if (prodCol >= 0) {
          AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
          if (prodTile)
            fuseGroupDomain[qFG] = state.getMemTileDomain(prodTile.getResult());
        }
      }
    }
    // Step 2: allocate aligned blocks for groups with >1 packet member.
    for (auto &entry : fuseGroupPacketCount) {
      llvm::StringRef qFG = entry.first();
      unsigned count = entry.second;
      if (count <= 1)
        continue;
      auto domIt = fuseGroupDomain.find(qFG);
      if (domIt == fuseGroupDomain.end() || !domIt->second)
        continue;
      auto startID =
          state.packetIDAllocator->allocateBlock(domIt->second, count);
      if (!startID) {
        state.passFailed = true;
        return;
      }
      fuseGroupPacketIDBase[qFG] = *startID;
      fuseGroupPacketIDNext[qFG] = 0;
    }
  }

  // Track emitted flow port pairs to prevent duplicate flows when MM2S
  // fuse group members share the same source→dest ports (e.g. channel_22
  // with routing_mode=packet and channel_18 with routing_mode=any sharing
  // the same MM2S channel and S2MM port via dma_channel_group).
  using FlowKey = std::tuple<void *, int32_t, void *, int32_t>;
  std::set<FlowKey> emittedFlowPorts;

  // Two-pass flow emission: process packet-mode conduits first (pass 0)
  // to record their FlowKeys before non-packet fuse group partners emit
  // circuit flows (pass 1).  Without this, nondeterministic conduitMap
  // iteration order could let a routing_mode=any conduit emit a circuit
  // flow before its packet-mode fuse group partner records the dedup
  // FlowKey, causing both packet_flow AND aie.flow for the same ports.
  for (int flowPass = 0; flowPass < 2; ++flowPass) {
  for (auto &[name, info] : state.conduitMap) {
    // Multi-device: ensure tile lookups target the correct device.
    if (state.isMultiDevice())
      state.switchToDeviceIndex(info.deviceIndex);

    // Pass 0: only packet conduits.  Pass 1: everything else.
    if (flowPass == 0 && info.routingMode != "packet") continue;
    if (flowPass == 1 && info.routingMode == "packet") continue;

    if (info.routingMode == "cascade")
      continue;
    if (info.sharedMemory)
      continue;
    if (state.linkSrcNamesEarly.count(name) ||
        state.linkJoinSrcNames.count(name))
      continue;
    // Link destinations: flows are emitted by linkPhase() — skip here to
    // avoid duplicate flows.
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

    if (!info.consumerTileBuffers.count(prodTileVal))
      continue;

    builder.setInsertionPoint(state.deviceBody->getTerminator());

    // ---- Determine hardware MM2S channel count for this tile. ----
    // Used by the mode=any exhaustion check (Step 3.5).
    uint32_t maxMM2S = 2; // hardware default: 2 MM2S per compute tile
    if (state.targetModel)
      maxMM2S = state.targetModel->getNumSourceSwitchboxConnections(
          static_cast<int>(prodCol), static_cast<int>(prodRow),
          AIE::WireBundle::DMA);

    // ---- Assign MM2S channel (fused groups share a channel). ----
    // Check if Phase 4b already assigned an MM2S channel for this conduit
    // (happens when the conduit has both compute and shim consumers — the
    // shim consumer flow and the compute consumer flow share the same
    // producer-side MM2S channel as a hardware broadcast).
    int32_t mm2sChannel = -1;
    bool usedPacketFallback = false;
    {
      auto existingIt = state.conduitMM2SChannel.find(name);
      if (existingIt != state.conduitMM2SChannel.end()) {
        mm2sChannel = existingIt->second;
      }
    }

    if (mm2sChannel >= 0) {
      // Already assigned by Phase 4b — reuse (broadcast from same MM2S port).
    } else if (!info.fuseGroup.empty()) {
      std::string qFG = state.qualifyFuseGroup(info.fuseGroup,
                                                info.deviceIndex);
      auto it = state.fuseGroupMM2SChannel.find(qFG);
      if (it != state.fuseGroupMM2SChannel.end()) {
        mm2sChannel = it->second;
      } else {
        mm2sChannel = state.tileNextMM2SChannel[prodTileVal]++;
        state.fuseGroupMM2SChannel[qFG] = mm2sChannel;
      }
      state.fuseGroupMembers[qFG].push_back(name);
      state.conduitMM2SChannel[name] = mm2sChannel;
    } else if (info.routingMode == "any") {
      // mode=any: check whether a circuit DMA channel is available.
      int32_t nextCh = state.tileNextMM2SChannel.count(prodTileVal)
                           ? state.tileNextMM2SChannel[prodTileVal]
                           : 0;
      if (static_cast<uint32_t>(nextCh) < maxMM2S) {
        // A free circuit-mode channel exists — use it.
        mm2sChannel = state.tileNextMM2SChannel[prodTileVal]++;
        state.conduitMM2SChannel[name] = mm2sChannel;
      } else {
        // Circuit DMA exhausted; flag that Step 3.5 handles emission below.
        usedPacketFallback = true;
      }
    } else {
      // Check if a circuit DMA channel is available before allocating.
      // If exhausted, fall back to packet-switched DMA for non-cascade/
      // non-shared-memory channels (extends the mode=any fallback).
      int32_t nextCh = state.tileNextMM2SChannel.count(prodTileVal)
                           ? state.tileNextMM2SChannel[prodTileVal]
                           : 0;
      if (static_cast<uint32_t>(nextCh) < maxMM2S) {
        mm2sChannel = state.tileNextMM2SChannel[prodTileVal]++;
        state.conduitMM2SChannel[name] = mm2sChannel;
        // Track packet-mode channel designation for Step 3.5c.
        if (info.routingMode == "packet" && prodTile) {
          auto key = std::make_pair(prodTile.getOperation(),
                                    static_cast<int>(mm2sChannel));
          state.pktChannelState.isPacketChannel[key] = true;
          // NOTE: usedPacketFallback is NOT set here; explicit packet-mode
          // broadcast is handled below (single multi-dest packet flow).
        }
      } else if (info.routingMode != "cascade" &&
                 info.routingMode != "shared_memory" &&
                 info.routingMode != "stream") {
        // Circuit DMA exhausted on producer tile — fall back to
        // packet-switched DMA regardless of explicit routing_mode.
        // tryPacketFallback (Step 3.5c) will reuse an existing
        // packet-designated MM2S channel if one exists.
        usedPacketFallback = true;
      } else {
        // Cascade/shared_memory/stream cannot use packet fallback.
        mm2sChannel = state.tileNextMM2SChannel[prodTileVal]++;
        state.conduitMM2SChannel[name] = mm2sChannel;
      }
    }

    // ---- Explicit packet-mode broadcast: single multi-dest flow. ----
    // For routing_mode="packet", emit one aie.packet_flow with all consumer
    // destinations and a single packet ID.  The switchbox hardware broadcasts
    // each packet to all destinations.  This matches the oracle's behavior
    // (AIEObjectFifoStatefulTransform) where one bdPacket ID is used for all
    // producer MM2S BDs and one packet_flow carries multiple packet_dest ops.
    if (info.routingMode == "packet" && mm2sChannel >= 0 &&
        !info.consumerTileCoords.empty()) {
      if (!state.packetIDAllocator) {
        state.module.emitError(
            "internal error: packetIDAllocator not initialized");
        state.passFailed = true;
        return;
      }
      // Use pre-allocated aligned block ID if this channel belongs to a fuse
      // group with multiple packet members; otherwise fall back to sequential.
      std::optional<uint8_t> pktID;
      std::string qFG;
      if (!info.fuseGroup.empty()) {
        qFG = state.qualifyFuseGroup(info.fuseGroup, info.deviceIndex);
        auto baseIt = fuseGroupPacketIDBase.find(qFG);
        if (baseIt != fuseGroupPacketIDBase.end()) {
          unsigned idx = fuseGroupPacketIDNext[qFG]++;
          pktID = static_cast<uint8_t>(baseIt->second + idx);
        }
      }
      if (!pktID) {
        mlir::Value pktDomain = state.getMemTileDomain(prodTileVal);
        pktID = state.packetIDAllocator->allocate(pktDomain);
      }
      if (!pktID) {
        state.passFailed = true;
        return;
      }
      state.conduitPacketID[name] = *pktID;

      auto pktFlow = builder.create<AIE::PacketFlowOp>(
          state.deviceOp.getLoc(), static_cast<int8_t>(*pktID),
          /*keep_pkt_header=*/mlir::BoolAttr{},
          /*priority_route=*/mlir::BoolAttr{});
      mlir::Region &region = pktFlow.getPorts();
      mlir::Block *pktBlock = builder.createBlock(&region);
      builder.setInsertionPointToStart(pktBlock);
      builder.create<AIE::PacketSourceOp>(state.deviceOp.getLoc(), prodTileVal,
                                          AIE::WireBundle::DMA,
                                          static_cast<int32_t>(mm2sChannel));

      for (unsigned consIdx = 0; consIdx < info.consumerTileCoords.size();
           ++consIdx) {
        auto [consCol, consRow] = info.consumerTileCoords[consIdx];
        if (consRow == 0)
          continue;

        AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
        if (!consTile)
          continue;
        mlir::Value consTileVal = consTile.getResult();

        // Allocate S2MM channel on the consumer tile.
        // Only share S2MM ports between packet channels that have the same
        // dma_channel_group.  Independent packet channels (no group) get
        // separate S2MM ports to prevent data crossover (e.g., Q/K vs V in
        // flash attention — same tile, different data).
        // Effective group key: dma_channel_group_s2mm if set, else
        // dma_channel_group.
        std::string s2mmGrp = state.qualifyFuseGroup(
            !info.fuseGroupS2MM.empty() ? info.fuseGroupS2MM
                                        : info.fuseGroup,
            info.deviceIndex);
        int32_t s2mmChannel;
        bool s2mmShared = false;
        if (!s2mmGrp.empty()) {
          auto it = state.fuseGroupS2MMChannel.find(s2mmGrp);
          if (it != state.fuseGroupS2MMChannel.end()) {
            s2mmChannel = it->second;
            s2mmShared = true;
          }
        }
        if (!s2mmShared) {
          uint32_t maxS2MM_pkt = 2;
          if (state.targetModel)
            maxS2MM_pkt = state.targetModel->getNumDestSwitchboxConnections(
                static_cast<int>(consCol), static_cast<int>(consRow),
                AIE::WireBundle::DMA);
          int32_t nextS2MM_pkt = state.tileNextS2MMChannel.count(consTileVal)
                                     ? state.tileNextS2MMChannel[consTileVal]
                                     : 0;
          if (static_cast<uint32_t>(nextS2MM_pkt) >= maxS2MM_pkt) {
            state.deviceOp.emitError(
                llvm::Twine("conduit-to-dma: S2MM DMA channel exhausted on "
                            "tile (") +
                llvm::Twine(consCol) + "," + llvm::Twine(consRow) +
                "): all " + llvm::Twine(maxS2MM_pkt) + " channels in use");
            state.passFailed = true;
            return;
          }
          s2mmChannel = state.tileNextS2MMChannel[consTileVal]++;
          if (!s2mmGrp.empty())
            state.fuseGroupS2MMChannel[s2mmGrp] = s2mmChannel;
        }
        state.conduitConsS2MMChannel[{name, consIdx}] = s2mmChannel;

        // Lock sharing: only share locks when channels share an S2MM port
        // (same dma_channel_group).  Independent channels use separate locks.
        if (s2mmShared) {
          auto lockIt = state.pktTileS2MMLock.find(consTileVal);
          if (lockIt != state.pktTileS2MMLock.end()) {
            info.consumerTileLocks[consTileVal] = {
                lockIt->second.first.getDefiningOp<AIE::LockOp>(),
                lockIt->second.second.getDefiningOp<AIE::LockOp>()};
          }
        } else if (!s2mmGrp.empty()) {
          // First in group: record locks for future group members.
          auto &locks = info.consumerTileLocks[consTileVal];
          if (locks.first && locks.second) {
            state.pktTileS2MMLock[consTileVal] = {locks.first.getResult(),
                                                   locks.second.getResult()};
          }
        }

        builder.create<AIE::PacketDestOp>(state.deviceOp.getLoc(), consTileVal,
                                          AIE::WireBundle::DMA,
                                          static_cast<int32_t>(s2mmChannel));

        // Record the emitted port pair so that fuse group partners sharing
        // the same MM2S+S2MM ports do not emit a duplicate circuit flow.
        FlowKey fk{prodTileVal.getAsOpaquePointer(), mm2sChannel,
                   consTileVal.getAsOpaquePointer(), s2mmChannel};
        emittedFlowPorts.insert(fk);
      }

      builder.create<AIE::EndOp>(state.deviceOp.getLoc());
      builder.setInsertionPointAfter(pktFlow);
      continue; // skip per-consumer circuit/fallback flow loop
    }

    // ---- Emit flows per consumer. ----
    for (unsigned consIdx = 0; consIdx < info.consumerTileCoords.size();
         ++consIdx) {
      auto [consCol, consRow] = info.consumerTileCoords[consIdx];
      if (consRow == 0)
        continue;

      // For single-consumer conduits, adjacent tiles use shared memory
      // (Phase 3c) — no DMA flow needed.  For broadcast (multi-consumer),
      // Phase 3c is skipped; all consumers use DMA, so flows are needed
      // for every consumer regardless of adjacency.
      // Exception: forceDMA forces DMA even for adjacent tiles.
      // Also: when shim consumers exist, the producer needs DMA MM2S
      // regardless (to reach the shim tile via the switchbox network),
      // so the compute consumer flow must also be emitted.
      if (!info.forceDMA && info.consumerTileCoords.size() == 1 &&
          info.shimConsumerTileCoords.empty()) {
        bool explicitSharedMem = (info.routingMode == "shared_memory");
        bool rightAdj = state.targetModel->isLegalMemAffinity(prodCol, prodRow,
                                                              consCol, consRow);
        bool leftAdj = state.targetModel->isLegalMemAffinity(consCol, consRow,
                                                             prodCol, prodRow);
        if (explicitSharedMem || rightAdj || leftAdj)
          continue;
      }

      AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
      if (!consTile)
        continue;

      mlir::Value consTileVal = consTile.getResult();

      if (usedPacketFallback) {
        // Step 3.5: attempt packet DMA fallback.
        bool ok =
            tryPacketFallback(state, name, info, prodTileVal, prodCol, prodRow,
                              consTileVal, consCol, consRow, consIdx);
        if (!ok) {
          // Step 4: all modes exhausted — emit a hard error.
          state.deviceOp.emitError(
              llvm::Twine("conduit-to-dma: no DMA resources available for "
                          "conduit '") +
              name + "': circuit DMA MM2S channels exhausted on tile (" +
              llvm::Twine(prodCol) + "," + llvm::Twine(prodRow) +
              ") and packet DMA fallback is also ineligible "
              "(check BD budget, lock budget, and packet flow ID budget)");
          // B-3 fix: return immediately so the outer conduit loop does not
          // continue processing subsequent conduits with broken state after
          // both circuit DMA and packet fallback have been exhausted.
          state.passFailed = true;
          return;
        }
        // tryPacketFallback records conduitMM2SChannel and
        // conduitConsS2MMChannel internally; skip the circuit path below.
        continue;
      }

      // S2MM fuse group: reuse existing S2MM channel if another conduit
      // in the same fuse group already allocated one on this tile.
      int32_t s2mmChannel;
      if (!info.fuseGroupS2MM.empty()) {
        std::string qS2MM = state.qualifyFuseGroup(info.fuseGroupS2MM,
                                                    info.deviceIndex);
        auto it = state.fuseGroupS2MMChannel.find(qS2MM);
        if (it != state.fuseGroupS2MMChannel.end()) {
          s2mmChannel = it->second;
        } else {
          uint32_t maxS2MM_4c = 2;
          if (state.targetModel)
            maxS2MM_4c = state.targetModel->getNumDestSwitchboxConnections(
                static_cast<int>(consCol), static_cast<int>(consRow),
                AIE::WireBundle::DMA);
          int32_t nextS2MM_4c = state.tileNextS2MMChannel.count(consTileVal)
                                    ? state.tileNextS2MMChannel[consTileVal]
                                    : 0;
          if (static_cast<uint32_t>(nextS2MM_4c) >= maxS2MM_4c) {
            state.deviceOp.emitError(
                llvm::Twine("conduit-to-dma: S2MM DMA channel exhausted on "
                            "tile (") +
                llvm::Twine(consCol) + "," + llvm::Twine(consRow) +
                "): all " + llvm::Twine(maxS2MM_4c) + " channels in use");
            state.passFailed = true;
            return;
          }
          s2mmChannel = state.tileNextS2MMChannel[consTileVal]++;
          state.fuseGroupS2MMChannel[qS2MM] = s2mmChannel;
        }
        std::string bdKey = name + "__s2mm_" + std::to_string(consIdx);
        state.fuseGroupMembers[qS2MM].push_back(bdKey);
      } else {
        // Bounds-check S2MM channels on the consumer tile.
        uint32_t maxS2MM_4c = 2;
        if (state.targetModel)
          maxS2MM_4c = state.targetModel->getNumDestSwitchboxConnections(
              static_cast<int>(consCol), static_cast<int>(consRow),
              AIE::WireBundle::DMA);
        int32_t nextS2MM_4c = state.tileNextS2MMChannel.count(consTileVal)
                                  ? state.tileNextS2MMChannel[consTileVal]
                                  : 0;
        if (static_cast<uint32_t>(nextS2MM_4c) >= maxS2MM_4c) {
          state.deviceOp.emitError(
              llvm::Twine("conduit-to-dma: S2MM DMA channel exhausted on "
                          "tile (") +
              llvm::Twine(consCol) + "," + llvm::Twine(consRow) +
              "): all " + llvm::Twine(maxS2MM_4c) + " channels in use");
          state.passFailed = true;
          return;
        }
        s2mmChannel = state.tileNextS2MMChannel[consTileVal]++;
      }
      state.conduitConsS2MMChannel[{name, consIdx}] = s2mmChannel;

      // Deduplicate: skip if the same source→dest port pair was already
      // emitted by the packet broadcast path or a fuse group partner.
      // This prevents duplicate packet_flow + aie.flow for the same ports
      // when MM2S fuse group members share the same consumer S2MM channel.
      FlowKey fk{prodTileVal.getAsOpaquePointer(), mm2sChannel,
                 consTileVal.getAsOpaquePointer(), s2mmChannel};
      if (emittedFlowPorts.count(fk))
        continue;
      emittedFlowPorts.insert(fk);

      state.emitFlow(info.routingMode, prodTileVal, AIE::WireBundle::DMA,
                     mm2sChannel, consTileVal, AIE::WireBundle::DMA,
                     s2mmChannel);
    }
  }
  } // end flowPass two-pass loop

  // -----------------------------------------------------------------------
  // Phase 4c: Emit aie.cascade_flow for cascade-mode conduits.
  //
  // Cascade is point-to-point: exactly one producer tile, one consumer tile.
  // No aie.buffer, aie.lock, aie.dma_bd, or aie.flow is emitted.
  // The existing --aie-lower-cascade-flows pass converts cascade_flow to
  // aie.configure_cascade (sets direction registers on both tiles).
  // -----------------------------------------------------------------------

  for (auto &[name, info] : state.conduitMap) {
    // Multi-device: ensure tile lookups target the correct device.
    if (state.isMultiDevice())
      state.switchToDeviceIndex(info.deviceIndex);

    if (info.routingMode != "cascade")
      continue;

    auto [prodCol, prodRow] = info.producerTileCoord;
    if (prodCol < 0) {
      state.deviceOp.emitWarning(
          llvm::Twine("conduit-to-dma: cascade conduit '") + name +
          "' has no producer tile — skipped");
      continue;
    }

    // Cascade is strictly point-to-point (no broadcast, no shim).
    if (info.consumerTileCoords.size() != 1) {
      state.deviceOp.emitError(
          llvm::Twine("conduit-to-dma: cascade conduit '") + name +
          "' must have exactly one consumer tile (cascade is point-to-point), "
          "got ")
          << info.consumerTileCoords.size();
      // B-3 fix: return immediately so the cascade loop does not continue
      // processing subsequent conduits with already-broken state.
      state.passFailed = true;
      return;
    }
    if (!info.shimConsumerTileCoords.empty()) {
      state.deviceOp.emitError(
          llvm::Twine("conduit-to-dma: cascade conduit '") + name +
          "' has a shim consumer tile — cascade cannot connect to shim tiles");
      // B-3 fix: return immediately.
      state.passFailed = true;
      return;
    }

    AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
    if (!prodTile) {
      state.deviceOp.emitWarning(
          llvm::Twine("conduit-to-dma: cascade conduit '") + name +
          "' producer tile not found in device — skipped");
      continue;
    }

    auto [consCol, consRow] = info.consumerTileCoords[0];
    AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
    if (!consTile) {
      state.deviceOp.emitWarning(
          llvm::Twine("conduit-to-dma: cascade conduit '") + name +
          "' consumer tile not found in device — skipped");
      continue;
    }

    builder.setInsertionPoint(state.deviceBody->getTerminator());
    builder.create<AIE::CascadeFlowOp>(
        state.deviceOp.getLoc(), prodTile.getResult(), consTile.getResult());
  }
}

} // namespace xilinx::conduit
