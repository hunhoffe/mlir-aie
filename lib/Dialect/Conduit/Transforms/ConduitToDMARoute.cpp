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
                               const std::string &conduitName,
                               ConduitInfo &info,
                               mlir::Value prodTileVal, int64_t prodCol,
                               int64_t prodRow,
                               mlir::Value consTileVal, int64_t consCol,
                               int64_t consRow,
                               unsigned consIdx) {
  mlir::Operation *prodTileOp = prodTileVal.getDefiningOp();
  mlir::Operation *consTileOp = consTileVal.getDefiningOp();

  // -----------------------------------------------------------------------
  // Step 3.5a: Check global packet ID budget.
  // -----------------------------------------------------------------------
  if (!state.packetIDAllocator || state.packetIDAllocator->remaining() == 0) {
    // No packet IDs left; Step 4 (error) will handle this.
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

    int32_t prodBDUsed = state.tileBDUsed.count(prodTileVal)
                             ? state.tileBDUsed[prodTileVal]
                             : 0;
    int32_t consBDUsed = state.tileBDUsed.count(consTileVal)
                             ? state.tileBDUsed[consTileVal]
                             : 0;
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
          llvm::Twine("conduit-to-dma: packet DMA ordering hazard: conduit '")
          + conduitName
          + "' and an existing flow both route to the same consumer tile "
            "through the same MM2S channel; ordering between them is not "
            "guaranteed under sustained load");
    }
  }

  // -----------------------------------------------------------------------
  // Step 3.5e: Allocate packet flow ID and emit aie.packet_flow.
  // -----------------------------------------------------------------------
  std::optional<uint8_t> pktID = state.packetIDAllocator->allocate();
  if (!pktID) {
    // Allocator emitted the error; signal failure.
    state.passFailed = true;
    return false;
  }

  // Assign S2MM channel on the consumer tile.
  int32_t s2mmChannel = state.tileNextS2MMChannel[consTileVal]++;
  state.conduitConsS2MMChannel[{conduitName, consIdx}] = s2mmChannel;

  // Record occupancy for future convergence checks.
  occupancy.push_back({*pktID, consTileOp});

  // Update BD usage budgets (each packet flow reserves depth BDs on each side).
  state.tileBDUsed[prodTileVal] += static_cast<int32_t>(depth);
  state.tileBDUsed[consTileVal] += static_cast<int32_t>(depth);

  // Record the MM2S channel assignment for Phase 5.5 (BD chain generation).
  state.conduitMM2SChannel[conduitName] = mm2sChannel;

  // Emit the packet flow directly using the pre-allocated ID.
  // We do NOT call state.emitFlow("packet", ...) here because emitFlow would
  // call packetIDAllocator->allocate() a second time (double allocation).
  // Instead, build the aie.PacketFlowOp directly with the ID we already hold.
  mlir::OpBuilder &builder = *state.builder;
  auto pktFlow = builder.create<AIE::PacketFlowOp>(
      state.deviceOp.getLoc(),
      static_cast<int8_t>(*pktID),
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

      builder.setInsertionPoint(state.deviceBody->getTerminator());

      // Shim-side prod/cons locks (AIE2 only).
      // For external-buffer conduits (AIE1), also emit a single shim lock
      // for the aie.shim_dma BD chain.
      // Note: shim-producer conduits (Phase 4a) cannot be link destinations
      // (link dsts are always MemTile/compute tile consumers, not shim
      // producers), so no linkDstNames guard is needed in 4a.
      // Skip lock allocation when disable_synchronization is set: the oracle
      // emits no locks or use_lock for these conduits.
      if (isAIE2 && !info.disableSynchronization) {
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
          !info.disableSynchronization) {
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
      int32_t shimMM2SCh =
          state.tileNextMM2SChannel[shimTile.getResult()]++;
      state.conduitMM2SChannel[name] = shimMM2SCh;

      std::string allocSym = name + "_shim_alloc";
      state.shimConduitNames.insert(name);
      if (!mlir::SymbolTable::lookupSymbolIn(
              state.deviceOp, mlir::StringAttr::get(ctx, allocSym)))
        builder.create<AIE::ShimDMAAllocationOp>(
            state.deviceOp.getLoc(), allocSym, shimTile.getResult(),
            AIE::DMAChannelDir::MM2S,
            /*channel_index=*/static_cast<int64_t>(shimMM2SCh),
            /*plio=*/false,
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
        int32_t s2mmCh =
            state.tileNextS2MMChannel[consTile.getResult()]++;
        state.conduitConsS2MMChannel[{name, consIdx}] = s2mmCh;
        state.emitFlow(info.routingMode, shimTile.getResult(),
                       AIE::WireBundle::DMA, shimMM2SCh,
                       consTile.getResult(), AIE::WireBundle::DMA,
                       s2mmCh);
      }
    }

    // --- Sub-case 4b: consumer is a shim tile (row==0) ---
    for (auto [shimCol, shimRow] : info.shimConsumerTileCoords) {
      if (shimRow != 0)
        continue;

      AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
      if (!prodTile)
        continue;

      AIE::TileOp shimTile = state.lookupTileByCoord(shimCol, shimRow);
      if (!shimTile)
        continue;

      builder.setInsertionPoint(state.deviceBody->getTerminator());

      // Shim-side consumer locks (AIE2: prod_lock + cons_lock;
      // AIE1: not needed — shim locks managed differently).
      // These are referenced by the host runtime when programming
      // shim S2MM DMA BDs; without them, the runtime has no locks
      // for flow control on the receive path.
      // Skip when disable_synchronization: oracle emits no locks for these.
      if (isAIE2 && !info.disableSynchronization) {
        {
          int lockIdx = state.lockIdCounter[shimTile.getResult()]++;
          // Naming convention: <conduit>_<endpoint-role>_<lock-role>_<idx>
          // "cons" = shim consumer endpoint; "prod" = this lock controls free
          // receive slots (DMA can write when >0).
          std::string symName = name + "_cons_prod_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), shimTile.getResult(), lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        }
        {
          int lockIdx = state.lockIdCounter[shimTile.getResult()]++;
          std::string symName = name + "_cons_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              state.deviceOp.getLoc(), shimTile.getResult(), lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        }
      }

      std::string allocSym = name + "_shim_alloc";
      state.shimConduitNames.insert(name);
      // Link-dst conduits: this allocation is intentionally kept —
      // linkPhase() emits the flow (memtile MM2S → shim S2MM) but does NOT
      // create a ShimDMAAllocationOp. routePhase owns the allocation for all
      // conduits with a shim consumer, including link-dst conduits.
      if (!mlir::SymbolTable::lookupSymbolIn(
              state.deviceOp, mlir::StringAttr::get(ctx, allocSym)))
        builder.create<AIE::ShimDMAAllocationOp>(
            state.deviceOp.getLoc(), allocSym, shimTile.getResult(),
            AIE::DMAChannelDir::S2MM,
            /*channel_index=*/static_cast<int64_t>(0),
            /*plio=*/false,
            /*packet=*/nullptr);

      // Emit the memtile→shim flow. For join-destination conduits the
      // producer tile is the MemTile; linkPhase() explicitly skips this
      // flow (see ConduitToDMALink.cpp) and expects routePhase to own it.
      state.emitFlow(info.routingMode, prodTile.getResult(),
                     AIE::WireBundle::DMA, static_cast<int32_t>(0),
                     shimTile.getResult(), AIE::WireBundle::DMA,
                     static_cast<int32_t>(0));
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
  // Fused channel groups: conduits with the same fused_dma_channel_group
  // label share one hardware MM2S channel slot.
  //
  // Step 3.5 (mode=any fallback): When routing_mode="any" and all MM2S
  // channels on the producer tile are allocated for circuit-switched DMA,
  // tryPacketFallback is invoked.  It attempts packet-switched DMA using an
  // existing or newly designated packet-mode physical channel, subject to
  // BD budget, lock budget, and global packet-flow-ID constraints.
  // -----------------------------------------------------------------------

  for (auto &[name, info] : state.conduitMap) {
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

    // ---- Determine hardware MM2S channel capacity for this tile. ----
    // Used by the mode=any exhaustion check (Step 3.5).
    uint32_t maxMM2S = 2; // hardware default: 2 MM2S per compute tile
    if (state.targetModel)
      maxMM2S = state.targetModel->getNumSourceSwitchboxConnections(
          static_cast<int>(prodCol), static_cast<int>(prodRow),
          AIE::WireBundle::DMA);

    // ---- Assign MM2S channel (fused groups share a channel). ----
    int32_t mm2sChannel = -1;
    bool usedPacketFallback = false;

    if (!info.fuseGroup.empty()) {
      auto it = state.fuseGroupMM2SChannel.find(info.fuseGroup);
      if (it != state.fuseGroupMM2SChannel.end()) {
        mm2sChannel = it->second;
      } else {
        mm2sChannel = state.tileNextMM2SChannel[prodTileVal]++;
        state.fuseGroupMM2SChannel[info.fuseGroup] = mm2sChannel;
      }
      state.fuseGroupMembers[info.fuseGroup].push_back(name);
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
      mm2sChannel = state.tileNextMM2SChannel[prodTileVal]++;
      state.conduitMM2SChannel[name] = mm2sChannel;
      // Track packet-mode channel designation for Step 3.5c.
      if (info.routingMode == "packet" && prodTile) {
        auto key = std::make_pair(prodTile.getOperation(),
                                  static_cast<int>(mm2sChannel));
        state.pktChannelState.isPacketChannel[key] = true;
      }
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
      // Exception: via_DMA=true forces DMA even for adjacent tiles.
      if (!info.viaDMA && info.consumerTileCoords.size() == 1) {
        bool rightAdj = state.targetModel->isLegalMemAffinity(prodCol, prodRow,
                                                              consCol, consRow);
        bool leftAdj = state.targetModel->isLegalMemAffinity(consCol, consRow,
                                                             prodCol, prodRow);
        if (rightAdj || leftAdj)
          continue;
      }

      AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
      if (!consTile)
        continue;

      mlir::Value consTileVal = consTile.getResult();

      if (usedPacketFallback) {
        // Step 3.5: attempt packet DMA fallback.
        bool ok = tryPacketFallback(state, name, info, prodTileVal, prodCol,
                                    prodRow, consTileVal, consCol, consRow,
                                    consIdx);
        if (!ok) {
          // Step 4: all modes exhausted — emit a hard error.
          state.deviceOp.emitError(
              llvm::Twine("conduit-to-dma: no DMA resources available for "
                          "conduit '")
              + name
              + "': circuit DMA MM2S channels exhausted on tile ("
              + llvm::Twine(prodCol) + "," + llvm::Twine(prodRow)
              + ") and packet DMA fallback is also ineligible "
                "(check BD budget, lock budget, and packet flow ID budget)");
          state.passFailed = true;
        }
        // tryPacketFallback records conduitMM2SChannel and
        // conduitConsS2MMChannel internally; skip the circuit path below.
        continue;
      }

      int32_t s2mmChannel = state.tileNextS2MMChannel[consTileVal]++;
      state.conduitConsS2MMChannel[{name, consIdx}] = s2mmChannel;

      state.emitFlow(info.routingMode, prodTileVal, AIE::WireBundle::DMA,
                     mm2sChannel, consTileVal, AIE::WireBundle::DMA,
                     s2mmChannel);
    }
  }

  // -----------------------------------------------------------------------
  // Phase 4c: Emit aie.cascade_flow for cascade-mode conduits.
  //
  // Cascade is point-to-point: exactly one producer tile, one consumer tile.
  // No aie.buffer, aie.lock, aie.dma_bd, or aie.flow is emitted.
  // The existing --aie-lower-cascade-flows pass converts cascade_flow to
  // aie.configure_cascade (sets direction registers on both tiles).
  // -----------------------------------------------------------------------

  for (auto &[name, info] : state.conduitMap) {
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
          "' must have exactly one consumer tile (cascade is point-to-point), got ")
          << info.consumerTileCoords.size();
      state.passFailed = true;
      continue;
    }
    if (!info.shimConsumerTileCoords.empty()) {
      state.deviceOp.emitError(
          llvm::Twine("conduit-to-dma: cascade conduit '") + name +
          "' has a shim consumer tile — cascade cannot connect to shim tiles");
      state.passFailed = true;
      continue;
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
    builder.create<AIE::CascadeFlowOp>(state.deviceOp.getLoc(),
                                        prodTile.getResult(),
                                        consTile.getResult());
  }
}

} // namespace xilinx::conduit
