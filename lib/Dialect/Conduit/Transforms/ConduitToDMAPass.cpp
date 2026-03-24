//===- ConduitToDMAPass.cpp - Pass class + runOnOperation shell --*-C++-*-===//
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
// This file contains only the pass class definition and runOnOperation()
// shell.  The actual lowering logic is split across per-phase files:
//
//   ConduitToDMACollect.cpp  — Phase 1-2.5: conduitMap, tile cache, depth
//   ConduitToDMAAlloc.cpp    — Phase 3: buffer + lock allocation
//   ConduitToDMARoute.cpp    — Phase 4-4.5a: shim DMA, flow emission
//   ConduitToDMALink.cpp     — Phase 5-5.5: link + BD chains
//   ConduitToDMALower.cpp    — Phase 6-8: acquire/release/async + erasure
//
//===----------------------------------------------------------------------===//

#include "ConduitToDMACommon.h"
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITTODMA
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

// ---------------------------------------------------------------------------
// Post-link verification: MemTile BD parity pool check.
//
// AIE2 MemTiles have 48 BDs partitioned by DMA channel index parity:
//   BDs 0–23  (24 BDs): EVEN-numbered channels (0, 2, 4) — both S2MM and MM2S
//   BDs 24–47 (24 BDs): ODD-numbered channels  (1, 3, 5) — both S2MM and MM2S
//
// Pass C previously checked only total BDs ≤ 48 per MemTile (via tileBDUsed)
// but not the per-pool constraint (≤ 24 per parity pool).  This caused
// false-positive exit 0: designs that fit in 48 total BDs but exceed 24 in
// one parity pool would pass aie-opt and then crash silently at aiecc.
//
// This function walks all aie.memtile_dma regions after BD chain generation,
// counts BDs per DMA channel by following DMAStartOp → NextBDOp chains,
// groups by channel index parity, and emits a hard error if either pool
// exceeds its hardware limit.
// ---------------------------------------------------------------------------
static bool verifyMemTileBDParity(ConduitToDMAState &state) {
  if (!state.deviceOp)
    return true;

  // B-10: The BD parity pool constraint (even-channel pool: BDs 0-23,
  // odd-channel pool: BDs 24-47, max 24 each) is AIE2-specific hardware.
  // AIE1 does not have this partitioning; applying this check to AIE1 would
  // incorrectly reject valid designs.  Future architectures should be
  // verified against their target model before enabling this check.
  //
  // TODO: query target model for BD pool split when the API is available.
  if (state.aieArch != AIE::AIEArch::AIE2)
    return true;

  bool passed = true;

  state.deviceOp.walk([&](AIE::MemTileDMAOp mtDMA) {
    int32_t evenPoolBDs = 0;
    int32_t oddPoolBDs = 0;

    // Walk all DMAStartOps inside this MemTileDMA region.
    mtDMA.walk([&](AIE::DMAStartOp dmaStart) {
      int32_t channelIndex = dmaStart.getChannelIndex();
      bool isEven = (channelIndex % 2 == 0);

      // Follow the BD chain starting from the first BD block (successor 0).
      if (dmaStart->getNumSuccessors() < 1)
        return;
      mlir::Block *startBlock = dmaStart->getSuccessor(0);
      if (!startBlock)
        return;

      llvm::DenseSet<mlir::Block *> visited;
      mlir::Block *current = startBlock;
      int32_t bdCount = 0;

      while (current && !visited.count(current)) {
        visited.insert(current);
        // Count DMABDOps in this block (each block has at most one).
        for (auto &op : *current)
          if (mlir::isa<AIE::DMABDOp>(op))
            bdCount++;
        // Follow NextBDOp to the next BD block in the chain.
        mlir::Operation *term = current->getTerminator();
        if (term && mlir::isa<AIE::NextBDOp>(term) &&
            term->getNumSuccessors() > 0)
          current = term->getSuccessor(0);
        else
          break;
      }

      if (isEven)
        evenPoolBDs += bdCount;
      else
        oddPoolBDs += bdCount;
    });

    // Extract tile coordinates for the diagnostic message.
    int col = -1, row = -1;
    if (auto tileOp = llvm::dyn_cast<AIE::TileOp>(
            mtDMA.getTile().getDefiningOp())) {
      col = tileOp.getCol();
      row = tileOp.getRow();
    }

    // Determine per-pool limit from the target model.  AIE2 MemTiles have
    // 48 BDs total, split evenly: 24 per parity pool.
    int32_t maxPerPool = 24;
    if (state.targetModel) {
      uint32_t totalBDs = state.targetModel->getNumBDs(col, row);
      maxPerPool = static_cast<int32_t>(totalBDs / 2);
    }

    if (evenPoolBDs > maxPerPool) {
      mtDMA.emitError(
          "MemTile BD parity constraint violated: even-channel pool has ")
          << evenPoolBDs << " BDs, exceeds hardware limit of " << maxPerPool
          << " (tile (" << col << "," << row << "))";
      passed = false;
    }
    if (oddPoolBDs > maxPerPool) {
      mtDMA.emitError(
          "MemTile BD parity constraint violated: odd-channel pool has ")
          << oddPoolBDs << " BDs, exceeds hardware limit of " << maxPerPool
          << " (tile (" << col << "," << row << "))";
      passed = false;
    }
  });

  return passed;
}

// ---------------------------------------------------------------------------
// Post-link verification: DMA channel budget check.
//
// Each tile has a hardware-limited number of DMA channels:
//   Compute tiles (AIE2): 2 MM2S + 2 S2MM
//   MemTiles (AIE2):      6 MM2S + 6 S2MM
//   Shim tiles:           2 MM2S + 2 S2MM (varies by architecture)
//
// Pass C tracks allocated channels via tileNextMM2SChannel and
// tileNextS2MMChannel counters.  If any tile exceeds its hardware limit,
// the generated IR contains invalid DMA channel indices that silently
// pass aie-opt verification but crash aiecc with an AIEPathFinder
// assertion (`i < sb.srcPorts.size()`).
//
// This function checks all allocated channels against the target model's
// limits and emits a hard error for any overflow.
// ---------------------------------------------------------------------------
static bool verifyDMAChannelBudgets(ConduitToDMAState &state) {
  if (!state.deviceOp || !state.targetModel)
    return true;

  bool passed = true;
  const auto &targetModel = *state.targetModel;

  for (auto &[tileVal, nextCh] : state.tileNextMM2SChannel) {
    auto tileOp = llvm::dyn_cast<AIE::TileOp>(tileVal.getDefiningOp());
    if (!tileOp)
      continue;
    int col = tileOp.getCol();
    int row = tileOp.getRow();
    // Skip shim tiles (row 0): shim DMA channels are managed by
    // aie.shim_dma_allocation and use a different hardware mechanism
    // (getNumSourceSwitchboxConnections returns 0 for shim tiles).
    if (targetModel.isShimNOCTile(col, row) ||
        targetModel.isShimPLTile(col, row))
      continue;
    uint32_t maxMM2S = targetModel.getNumSourceSwitchboxConnections(
        col, row, AIE::WireBundle::DMA);
    if (static_cast<uint32_t>(nextCh) > maxMM2S) {
      tileOp.emitError(
          "DMA channel budget exceeded: tile(")
          << col << "," << row << ") needs " << nextCh
          << " MM2S channels but hardware maximum is " << maxMM2S;
      passed = false;
    }
  }

  for (auto &[tileVal, nextCh] : state.tileNextS2MMChannel) {
    auto tileOp = llvm::dyn_cast<AIE::TileOp>(tileVal.getDefiningOp());
    if (!tileOp)
      continue;
    int col = tileOp.getCol();
    int row = tileOp.getRow();
    // Skip shim tiles — same reason as above.
    if (targetModel.isShimNOCTile(col, row) ||
        targetModel.isShimPLTile(col, row))
      continue;
    uint32_t maxS2MM = targetModel.getNumDestSwitchboxConnections(
        col, row, AIE::WireBundle::DMA);
    if (static_cast<uint32_t>(nextCh) > maxS2MM) {
      tileOp.emitError(
          "DMA channel budget exceeded: tile(")
          << col << "," << row << ") needs " << nextCh
          << " S2MM channels but hardware maximum is " << maxS2MM;
      passed = false;
    }
  }

  return passed;
}

namespace {

struct ConduitToDMAPass : impl::ConduitToDMABase<ConduitToDMAPass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());

    // B-3/B-9: Error protocol for multi-phase pass.
    // Phases set state.passFailed = true and return early when they encounter
    // an unrecoverable error. runOnOperation() checks passFailed after each
    // phase and calls signalPassFailure() + returns if set.
    // Do NOT call signalPassFailure() inside a module.walk() callback — it
    // does not stop the walk. Instead, set state.passFailed and return from
    // the walk callback; runOnOperation() handles signalPassFailure() centrally.

    // Initialize shared state.
    ConduitToDMAState state;
    state.module = module;
    state.builder = &builder;
    state.ctx = module.getContext();

    // Phase 1-2.5: Collect conduit metadata, find device, build tile cache,
    // compute effective depths, gather link source names.
    collectPhase(state);
    if (state.passFailed) { signalPassFailure(); return; }

    // Initialize the packet flow ID allocator after collectPhase has resolved
    // the architecture (AIETargetModel).  AIE1 and AIE2 both support up to 32
    // distinct packet flow IDs (5-bit hardware field, values 0–31).  If the
    // target model ever exposes a different limit, query it here.
    //
    // NOTE: We query the model after collectPhase because collectPhase is where
    // state.targetModel and state.aieArch are populated.
    uint8_t pktIDLimit = 32; // default for AIE1 and AIE2
    // Future: if targetModel exposes getNumPacketFlowIDs(), use it here.
    state.packetIDAllocator.emplace(module, pktIDLimit);

    // Phase 3: Allocate buffers and locks for each conduit.
    allocPhase(state);
    if (state.passFailed) { signalPassFailure(); return; }

    // Phase 4-4.5a: Shim DMA allocation, symbol rewriting, flow emission.
    routePhase(state);
    if (state.passFailed) { signalPassFailure(); return; }

    // Phase 5-5.5: Link lowering, aie.mem BD chains, fused chains.
    linkPhase(state);
    if (state.passFailed) { signalPassFailure(); return; }

    // Post-link verification: DMA channel budget.
    // Must run after routePhase() + linkPhase() which finalize all channel
    // allocations.  Catches MemTile MM2S/S2MM overflow (max 6 each) and
    // compute tile overflow (max 2 each) before the invalid IR reaches
    // downstream tools (aiecc pathfinder assertion crash).
    if (!verifyDMAChannelBudgets(state)) { signalPassFailure(); return; }

    // Post-link verification: MemTile BD parity pool constraint.
    // Must run after linkPhase() which generates all MemTile BD chains.
    if (!verifyMemTileBDParity(state)) { signalPassFailure(); return; }

    // Phase 6-8: Acquire/release lowering, op erasure, async path.
    lowerPhase(state);
    if (state.passFailed) { signalPassFailure(); return; }
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
