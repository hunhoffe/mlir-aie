//===- ConduitCheckLoopBalance.cpp - conduit-check-loop-balance ---*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// MVE-1 analysis pass: --conduit-check-loop-balance
//
// Detects the Exp C class of token-deficit deadlock at compile time.
//
// Background:
//   `conduit.create` carries two attributes that bound the total DMA sends:
//
//   - `dma_repeat` (OptionalAttr<I64Attr>): the total number of times the DMA
//     engine runs the whole BD chain.  Maps to DMAStartOp.repeat_count =
//     dma_repeat - 1 with a non-circular BD chain.  dma_repeat=N means the DMA
//     fires exactly N times total, then stops.  Primary signal for finite-send.
//
//   - `bd_repeat` (OptionalAttr<I64Attr>): a per-BD unroll factor within one
//     chain pass.  Each BD in the chain fires bd_repeat times before advancing.
//     This is NOT the total send count; it depends on depth and BD structure.
//     Checking bd_repeat alone for finite-send detection is unsound.
//
// This pass checks ONLY `dma_repeat`, which is the unambiguous total send
// count.
//
// Check:
//   For each conduit.create with dma_repeat=N:
//     For each conduit.acquire on the Consume port referencing that channel:
//       Walk the acquire's parent op chain upward to find an enclosing scf.for.
//       If the scf.for has statically constant bounds (lb, ub, step):
//         T = (ub - lb) / step
//         If T > N: emit warning on the conduit.create.
//
// The Exp C deadlock pattern:
//   conduit.create @weights {dma_repeat = 64 : i64, depth = 1, ...}
//   aie.core { scf.for %i = 0 to 128 step 1 {  // T=128 > N=64 → DEADLOCK
//     conduit.acquire {name = @weights, ...}
//   }}
//
// Name matching:
//   conduit.acquire uses FlatSymbolRefAttr:$name.  The generated getName()
//   accessor calls getNameAttr().getValue(), returning the root reference
//   string (e.g. "weights" for @weights).  conduit.create uses
//   SymbolNameAttr:$sym_name with getName() as a backward-compat alias for
//   getSymName() → same StringRef. Both sides compare equal for matching
//   channels.
//
// This pass emits warnings only (never signals pass failure).
// It is OPT-IN and NOT part of the default pipeline.
//
// Run with:  aie-opt --conduit-check-loop-balance <input.mlir>
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITCHECKLOOPBALANCE
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// "Loop forever" sentinel bounds.  Duplicated here from ObjectFifoToConduit.cpp
// (canonical defs at :204 / :214) — a shared header is the better long-term
// home, but the warning emit below is the only second consumer today, so a
// local copy keeps this fix atomic.  Keep these in sync if either constant
// moves.  See ObjectFifoToConduit.cpp:200-213 for the empirical 2026-05-03
// validation context (companion commit aa20c968b1).
//
//  - kTripCountUnboundedSentinel: legacy `cmax = i64::MAX`-shaped bound
//    (compared with `>=` since the original lowering may further widen it).
//  - kAie24bitBdLoopSentinel: IRON's documented "loop forever" idiom using
//    the AIE 24-bit BD-loop saturation value (0xFFFFFE = 2^24 - 2).  This
//    value is BELOW kTripCountUnboundedSentinel, so the `>=` check alone
//    never fires for IRON-emitted infinite-loop cores; matched exactly.
static constexpr int64_t kTripCountUnboundedSentinel = int64_t{1} << 30;
static constexpr int64_t kAie24bitBdLoopSentinel = (int64_t{1} << 24) - 2;

/// Return the static trip count of an scf::ForOp, or -1 if not statically
/// known.  Uses getStaticTripCount() which is available when all three of
/// lb / ub / step are constants.
static int64_t getStaticTripCount(mlir::scf::ForOp forOp) {
  std::optional<llvm::APInt> tc = forOp.getStaticTripCount();
  if (!tc)
    return -1;
  return tc->getSExtValue();
}

/// Walk the parent op chain from `op` upward and return the first enclosing
/// scf::ForOp, or nullptr if none is found before a ModuleOp or null parent.
static mlir::scf::ForOp findEnclosingForOp(mlir::Operation *op) {
  mlir::Operation *cur = op->getParentOp();
  while (cur) {
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(cur))
      return forOp;
    cur = cur->getParentOp();
  }
  return nullptr;
}

struct ConduitCheckLoopBalancePass
    : public impl::ConduitCheckLoopBalanceBase<ConduitCheckLoopBalancePass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Build a map from channel name → total DMA send count.  dma_repeat
    // is 0-indexed (= "additional fires beyond the initial one"; see
    // CanonicalizeChannelPutsUtils.h::getDmaRepeatOr0 + Bug #98 / Task #39),
    // so total fires = dma_repeat + 1.  bd_repeat is a per-BD unroll factor
    // and is NOT the total send count; it is not checked here.  Only
    // channels with dma_repeat set are candidates.  ODS generates
    // std::optional<uint64_t> for I64Attr optional accessors.
    llvm::StringMap<int64_t> channelDmaRepeat;

    module.walk([&](Create createOp) {
      if (auto ic = createOp.getDmaRepeat())
        channelDmaRepeat[createOp.getSymName()] = static_cast<int64_t>(*ic) + 1;
    });

    if (channelDmaRepeat.empty())
      return;

    // For each conduit.acquire on the Consume port, check whether it is inside
    // a statically bounded scf.for with trip count exceeding dma_repeat.
    // getName() on Acquire returns FlatSymbolRefAttr::getValue() — the root
    // reference string (e.g. "ch" for @ch), matching getSymName() on Create.
    module.walk([&](Acquire acqOp) {
      // Only Consume-side acquires are bounded by the producer DMA send count.
      if (acqOp.getPort() != Port::Consume)
        return;

      llvm::StringRef chanName = acqOp.getName();
      auto it = channelDmaRepeat.find(chanName);
      if (it == channelDmaRepeat.end())
        return; // channel has no dma_repeat — skip

      // `totalFires` = dma_repeat + 1 (0-indexed convention; see map-build
      // comment above).  Compare loop trip count against total fires.
      int64_t totalFires = it->second;

      // Walk upward to find an enclosing scf.for.
      mlir::scf::ForOp forOp = findEnclosingForOp(acqOp);
      if (!forOp)
        return; // not inside any loop — no mismatch possible

      int64_t tripCount = getStaticTripCount(forOp);
      if (tripCount < 0)
        return; // dynamic bounds — cannot check statically

      // Suppress the warning for "loop forever" sentinel bounds: IRON's
      // 24-bit BD-loop saturation idiom (0xFFFFFE) and the legacy
      // i64::MAX-shaped bound both encode "termination is host-dispatch +
      // lock-blocking, not iter-count".  Comparing such a bound against
      // dma_repeat is virtually always > and produces a misleading
      // "token deficit" warning.  Mirror the silent-skip already used in
      // Pass A (ObjectFifoToConduit.cpp:715-716).  Companion commit
      // aa20c968b1 (2026-05-03) added the kAie24bitBdLoopSentinel handling
      // on the dma_repeat-stamping side; this is the second
      // getStaticTripCount caller that needs the same awareness.
      if (tripCount == kAie24bitBdLoopSentinel ||
          tripCount >= kTripCountUnboundedSentinel)
        return;

      if (tripCount > totalFires) {
        // Find the conduit.create to attach the warning to the declaration.
        module.walk([&](Create createOp) {
          if (createOp.getSymName() != chanName)
            return;
          createOp.emitWarning()
              << "conduit-check-loop-balance: channel '@" << chanName
              << "' fires " << totalFires
              << " total DMA sends (dma_repeat = " << (totalFires - 1)
              << ", 0-indexed) but consumer acquire is inside a loop"
              << " with trip count " << tripCount
              << " — consumer will stall after " << totalFires
              << " iterations (token deficit)";
        });
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCheckLoopBalancePass() {
  return std::make_unique<ConduitCheckLoopBalancePass>();
}

} // namespace xilinx::conduit
