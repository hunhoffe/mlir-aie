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
//   - `iter_count` (OptionalAttr<I64Attr>): the total number of DMA task-queue
//     iterations.  Maps to DMAStartOp.repeat_count = iter_count - 1 with a
//     non-circular BD chain.  iter_count=N means the DMA fires exactly N times
//     total, then stops.  This is the primary signal for finite-send channels.
//
//   - `repeat_count` (OptionalAttr<I64Attr>): a per-BD repeat count.  Each BD
//     in the chain fires repeat_count times before advancing.  This is NOT the
//     total send count; it depends on depth (chain length) and BD structure.
//     Checking repeat_count alone for finite-send detection is unsound.
//
// This pass checks ONLY `iter_count`, which is the unambiguous total send count.
//
// Check:
//   For each conduit.create with iter_count=N:
//     For each conduit.acquire on the Consume port referencing that channel:
//       Walk the acquire's parent op chain upward to find an enclosing scf.for.
//       If the scf.for has statically constant bounds (lb, ub, step):
//         T = (ub - lb) / step
//         If T > N: emit warning on the conduit.create.
//
// The Exp C deadlock pattern:
//   conduit.create @weights {iter_count = 64 : i64, depth = 1, ...}
//   aie.core { scf.for %i = 0 to 128 step 1 {  // T=128 > N=64 → DEADLOCK
//     conduit.acquire {name = @weights, ...}
//   }}
//
// Name matching:
//   conduit.acquire uses FlatSymbolRefAttr:$name.  The generated getName()
//   accessor calls getNameAttr().getValue(), returning the root reference string
//   (e.g. "weights" for @weights).  conduit.create uses SymbolNameAttr:$sym_name
//   with getName() as a backward-compat alias for getSymName() → same StringRef.
//   Both sides compare equal for matching channels.
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

    // Build a map from channel name → total DMA send count (iter_count only).
    // iter_count=N means the DMA fires exactly N times total.  repeat_count
    // is a per-BD repeat factor and is NOT the total send count; it is not
    // checked here.  Only channels with iter_count set are candidates.
    // ODS generates std::optional<uint64_t> for I64Attr optional accessors.
    llvm::StringMap<int64_t> channelIterCount;

    module.walk([&](Create createOp) {
      if (auto ic = createOp.getIterCount())
        channelIterCount[createOp.getSymName()] = static_cast<int64_t>(*ic);
    });

    if (channelIterCount.empty())
      return;

    // For each conduit.acquire on the Consume port, check whether it is inside
    // a statically bounded scf.for with trip count exceeding iter_count.
    // getName() on Acquire returns FlatSymbolRefAttr::getValue() — the root
    // reference string (e.g. "ch" for @ch), matching getSymName() on Create.
    module.walk([&](Acquire acqOp) {
      // Only Consume-side acquires are bounded by the producer DMA send count.
      if (acqOp.getPort() != Port::Consume)
        return;

      llvm::StringRef chanName = acqOp.getName();
      auto it = channelIterCount.find(chanName);
      if (it == channelIterCount.end())
        return; // channel has no iter_count — skip

      int64_t iterCount = it->second;

      // Walk upward to find an enclosing scf.for.
      mlir::scf::ForOp forOp = findEnclosingForOp(acqOp);
      if (!forOp)
        return; // not inside any loop — no mismatch possible

      int64_t tripCount = getStaticTripCount(forOp);
      if (tripCount < 0)
        return; // dynamic bounds — cannot check statically

      if (tripCount > iterCount) {
        // Find the conduit.create to attach the warning to the declaration.
        module.walk([&](Create createOp) {
          if (createOp.getSymName() != chanName)
            return;
          createOp.emitWarning()
              << "conduit-check-loop-balance: channel '@" << chanName
              << "' has iter_count " << iterCount
              << " (total DMA sends) but consumer acquire is inside a loop"
              << " with trip count " << tripCount
              << " — consumer will stall after " << iterCount
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
