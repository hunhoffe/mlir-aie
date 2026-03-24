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
// Detects the Exp C class of token-deficit deadlock at compile time:
//   A conduit.create with repeat_count=N (or iter_count=N) means the DMA
//   fires exactly N times.  If the consumer-side conduit.acquire for that
//   channel is inside an scf.for with a static trip count T > N, the
//   consumer will stall after N iterations because no more tokens arrive.
//
// Check:
//   For each conduit.create with repeat_count=N or iter_count=N:
//     For each conduit.acquire on the Consume port referencing that channel:
//       Walk the acquire's parent op chain upward.
//       If an enclosing scf.for is found with statically constant bounds:
//         T = (ub - lb) / step
//         If T > N: emit warning on the conduit.create.
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

    // Build a map from channel name → finite DMA count (repeat_count or
    // iter_count).  Only channels with at least one of these attributes are
    // checked.
    llvm::StringMap<int64_t> channelDMACount;

    module.walk([&](Create createOp) {
      // Prefer repeat_count; fall back to iter_count.
      // The ODS-generated getter returns std::optional<uint64_t> (the raw
      // integer value stored in the I64Attr), not llvm::APInt.
      if (auto rc = createOp.getRepeatCount()) {
        channelDMACount[createOp.getSymName()] =
            static_cast<int64_t>(*rc);
      } else if (auto ic = createOp.getIterCount()) {
        channelDMACount[createOp.getSymName()] =
            static_cast<int64_t>(*ic);
      }
    });

    if (channelDMACount.empty())
      return;

    // For each conduit.acquire on the Consume port, check whether it is inside
    // a statically bounded scf.for with trip count exceeding the DMA count.
    module.walk([&](Acquire acqOp) {
      // Only Consume-side acquires are bounded by the producer DMA count.
      if (acqOp.getPort() != Port::Consume)
        return;

      llvm::StringRef chanName = acqOp.getName();
      auto it = channelDMACount.find(chanName);
      if (it == channelDMACount.end())
        return; // channel has no finite DMA count — skip

      int64_t dmaCount = it->second;

      // Walk upward to find an enclosing scf.for.
      mlir::scf::ForOp forOp = findEnclosingForOp(acqOp);
      if (!forOp)
        return; // not inside any loop — no mismatch possible

      int64_t tripCount = getStaticTripCount(forOp);
      if (tripCount < 0)
        return; // dynamic bounds — cannot check statically

      if (tripCount > dmaCount) {
        // Find the conduit.create to attach the warning.
        module.walk([&](Create createOp) {
          if (createOp.getSymName() != chanName)
            return;
          createOp.emitWarning()
              << "conduit-check-loop-balance: channel '@" << chanName
              << "' has DMA count " << dmaCount
              << " but consumer acquire is inside a loop with trip count "
              << tripCount
              << " — consumer will stall after " << dmaCount
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
