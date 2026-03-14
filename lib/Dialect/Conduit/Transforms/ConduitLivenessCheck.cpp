//===- ConduitLivenessCheck.cpp - conduit-check-liveness pass ----*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// M11: Window liveness verifier.
//
// Every !conduit.window<T> SSA value represents a granted hardware lock.
// A window that is never released causes permanent hardware deadlock because
// the lock counter never returns to the value that unblocks the peer.
//
// This pass checks that every window-producing op (conduit.acquire and
// conduit.wait_window) has at least one release path:
//
//   1. Direct SSA release: the window value appears as an operand of at least
//      one conduit.release in the same function.
//   2. Async name release: a conduit.release_async op with the same channel
//      name appears anywhere in the same function (async releases reference the
//      channel by name rather than by SSA value).
//
// A violation is reported as an error (not a warning) because a leaked lock
// grant always causes hardware deadlock — there is no valid program that
// intentionally acquires a lock and never releases it on any code path.
//
// Relationship to other verifiers:
//   M8a (checkWindowReleaseCumulativeCount): fires when release count > acquire
//        count — catches over-release (double-release).
//   M9  (ConduitPairingCheck): fires when no release is in the SAME BLOCK —
//        weaker heuristic, covers the common case but misses cross-block release.
//   M11 (this pass): fires when no release exists ANYWHERE in the function —
//        catches the strict "lock leaked entirely" case.
//
// M11 is strictly stronger than M9 in one direction: if M9 fires (no
// same-block release) but there IS a release in a different block, M11 does
// NOT fire — the lock is not leaked, only potentially out of order.  That
// ordering concern is left to future liveness analysis (M12).
//
// Limitations:
//   - Loop-carried patterns: if the acquire is inside a loop and the release is
//     conditionally skipped on the last iteration, M11 cannot detect this
//     (requires M12 / PostDominatorTree analysis).
//   - Name-based release_async: if a release_async uses a channel name that
//     does not correspond to any live acquire SSA value (e.g., through alias),
//     M11 may suppress a false negative.  This is conservative (no false
//     positives).
//   - M11 only checks functions (func.func); ops in module-level regions are
//     not checked.
//
// Run with:  aie-opt --conduit-check-liveness <input.mlir>
//
// This pass is OPT-IN and NOT part of the default pipeline.  It signals pass
// failure on the first leaked window found (unlike M9 which only warns).
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITLIVENESSCHECK
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// Collect the set of channel names for which a conduit.release_async exists
// anywhere in the given function.  Used to suppress M11 on windows that are
// released asynchronously by name rather than by SSA value.
static llvm::StringSet<>
collectAsyncReleaseNames(mlir::func::FuncOp func) {
  llvm::StringSet<> names;
  func.walk([&](ReleaseAsync relOp) { names.insert(relOp.getName()); });
  return names;
}

// Check a single window-producing op.  Returns failure() if the window is
// leaked (no release path found anywhere in the enclosing function).
//
// 'asyncReleaseNames' is pre-computed per-function to amortize the walk cost.
static mlir::LogicalResult
checkWindowLiveness(mlir::Operation *producerOp, mlir::Value windowVal,
                    llvm::StringRef conduitName,
                    const llvm::StringSet<> &asyncReleaseNames) {
  // Fast path 1: async release by name — release_async does not take an SSA
  // window operand, so if any release_async for this channel exists anywhere
  // in the function, the lock is eventually returned.
  if (asyncReleaseNames.count(conduitName))
    return mlir::success();

  // Fast path 2: direct SSA release — the window value must appear as the
  // operand of at least one conduit.release.
  for (mlir::OpOperand &use : windowVal.getUses()) {
    if (mlir::isa<Release>(use.getOwner()))
      return mlir::success();
  }

  // No release found via either path: the lock grant is permanently held.
  return producerOp->emitError(
      "M11: window lock grant is never released — "
      "no conduit.release or conduit.release_async for channel '")
      << conduitName
      << "' found in this function; leaked lock causes hardware deadlock";
}

struct ConduitLivenessCheckPass
    : public impl::ConduitLivenessCheckBase<ConduitLivenessCheckPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    bool anyFailure = false;

    // Process one function at a time so async-release name lookups are scoped.
    module.walk([&](mlir::func::FuncOp func) {
      llvm::StringSet<> asyncNames = collectAsyncReleaseNames(func);

      // Check conduit.acquire ops (blocking acquire path).
      func.walk([&](Acquire acqOp) {
        if (failed(checkWindowLiveness(acqOp, acqOp.getWindow(),
                                       acqOp.getName(), asyncNames)))
          anyFailure = true;
      });

      // Check conduit.wait_window ops (async acquire path).
      // acquire_async produces a window.token; wait_window materializes it
      // into a !conduit.window<T>.  That window value must be released.
      func.walk([&](WaitWindow waitOp) {
        if (failed(checkWindowLiveness(waitOp, waitOp.getWindow(),
                                       waitOp.getName(), asyncNames)))
          anyFailure = true;
      });
    });

    if (anyFailure)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitLivenessCheckPass() {
  return std::make_unique<ConduitLivenessCheckPass>();
}

} // namespace xilinx::conduit
