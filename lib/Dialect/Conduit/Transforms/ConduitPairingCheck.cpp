//===- ConduitPairingCheck.cpp - conduit-check-pairing pass ------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Optional analysis pass: checks that every conduit.acquire / wait_window op
// in a basic block has a matching release in the same block (M9 Phase 2).
//
// This pass is extracted from the M9 Phase 2 logic that previously lived inside
// Acquire::verify() / AcquireAsync::verify() / WaitWindow::verify().  Moving it
// here avoids two problems with emitting diagnostics inside verify():
//   1. MLIR diagnostic infinite recursion when verify() calls emitWarning().
//   2. Diagnostics non-capturable by FileCheck (llvm::errs() workaround).
//
// Run with:  aie-opt --conduit-check-pairing <input.mlir>
//
// This pass is OPT-IN and NOT part of the default pipeline.  It fires warnings
// only; it never signals pass failure.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITPAIRINGCHECK
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

/// Check whether a window value has a matching release in the same block.
/// Returns true if a release (sync or async by name) is found.
static bool hasReleaseInBlock(mlir::Value win, mlir::Block *block,
                               llvm::StringRef conduitName) {
  // First check direct Release users in same block.
  for (mlir::OpOperand &use : win.getUses()) {
    if (mlir::isa<Release>(use.getOwner()) &&
        use.getOwner()->getBlock() == block)
      return true;
  }
  // Also accept ReleaseAsync on the same conduit name in the same block
  // (async release does not take the window SSA value directly).
  for (mlir::Operation &op : *block) {
    if (auto relAsync = mlir::dyn_cast<ReleaseAsync>(op)) {
      if (relAsync.getName() == conduitName)
        return true;
    }
  }
  return false;
}

struct ConduitPairingCheckPass
    : public impl::ConduitPairingCheckBase<ConduitPairingCheckPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Check conduit.acquire ops.
    module.walk([&](Acquire acquireOp) {
      mlir::Value win = acquireOp.getWindow();
      mlir::Block *block = acquireOp->getBlock();
      if (!block)
        return;
      if (!hasReleaseInBlock(win, block, acquireOp.getName()))
        acquireOp.emitWarning(
            "M9: conduit.acquire has no matching release in same block");
    });

    // Check conduit.wait_window ops (async acquire path).
    module.walk([&](WaitWindow waitWinOp) {
      mlir::Value win = waitWinOp.getWindow();
      mlir::Block *block = waitWinOp->getBlock();
      if (!block)
        return;
      if (!hasReleaseInBlock(win, block, waitWinOp.getName()))
        waitWinOp.emitWarning(
            "M9: conduit.wait_window has no matching release in same block");
    });

    // Check conduit.acquire_async → wait_window chains.
    // The warning above on wait_window covers this case; no duplicate needed.

    // -----------------------------------------------------------------------
    // Cascade pairing check:
    // Every conduit.put_cascade must have exactly one matching
    // conduit.get_cascade in some core body (same conduit name).
    // -----------------------------------------------------------------------

    // Collect all put_cascade ops by name across the module.
    // putMap: name → list of PutCascade ops (there should be one per name)
    llvm::StringMap<llvm::SmallVector<PutCascade, 1>> putMap;
    module.walk([&](PutCascade putOp) {
      putMap[putOp.getName()].push_back(putOp);
    });

    // Collect all get_cascade ops by name across the module.
    // getMap: name → list of GetCascade ops (there should be one per name)
    llvm::StringMap<llvm::SmallVector<GetCascade, 1>> getMap;
    module.walk([&](GetCascade getOp) {
      getMap[getOp.getName()].push_back(getOp);
    });

    // For each name with a put_cascade: check that a matching get_cascade exists.
    for (auto &[name, puts] : putMap) {
      auto it = getMap.find(name);
      if (it == getMap.end()) {
        // No get_cascade for this name anywhere.
        for (PutCascade putOp : puts)
          putOp.emitWarning(
              "M9: unmatched conduit.put_cascade: no corresponding "
              "get_cascade found in any consumer core");
      } else {
        // get_cascade exists — check for ambiguous multiple consumers.
        if (it->second.size() > 1) {
          for (GetCascade getOp : it->second)
            getOp.emitWarning(
                "M9: ambiguous cascade: multiple get_cascade ops for the "
                "same conduit name");
        }
      }
    }

    // For each name with a get_cascade: check that a matching put_cascade exists.
    for (auto &[name, gets] : getMap) {
      if (putMap.find(name) == putMap.end()) {
        // No put_cascade for this name anywhere.
        for (GetCascade getOp : gets)
          getOp.emitWarning(
              "M9: unmatched conduit.get_cascade: no corresponding "
              "put_cascade found in any producer core");
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitPairingCheckPass() {
  return std::make_unique<ConduitPairingCheckPass>();
}

} // namespace xilinx::conduit
