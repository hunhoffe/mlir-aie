//===- DeviceMergeUtils.cpp - Device-merge host-orchestrator helpers ------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "DeviceMergeUtils.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/SmallVector.h"

namespace xilinx::conduit::detail {

namespace {

// Find the (typically single) aiex.run op inside a ConfigureOp's body.
// Returns nullptr if none is present.  If multiple are present (atypical for
// IRON-generated host code), the FIRST is returned.
static AIEX::RunOp findRunOpInConfigure(AIEX::ConfigureOp conf) {
  AIEX::RunOp result;
  for (mlir::Operation &op : conf.getBody().front()) {
    if (auto run = mlir::dyn_cast<AIEX::RunOp>(op)) {
      result = run;
      break;
    }
  }
  return result;
}

// Count aiex.run ops in a ConfigureOp body.
static unsigned countRunOpsInConfigure(AIEX::ConfigureOp conf) {
  unsigned n = 0;
  for (mlir::Operation &op : conf.getBody().front()) {
    if (mlir::isa<AIEX::RunOp>(op))
      ++n;
  }
  return n;
}

// Find the closest preceding sibling aiex.configure with the given symbol
// name in the same parent block as `conf`.  Returns null if no such sibling
// exists.
static AIEX::ConfigureOp
findSiblingConfigureBefore(AIEX::ConfigureOp conf, llvm::StringRef symName) {
  mlir::Block *parent = conf->getBlock();
  if (!parent)
    return {};
  AIEX::ConfigureOp closest;
  for (mlir::Operation &op : *parent) {
    if (&op == conf.getOperation())
      break;
    if (auto sib = mlir::dyn_cast<AIEX::ConfigureOp>(op)) {
      if (sib.getSymbol() == symName)
        closest = sib;
    }
  }
  return closest;
}

} // namespace

mlir::LogicalResult
rewriteHostConfigureOnDeviceMerge(mlir::ModuleOp module, AIE::DeviceOp devA,
                                  AIE::DeviceOp devB, mlir::Operation *seqA) {
  mlir::MLIRContext *ctx = module.getContext();

  llvm::StringRef devAName = devA.getSymName();
  llvm::StringRef devBName = devB.getSymName();
  if (devAName == devBName) {
    // Same symbol → same device; nothing to rewrite.
    return mlir::success();
  }

  // The surviving runtime_sequence's sym_name (if any).  After the body
  // merge, devB's sequence has either been spliced into seqA (seqA's name
  // preserved) or, if devA had no sequence pre-merge, seqB was promoted
  // (its name preserved).  Either way `seqA` here is the surviving op.
  llvm::StringRef survivingSeqName;
  if (auto rs = mlir::dyn_cast_or_null<AIE::RuntimeSequenceOp>(seqA))
    survivingSeqName = rs.getSymName();

  mlir::FlatSymbolRefAttr devARef =
      mlir::FlatSymbolRefAttr::get(ctx, devAName);
  mlir::FlatSymbolRefAttr survSeqRef;
  if (!survivingSeqName.empty())
    survSeqRef = mlir::FlatSymbolRefAttr::get(ctx, survivingSeqName);

  // Collect every aiex.configure in the module whose symbol matches devB.
  // Snapshot first; we'll mutate during the loop.
  llvm::SmallVector<AIEX::ConfigureOp> bRefs;
  module.walk([&](AIEX::ConfigureOp conf) {
    if (conf.getSymbol() == devBName)
      bRefs.push_back(conf);
  });

  if (bRefs.empty())
    return mlir::success();

  for (AIEX::ConfigureOp confB : bRefs) {
    AIEX::ConfigureOp confA = findSiblingConfigureBefore(confB, devAName);
    AIEX::RunOp runB = findRunOpInConfigure(confB);

    if (confA) {
      // --- Fold confB's body into confA (collapse two LoadPDI → one). ---
      AIEX::RunOp runA = findRunOpInConfigure(confA);

      // Refuse the fold if either side has multiple RunOps — IRON emits
      // exactly one per configure; multiple is a sign that the structure
      // doesn't match our spec, and the safe action is to leave the host
      // code alone and surface the issue to the caller.
      if (countRunOpsInConfigure(confA) > 1 ||
          countRunOpsInConfigure(confB) > 1) {
        confB.emitError(
            "device-merge: cannot fold aiex.configure with multiple aiex.run "
            "ops; expected at most one aiex.run per configure (IRON "
            "convention)");
        return mlir::failure();
      }

      mlir::Block &bodyA = confA.getBody().front();
      mlir::Block &bodyB = confB.getBody().front();

      // Move all non-RunOp ops in confB to just before confA's RunOp (or at
      // the end of bodyA if confA has no run).
      //
      // NOTE: do NOT call op->remove() before moveBefore(): moveBefore is
      // implemented as a list-splice and asserts the op is currently in a
      // block (Operation.cpp:558).  remove() detaches it (block becomes null),
      // which trips the assertion.  moveBefore handles the detach+reinsert
      // atomically.  For the push_back fallback we DO need to detach first
      // because Block::push_back expects a free-standing op.
      mlir::Operation *insertBefore = runA ? runA.getOperation() : nullptr;
      llvm::SmallVector<mlir::Operation *> toMove;
      for (mlir::Operation &op : bodyB) {
        if (mlir::isa<AIEX::RunOp>(op))
          continue;
        toMove.push_back(&op);
      }
      for (mlir::Operation *op : toMove) {
        if (insertBefore) {
          op->moveBefore(insertBefore);
        } else {
          op->remove();
          bodyA.push_back(op);
        }
      }

      // Concatenate the two RunOps into one.
      if (runA && runB) {
        llvm::SmallVector<mlir::Value> newOperands(runA.getArgs().begin(),
                                                   runA.getArgs().end());
        for (mlir::Value v : runB.getArgs())
          newOperands.push_back(v);

        mlir::FlatSymbolRefAttr seqRef =
            survSeqRef ? survSeqRef : runA.getRuntimeSequenceSymbolAttr();

        mlir::OpBuilder builder(runA);
        auto newRun = builder.create<AIEX::RunOp>(runA.getLoc(), seqRef,
                                                  newOperands);
        runA.erase();
        (void)newRun;
      } else if (runB && !runA) {
        // confA had no run (atypical) — move runB into confA.  insertBefore
        // is null here (it's runA), so we detach + push_back.
        runB->remove();
        bodyA.push_back(runB);
        if (survSeqRef)
          runB.setRuntimeSequenceSymbolAttr(survSeqRef);
      }
      // (runA && !runB): nothing to concatenate; leave runA alone.

      confB.erase();
    } else {
      // --- No sibling confA: rewrite confB in place. ---
      confB.setSymbolAttr(devARef);
      if (runB && survSeqRef)
        runB.setRuntimeSequenceSymbolAttr(survSeqRef);
    }
  }

  // Sanity check: every aiex.run inside an aiex.configure @devA that targets
  // the surviving sequence must agree with seqA's block-arg count.
  if (auto rs = mlir::dyn_cast_or_null<AIE::RuntimeSequenceOp>(seqA)) {
    unsigned expected = rs.getBody().front().getNumArguments();
    bool argMismatch = false;
    module.walk([&](AIEX::RunOp run) {
      auto parentConf = run->getParentOfType<AIEX::ConfigureOp>();
      if (!parentConf)
        return;
      if (parentConf.getSymbol() != devAName)
        return;
      if (run.getRuntimeSequenceSymbol() != survivingSeqName)
        return;
      if (run.getArgs().size() != expected) {
        run->emitError("aiex.run arg count ")
            << run.getArgs().size()
            << " disagrees with merged aie.runtime_sequence @"
            << survivingSeqName << " arity " << expected;
        argMismatch = true;
      }
    });
    if (argMismatch)
      return mlir::failure();
  }

  return mlir::success();
}

} // namespace xilinx::conduit::detail
