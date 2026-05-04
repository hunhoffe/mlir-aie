//===- CanonicalizeChannelPuts.cpp ------------------------------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --conduit-canonicalize-channel-puts:
//   Driver pass for the channel-puts canonicalization.  Registers the
//   homogeneous-repeat collapse patterns (puts + gets) and runs them via
//   the greedy rewrite driver.
//
// Background: see CLAUDE.md "Active Open Bugs" HIGH row (compute-tile
// overlong-BD-chain crash) and "Converged cure" subsection.
//
//===----------------------------------------------------------------------===//

#include "patterns/ArithProgressionPattern.h"
#include "patterns/HomogeneousRepeatPattern.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITCANONICALIZECHANNELPUTS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

struct ConduitCanonicalizeChannelPutsPass
    : public impl::ConduitCanonicalizeChannelPutsBase<
          ConduitCanonicalizeChannelPutsPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    detail::populateHomogeneousRepeatPatterns(patterns, &getContext());
    detail::populateArithProgressionPatterns(patterns, &getContext());
    mlir::GreedyRewriteConfig cfg;
    cfg.setUseTopDownTraversal(true);
    if (mlir::failed(
            mlir::applyPatternsGreedily(module, std::move(patterns), cfg)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCanonicalizeChannelPutsPass() {
  return std::make_unique<ConduitCanonicalizeChannelPutsPass>();
}

} // namespace xilinx::conduit
