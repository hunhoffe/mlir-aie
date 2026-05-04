//===- ConduitCheckDeps.cpp - conduit-check-deps pass ------------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// M12: Dep-token DAG acyclicity check.
//
// The $deps operand on conduit.put_memref_async, conduit.get_memref_async, and
// conduit.wait_all_async carries !conduit.dma.token values that form a directed
// acyclic graph (DAG) of completion dependencies.  A cycle in this DAG means
// that operation A cannot start until operation B completes, and B cannot start
// until A completes — a static deadlock that no amount of buffering or
// scheduling can resolve.
//
// This pass builds the explicit dep-token DAG and runs DFS cycle detection with
// gray/black node coloring.  Any back edge (gray → gray) is reported as a hard
// error with a descriptive diagnostic.
//
// DAG construction:
//   Nodes: every !conduit.dma.token-typed SSA value in the module.
//   Edges: for each op with a $deps operand list, add an edge from each $deps
//          token (incoming: predecessor) to the result token of that op
//          (outgoing: successor).  This models "successor cannot start until
//          predecessor completes."
//
// Ops considered as token producers (nodes with outgoing edges):
//   - conduit.put_memref_async  → result: !conduit.dma.token
//   - conduit.get_memref_async  → result: !conduit.dma.token
//   - conduit.wait_all_async    → result: !conduit.dma.token
//
// Ops considered as token consumers (nodes with incoming edges):
//   - $deps operand list on any of the above ops
//
// IMPORTANT — PASSB-DEP-001 coverage gap:
//   This pass requires Task #24 (PASSB-DEP-001) to be fixed before it provides
//   complete coverage.  The bug: air.wait_all fan-in tokens are silently
//   dropped by Pass B — the $deps operand list on conduit.wait_all_async is
//   empty when it should carry the fan-in tokens.  Until the fix lands, cycles
//   that route through conduit.wait_all_async will not be detected because the
//   incoming edges into those nodes are missing.  After Task #24 lands, the
//   dep-DAG will be complete and this pass will catch all static dep-token
//   deadlocks.
//
// Severity: hard error + signalPassFailure().  A dep-token cycle is always a
// deadlock — there is no valid program with a circular completion dependency.
//
// Run with:  aie-opt --conduit-check-deps <input.mlir>
//
// This pass is OPT-IN and NOT part of the default pipeline.  It should run
// AFTER Pass B (which emits the dep tokens) and BEFORE Pass C (which lowers
// them to aie.use_lock ordering).
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITCHECKDEPS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// DAG representation
//
// Each node is identified by its mlir::Value (the !conduit.dma.token SSA
// value).  Edges are stored as adjacency lists: successors[v] is the list of
// token values that depend on v (i.e., the ops that list v in their $deps).
// ---------------------------------------------------------------------------

using TokenValue = mlir::Value;

// DFS node color for cycle detection.
enum class Color { White, Gray, Black };

// Collect all !conduit.dma.token SSA values produced in the module.
// Returns the set of all node values and their adjacency lists (successors).
//
// Two kinds of edges are added:
//
//   1. Direct dep edges: for each conduit op with a $deps operand list,
//      add an edge from each dep token to the op's result token.
//      This models "result cannot start until dep completes."
//
//   2. Loop back-edges (loop-carried tokens): for each RegionBranchOpInterface
//      op (e.g., scf.while, scf.for), walk the successor region's entry block
//      arguments and the values yielded back to those arguments.  If a yielded
//      value is a !conduit.dma.token and the corresponding block argument is
//      also a !conduit.dma.token, add an edge from the yielded value to the
//      block argument.  This closes the loop-carried dep cycle in the DAG.
//
//      Without back-edges, a loop-carried cycle (put deps on last iteration's
//      token, which is the block arg fed by this iteration's result) would be
//      invisible — the block arg node exists but has no incoming edges.
static void buildDepDAG(
    mlir::ModuleOp module,
    llvm::DenseMap<TokenValue, llvm::SmallVector<TokenValue, 4>> &successors,
    llvm::DenseMap<TokenValue, mlir::Operation *> &tokenToProducer) {

  // Pass 1: walk all ops for direct dep edges and op-result node registration.
  module.walk([&](mlir::Operation *op) {
    // Determine if this op produces a !conduit.dma.token result.
    mlir::Value resultToken;
    if (mlir::isa<PutMemrefAsync, GetMemrefAsync, WaitAllAsync>(op)) {
      // All three ops produce exactly one result of type !conduit.dma.token.
      if (op->getNumResults() == 1 &&
          mlir::isa<DMATokenType>(op->getResult(0).getType())) {
        resultToken = op->getResult(0);
      }
    }

    if (!resultToken)
      return;

    // Register the node (even if it has no deps — it may be a dep of others).
    tokenToProducer[resultToken] = op;
    if (!successors.count(resultToken))
      successors[resultToken] = {}; // ensure node exists in adjacency map

    // For each token in the $deps operand list, add edge: dep → resultToken.
    // The $deps operand is the variadic "tokens" operand on these ops.
    // We identify dep tokens by type: !conduit.dma.token.
    for (mlir::Value operand : op->getOperands()) {
      if (!mlir::isa<DMATokenType>(operand.getType()))
        continue;
      // This operand is a dep token.  Add edge: operand → resultToken.
      successors[operand].push_back(resultToken);
      // Ensure the dep node itself is registered (it may be a block argument
      // or produced by an op we haven't walked yet; pre-register it).
      if (!successors.count(operand))
        successors[operand] = {};
    }
  });

  // Pass 2: add loop back-edges by matching region entry block arguments to
  // yielded values from region terminators.
  //
  // For any op with regions (scf.while, scf.for, etc.), a loop-carried
  // !conduit.dma.token appears as a block argument of the region's entry block.
  // The corresponding terminator operand at the same index is the value yielded
  // back to that argument on each iteration.  Adding an edge from the yielded
  // value to the block argument closes the loop-carried cycle in the DAG.
  //
  // Approach: for each region in any op, walk all blocks.  For each block with
  // a terminator, for each !conduit.dma.token terminator operand at index i,
  // find the entry block argument at index i (if it is also a DMA token) and
  // add the back-edge: terminator_operand[i] → entry_block_arg[i].
  //
  // Also add the forward edge: op_operand[i] → entry_block_arg[i] for the
  // initial value supplied to the block argument from outside the region (the
  // "seed" value for the first iteration).
  module.walk([&](mlir::Operation *op) {
    for (mlir::Region &region : op->getRegions()) {
      if (region.empty())
        continue;
      mlir::Block &entryBlock = region.front();
      unsigned numArgs = entryBlock.getNumArguments();
      if (numArgs == 0)
        continue;

      // (a) Forward edge: op operand → entry block argument (initial value).
      // Conservative: match op operands to block args positionally only when
      // the op has exactly numArgs operands with matching DMA token types.
      // This handles scf.while (op operands match "before" region args 1:1).
      if (op->getNumOperands() == numArgs) {
        for (unsigned i = 0; i < numArgs; ++i) {
          mlir::Value initVal = op->getOperand(i);
          mlir::Value blockArg = entryBlock.getArgument(i);
          if (!mlir::isa<DMATokenType>(initVal.getType()))
            continue;
          if (!mlir::isa<DMATokenType>(blockArg.getType()))
            continue;
          successors[initVal].push_back(blockArg);
          if (!successors.count(blockArg))
            successors[blockArg] = {};
        }
      }

      // (b) Back-edge: terminator operand → entry block argument (yielded val).
      // Walk all blocks in the region; for each terminator, pair operands with
      // entry block args positionally.  This handles scf.condition (in
      // scf.while's "before" region) and scf.yield (in scf.for's body).
      for (mlir::Block &block : region) {
        mlir::Operation *term = block.getTerminator();
        if (!term || term->getNumOperands() == 0)
          continue;
        unsigned n = std::min((unsigned)term->getNumOperands(), numArgs);
        for (unsigned i = 0; i < n; ++i) {
          mlir::Value yieldedVal = term->getOperand(i);
          mlir::Value blockArg = entryBlock.getArgument(i);
          if (!mlir::isa<DMATokenType>(yieldedVal.getType()))
            continue;
          if (!mlir::isa<DMATokenType>(blockArg.getType()))
            continue;
          successors[yieldedVal].push_back(blockArg);
          if (!successors.count(yieldedVal))
            successors[yieldedVal] = {};
          if (!successors.count(blockArg))
            successors[blockArg] = {};
        }
      }
    }
  });
}

// ---------------------------------------------------------------------------
// DFS cycle detection with gray/black coloring.
//
// Gray: currently on the DFS stack (ancestor in DFS tree).
// Black: fully explored; no cycle through this node.
//
// If we reach a gray node during DFS, we have found a back edge → cycle.
// ---------------------------------------------------------------------------

static bool dfsDetectCycle(
    TokenValue node,
    llvm::DenseMap<TokenValue, llvm::SmallVector<TokenValue, 4>> &successors,
    llvm::DenseMap<TokenValue, mlir::Operation *> &tokenToProducer,
    llvm::DenseMap<TokenValue, Color> &color) {

  color[node] = Color::Gray;

  auto it = successors.find(node);
  if (it != successors.end()) {
    for (TokenValue succ : it->second) {
      Color succColor = color.count(succ) ? color[succ] : Color::White;

      if (succColor == Color::Gray) {
        // Back edge found — cycle detected.
        // Report the error on the nearest producer op we can find.
        // Prefer the op that produced 'succ' (the gray ancestor we just hit),
        // then the op that produced 'node' (the current node).
        // Block arguments have no defining op — use the module op as fallback.
        mlir::Operation *cycleOp = nullptr;
        if (tokenToProducer.count(succ))
          cycleOp = tokenToProducer[succ];
        else if (tokenToProducer.count(node))
          cycleOp = tokenToProducer[node];

        if (cycleOp) {
          cycleOp->emitError(
              "M12: dep token cycle detected: program is statically "
              "deadlocked -- circular completion dependency between DMA "
              "operations; no scheduling can resolve this");
        } else {
          // Fallback: both nodes are block arguments (loop-carried, no
          // defining op).  Emit on the parent op of the block argument.
          mlir::Block *parentBlock = nullptr;
          if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(succ))
            parentBlock = ba.getOwner();
          else if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(node))
            parentBlock = ba.getOwner();
          mlir::Operation *parentOp =
              parentBlock ? parentBlock->getParentOp() : nullptr;
          if (parentOp) {
            parentOp->emitError(
                "M12: dep token cycle detected: program is statically "
                "deadlocked -- circular completion dependency through "
                "loop-carried DMA token; no scheduling can resolve this");
          }
        }
        return true;
      }

      if (succColor == Color::White) {
        if (dfsDetectCycle(succ, successors, tokenToProducer, color))
          return true;
      }
      // Black: already fully explored, no cycle through succ.
    }
  }

  color[node] = Color::Black;
  return false;
}

// ---------------------------------------------------------------------------
// Main pass
// ---------------------------------------------------------------------------

struct ConduitCheckDepsPass
    : public impl::ConduitCheckDepsBase<ConduitCheckDepsPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Step 1: Build the dep-token DAG.
    // successors[v] = list of token values that have v as a dep (v → succ).
    llvm::DenseMap<TokenValue, llvm::SmallVector<TokenValue, 4>> successors;
    llvm::DenseMap<TokenValue, mlir::Operation *> tokenToProducer;
    buildDepDAG(module, successors, tokenToProducer);

    if (successors.empty())
      return; // No dep tokens in the module — nothing to check.

    // Step 2: DFS cycle detection with gray/black coloring.
    // Initialize all nodes as White (unvisited).
    llvm::DenseMap<TokenValue, Color> color;
    for (auto &[node, _] : successors)
      color[node] = Color::White;

    bool anyFailure = false;
    for (auto &[node, _] : successors) {
      if (color[node] == Color::White) {
        if (dfsDetectCycle(node, successors, tokenToProducer, color)) {
          anyFailure = true;
          // Continue checking other components — report all cycles, not just
          // the first.  Each DFS tree rooted at a White node is independent.
          // Reset the current stack state to avoid cascading false positives:
          // re-color all Gray nodes (nodes on the current DFS stack that we
          // popped back to due to early return) to Black so they don't appear
          // as false cycle participants in subsequent DFS roots.
          for (auto &[n, c] : color)
            if (c == Color::Gray)
              c = Color::Black;
        }
      }
    }

    if (anyFailure)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCheckDepsPass() {
  return std::make_unique<ConduitCheckDepsPass>();
}

} // namespace xilinx::conduit
