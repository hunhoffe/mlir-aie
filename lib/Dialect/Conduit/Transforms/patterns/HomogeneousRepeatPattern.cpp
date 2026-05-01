//===- HomogeneousRepeatPattern.cpp ------------------------------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "HomogeneousRepeatPattern.h"
#include "../CanonicalizeChannelPutsUtils.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::conduit;
using namespace xilinx::conduit::detail;

namespace {

//===----------------------------------------------------------------------===//
// Canonicalize: per-channel collapse
//===----------------------------------------------------------------------===//

// Try to collapse N structurally-identical PutMemrefAsync ops on `createOp`
// into 1 put + dma_repeat=N. Returns true iff the IR was modified.
bool tryCollapsePuts(Create createOp, PatternRewriter &rewriter) {
  StringRef chanName = createOp.getName();

  AIE::DeviceOp dev = findEnclosingDevice(createOp.getOperation());
  // Walk the enclosing scope (device, or module if no device) for puts.
  Operation *scope = dev ? dev.getOperation()
                         : createOp->getParentOfType<ModuleOp>().getOperation();
  if (!scope)
    return false;

  // Collect all PutMemrefAsync ops on this channel name within `scope`.
  llvm::SmallVector<PutMemrefAsync> puts;
  scope->walk([&](PutMemrefAsync p) {
    if (p.getName() == chanName)
      puts.push_back(p);
  });

  if (puts.size() < 2)
    return false;

  // All puts must live in the same block (so collapse keeps the surviving
  // put + chain in a single ordered place).
  Block *parentBlock = puts.front()->getBlock();
  for (PutMemrefAsync p : puts)
    if (p->getBlock() != parentBlock)
      return false;

  // Sort puts in IR order.
  llvm::sort(puts, [](PutMemrefAsync a, PutMemrefAsync b) {
    return a.getOperation()->isBeforeInBlock(b.getOperation());
  });

  // Structural identity check.
  PutMemrefAsync ref = puts.front();
  for (PutMemrefAsync p : llvm::drop_begin(puts))
    if (!putsAreStructurallyIdentical(ref, p))
      return false;

  // No deps: collapse semantics are "fire dma_repeat times" on a single
  // BD chain — incompatible with per-iteration dep edges.
  for (PutMemrefAsync p : puts)
    if (!p.getDeps().empty())
      return false;

  // Each put must have its own sync chain; all chains must have the same
  // shape. (Empty chain is uniform-only — also OK.)
  llvm::SmallVector<llvm::SmallVector<WaitAll>> chains;
  chains.reserve(puts.size());
  for (PutMemrefAsync p : puts) {
    auto maybeChain = collectSyncChain(p.getToken());
    if (!maybeChain)
      return false;
    chains.push_back(std::move(*maybeChain));
  }
  llvm::SmallVector<bool> refShape = chainShape(chains.front());
  for (auto &c : llvm::drop_begin(chains))
    if (chainShape(c) != refShape)
      return false;

  // Existing dma_repeat must be unset (= 0 additional fires = 1 total).
  // Refuse to multiply.  0-indexed convention per Bug #98 / Task #39.
  if (getDmaRepeatOr0(createOp) != 0)
    return false;

  // HW-cap check: refuse collapse if N exceeds EITHER the producer-tile
  // (MM2S) BD cap or the consumer-tile (S2MM) BD cap.  Pass C emits BD
  // blocks on both sides; canon's safety contract is to refuse rather
  // than push the verifier crash one branch downstream.
  int64_t N = static_cast<int64_t>(puts.size());
  Value producerTile = lookupProducerTile(scope, chanName);
  Value consumerTile = lookupConsumerTile(scope, chanName);
  uint32_t worstCap = std::numeric_limits<uint32_t>::max();
  bool capKnown = false;
  if (auto pc = tileBDCap(scope, producerTile)) {
    worstCap = std::min(worstCap, *pc);
    capKnown = true;
  }
  if (auto cc = tileBDCap(scope, consumerTile)) {
    worstCap = std::min(worstCap, *cc);
    capKnown = true;
  }
  if (capKnown && N > static_cast<int64_t>(worstCap)) {
    createOp.emitWarning()
        << "canonicalize-loop-unroll-puts: refusing to collapse " << N
        << " puts on @" << chanName << " — exceeds tile BD cap of " << worstCap
        << " (downstream Pass C will surface the underlying issue)";
    return false;
  }

  // Match holds — perform collapse. Erase puts[1..N-1] and their chains;
  // keep puts[0] and chains[0]. Erasures MUST go through the rewriter so
  // the greedy driver's worklist tracks the deletions (otherwise it pops
  // dangling pointers and SEGFAULTs in Operation::fold).
  for (size_t i = 1; i < puts.size(); ++i) {
    for (WaitAll w : chains[i])
      rewriter.eraseOp(w);
    rewriter.eraseOp(puts[i]);
  }

  // Stamp dma_repeat = N - 1 on the create (0-indexed convention; Bug #98 /
  // Task #39).  IRON's convention (aiex.py:289-291) is "repeat_count =
  // sizes[0] - 1" = "additional fires beyond the initial one"; canon must
  // match so Pass C's verbatim surface to configure_task.repeat_count
  // (ConduitToDMALower.cpp:1356-1359) yields the right firmware fire count
  // (`value + 1` per AIEDmaToNpu.cpp:180-183).  N >= 2 is guaranteed by
  // the `puts.size() < 2 → return false` early-out above, so N - 1 >= 1
  // and we never stamp a no-op `dma_repeat = 0`.  The mutation must
  // happen inside the modifyOpInPlace callback so the rewriter notifies
  // the driver correctly.
  rewriter.modifyOpInPlace(createOp, [&] {
    Builder b(createOp.getContext());
    createOp.setDmaRepeatAttr(b.getI64IntegerAttr(N - 1));
  });
  return true;
}

// Symmetric collapse for GetMemrefAsync. Same constraints, consumer-side cap.
bool tryCollapseGets(Create createOp, PatternRewriter &rewriter) {
  StringRef chanName = createOp.getName();

  AIE::DeviceOp dev = findEnclosingDevice(createOp.getOperation());
  Operation *scope = dev ? dev.getOperation()
                         : createOp->getParentOfType<ModuleOp>().getOperation();
  if (!scope)
    return false;

  llvm::SmallVector<GetMemrefAsync> gets;
  scope->walk([&](GetMemrefAsync g) {
    if (g.getName() == chanName)
      gets.push_back(g);
  });

  if (gets.size() < 2)
    return false;

  Block *parentBlock = gets.front()->getBlock();
  for (GetMemrefAsync g : gets)
    if (g->getBlock() != parentBlock)
      return false;

  llvm::sort(gets, [](GetMemrefAsync a, GetMemrefAsync b) {
    return a.getOperation()->isBeforeInBlock(b.getOperation());
  });

  GetMemrefAsync ref = gets.front();
  for (GetMemrefAsync g : llvm::drop_begin(gets))
    if (!getsAreStructurallyIdentical(ref, g))
      return false;

  for (GetMemrefAsync g : gets)
    if (!g.getDeps().empty())
      return false;

  llvm::SmallVector<llvm::SmallVector<WaitAll>> chains;
  chains.reserve(gets.size());
  for (GetMemrefAsync g : gets) {
    auto maybeChain = collectSyncChain(g.getToken());
    if (!maybeChain)
      return false;
    chains.push_back(std::move(*maybeChain));
  }
  llvm::SmallVector<bool> refShape = chainShape(chains.front());
  for (auto &c : llvm::drop_begin(chains))
    if (chainShape(c) != refShape)
      return false;

  // Existing dma_repeat must be unset (= 0 additional fires = 1 total).
  // 0-indexed convention per Bug #98 / Task #39.
  if (getDmaRepeatOr0(createOp) != 0)
    return false;

  int64_t N = static_cast<int64_t>(gets.size());
  Value producerTile = lookupProducerTile(scope, chanName);
  Value consumerTile = lookupConsumerTile(scope, chanName);
  uint32_t worstCap = std::numeric_limits<uint32_t>::max();
  bool capKnown = false;
  if (auto pc = tileBDCap(scope, producerTile)) {
    worstCap = std::min(worstCap, *pc);
    capKnown = true;
  }
  if (auto cc = tileBDCap(scope, consumerTile)) {
    worstCap = std::min(worstCap, *cc);
    capKnown = true;
  }
  if (capKnown && N > static_cast<int64_t>(worstCap)) {
    createOp.emitWarning()
        << "canonicalize-loop-unroll-puts: refusing to collapse " << N
        << " gets on @" << chanName << " — exceeds tile BD cap of " << worstCap
        << " (downstream Pass C will surface the underlying issue)";
    return false;
  }

  for (size_t i = 1; i < gets.size(); ++i) {
    for (WaitAll w : chains[i])
      rewriter.eraseOp(w);
    rewriter.eraseOp(gets[i]);
  }

  // Stamp dma_repeat = N - 1 on the create (0-indexed convention; symmetric
  // to tryCollapsePuts above; see Bug #98 / Task #39).  N >= 2 guaranteed
  // by the `gets.size() < 2 → return false` early-out so N - 1 >= 1.
  rewriter.modifyOpInPlace(createOp, [&] {
    Builder b(createOp.getContext());
    createOp.setDmaRepeatAttr(b.getI64IntegerAttr(N - 1));
  });
  return true;
}

//===----------------------------------------------------------------------===//
// OpRewritePattern entry points
//===----------------------------------------------------------------------===//

struct CollapsePutsPattern : public OpRewritePattern<Create> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Create createOp,
                                PatternRewriter &rewriter) const override {
    // The collapse mutates peer ops in the same block (puts + their wait_all
    // chains). All mutations are routed through `rewriter` so the greedy
    // driver's worklist tracks erasures/modifications correctly.
    if (!tryCollapsePuts(createOp, rewriter))
      return failure();
    return success();
  }
};

struct CollapseGetsPattern : public OpRewritePattern<Create> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Create createOp,
                                PatternRewriter &rewriter) const override {
    if (!tryCollapseGets(createOp, rewriter))
      return failure();
    return success();
  }
};

} // namespace

namespace xilinx::conduit::detail {

void populateHomogeneousRepeatPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<CollapsePutsPattern, CollapseGetsPattern>(ctx);
}

} // namespace xilinx::conduit::detail
