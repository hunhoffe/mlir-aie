//===- ArithProgressionPattern.cpp -------------------------------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "ArithProgressionPattern.h"
#include "../CanonicalizeChannelPutsUtils.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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
// Defensive: refuse arith collapse when the side that will receive the new
// outer wrap+stride dim is a compute (core) tile.
//
// Why: canon stamps the outer dim on producer_dimensions (puts) /
// consumer_dimensions (gets).  Pass C's runtime-sequence shim BD emit
// (ConduitToDMALower.cpp) pads to 4 dims to satisfy the AIEX
// lower-3-product==len rule, which is fine because runtime-sequence
// dma_bd ops bypass the AIEDialect dma_bd verifier (parent-class skip
// at AIEDialect.cpp:2194).  But Pass C's compute/memtile dma_bd emit
// (ConduitToDMALink.cpp) goes through the AIEDialect verifier, which
// caps non-MemTile (= compute) parents at maxNDims=3
// (AIEDialect.cpp:2233-2236).  A 4-dim padded form on a compute tile
// would crash that verifier.
//
// Today canon only fires on shim-flow channels in current Llama / lit
// workloads, so this is defensive: protects against future routing
// changes that could route an arith-collapsed channel's dim mutation
// to a compute tile.  Returns false (cannot determine) when tile lookup
// fails — fail-open here is correct because the existing tileBDCap
// check downstream still gates by HW BD count.
bool isComputeTile(Operation *scope, Value tile) {
  if (!tile)
    return false;
  auto tileOp = tile.getDefiningOp<AIE::TileOp>();
  if (!tileOp)
    return false;
  AIE::DeviceOp dev = scope->getParentOfType<AIE::DeviceOp>();
  if (!dev)
    dev = dyn_cast<AIE::DeviceOp>(scope);
  if (!dev)
    return false;
  const AIE::AIETargetModel &tm = AIE::getTargetModel(dev);
  return tm.isCoreTile(static_cast<int>(tileOp.getCol()),
                       static_cast<int>(tileOp.getRow()));
}

//===----------------------------------------------------------------------===//
// Local structural helpers (relax HomogeneousRepeat's check by allowing the
// leading offsets[0] entry to differ — that's the arith-progression slot).
//===----------------------------------------------------------------------===//

// Two DenseI64ArrayAttrs are equal ignoring index 0; both must have the same
// length (≥1) and identical entries from index 1 onward.
bool offsetsEqualIgnoringHead(DenseI64ArrayAttr a, DenseI64ArrayAttr b) {
  if (!a || !b)
    return false;
  ArrayRef<int64_t> av = a.asArrayRef();
  ArrayRef<int64_t> bv = b.asArrayRef();
  if (av.size() != bv.size() || av.empty())
    return false;
  for (size_t i = 1; i < av.size(); ++i)
    if (av[i] != bv[i])
      return false;
  return true;
}

bool denseI64ArrayEqLocal(DenseI64ArrayAttr a, DenseI64ArrayAttr b) {
  if (!a && !b)
    return true;
  if (!a || !b)
    return false;
  return a == b;
}

bool optionalAttrEqLocal(Attribute a, Attribute b) {
  if (!a && !b)
    return true;
  if (!a || !b)
    return false;
  return a == b;
}

// Like putsAreStructurallyIdentical but allows offsets[0] to differ.
bool putsArithCompatible(PutMemrefAsync a, PutMemrefAsync b) {
  if (a.getNameAttr() != b.getNameAttr())
    return false;
  if (a.getNumElems() != b.getNumElems())
    return false;
  if (!offsetsEqualIgnoringHead(a.getOffsetsAttr(), b.getOffsetsAttr()))
    return false;
  if (!denseI64ArrayEqLocal(a.getSizesAttr(), b.getSizesAttr()))
    return false;
  if (!denseI64ArrayEqLocal(a.getStridesAttr(), b.getStridesAttr()))
    return false;
  if (!optionalAttrEqLocal(a.getProducerDimensionsAttr(),
                           b.getProducerDimensionsAttr()))
    return false;
  return true;
}

bool getsArithCompatible(GetMemrefAsync a, GetMemrefAsync b) {
  if (a.getNameAttr() != b.getNameAttr())
    return false;
  if (a.getNumElems() != b.getNumElems())
    return false;
  if (!offsetsEqualIgnoringHead(a.getOffsetsAttr(), b.getOffsetsAttr()))
    return false;
  if (!denseI64ArrayEqLocal(a.getSizesAttr(), b.getSizesAttr()))
    return false;
  if (!denseI64ArrayEqLocal(a.getStridesAttr(), b.getStridesAttr()))
    return false;
  if (!optionalAttrEqLocal(a.getConsumerDimensionsAttr(),
                           b.getConsumerDimensionsAttr()))
    return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Attribute construction helpers
//===----------------------------------------------------------------------===//

// Build a fresh BDDimLayoutArrayAttr with `outer` prepended to `existing`.
// `existing` may be null/empty.  Returns nullptr only on failure modes the
// caller should never trigger.
AIE::BDDimLayoutArrayAttr prependDim(MLIRContext *ctx,
                                     AIE::BDDimLayoutArrayAttr existing,
                                     AIE::BDDimLayoutAttr outer) {
  llvm::SmallVector<AIE::BDDimLayoutAttr> dims;
  dims.push_back(outer);
  if (existing)
    for (AIE::BDDimLayoutAttr d : existing.getValue())
      dims.push_back(d);
  return AIE::BDDimLayoutArrayAttr::get(ctx, dims);
}

// Build a fresh BDDimLayoutArrayArrayAttr.  When `existing` is non-empty,
// prepend `outer` to each inner array.  When absent, produce a single
// inner array containing only `outer`.
AIE::BDDimLayoutArrayArrayAttr
prependDimArrayArray(MLIRContext *ctx, AIE::BDDimLayoutArrayArrayAttr existing,
                     AIE::BDDimLayoutAttr outer) {
  llvm::SmallVector<AIE::BDDimLayoutArrayAttr> outer_arrays;
  if (existing && !existing.getValue().empty()) {
    for (AIE::BDDimLayoutArrayAttr inner : existing.getValue())
      outer_arrays.push_back(prependDim(ctx, inner, outer));
  } else {
    outer_arrays.push_back(prependDim(ctx, /*existing=*/nullptr, outer));
  }
  return AIE::BDDimLayoutArrayArrayAttr::get(ctx, outer_arrays);
}

//===----------------------------------------------------------------------===//
// Canonicalize: per-channel arith-progression collapse
//===----------------------------------------------------------------------===//

// Try to collapse N PutMemrefAsync ops on `createOp` whose leading offset
// is in arithmetic progression into 1 put + outer wrap+stride dim +
// dma_repeat = N.  Returns true iff the IR was modified.
bool tryCollapseArithPuts(Create createOp, PatternRewriter &rewriter) {
  StringRef chanName = createOp.getName();
  MLIRContext *ctx = createOp.getContext();

  AIE::DeviceOp dev = findEnclosingDevice(createOp.getOperation());
  Operation *scope = dev ? dev.getOperation()
                         : createOp->getParentOfType<ModuleOp>().getOperation();
  if (!scope)
    return false;

  llvm::SmallVector<PutMemrefAsync> puts;
  scope->walk([&](PutMemrefAsync p) {
    if (p.getName() == chanName)
      puts.push_back(p);
  });

  if (puts.size() < 2)
    return false;

  Block *parentBlock = puts.front()->getBlock();
  for (PutMemrefAsync p : puts)
    if (p->getBlock() != parentBlock)
      return false;

  llvm::sort(puts, [](PutMemrefAsync a, PutMemrefAsync b) {
    return a.getOperation()->isBeforeInBlock(b.getOperation());
  });

  // Structural identity (modulo offsets[0]).
  PutMemrefAsync ref = puts.front();
  for (PutMemrefAsync p : llvm::drop_begin(puts))
    if (!putsArithCompatible(ref, p))
      return false;

  // No deps: collapse semantics replay one BD chain N times.
  for (PutMemrefAsync p : puts)
    if (!p.getDeps().empty())
      return false;

  // Sync chain shape must match across siblings.
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

  // Refuse when the per-put sync chain requests per-issue acknowledgment
  // (any wait_all{token=true}).  Symmetric to HomogeneousRepeatPattern's
  // refusal; see chainHasAwait docstring.  Collapsed arith form encodes
  // the per-cycle offset variation as an outer wrap+stride dim — same
  // single-configure / single-ack failure mode as the homogeneous case.
  if (chainHasAwait(refShape)) {
    createOp.emitRemark()
        << "canon: refusing to collapse channel '" << chanName
        << "' — sync chain requests per-issue acknowledgment "
           "(wait_all{token=true}); collapsing N tokens → 1 would "
           "starve the per-chunk consumer-side ack and stall HW";
    return false;
  }

  // Refuse on linked channels.  See isLinkedChannel docstring; same
  // root-cause class as the homogeneous-repeat refusal — collapsing
  // produces a single configure with an outer wrap+stride dim that
  // does not compose with multi-round consumer pacing on the linked
  // Pass C path.
  if (isLinkedChannel(scope, chanName)) {
    createOp.emitRemark()
        << "canon: refusing to collapse channel '" << chanName
        << "' — participates in aie.objectfifo.link; linked path "
           "wants N separate paced configures, not collapsed dma_repeat";
    return false;
  }

  // Arith-progression on offsets[0]: offsets[i][0] == base + i × stride.
  ArrayRef<int64_t> firstOff = ref.getOffsetsAttr().asArrayRef();
  if (firstOff.empty())
    return false;
  int64_t base = firstOff[0];
  ArrayRef<int64_t> secondOff = puts[1].getOffsetsAttr().asArrayRef();
  int64_t stride = secondOff[0] - base;
  if (stride == 0)
    return false;
  // BDDimLayoutAttr stride is uint32_t — refuse non-positive strides.
  if (stride < 0 || stride > std::numeric_limits<uint32_t>::max())
    return false;
  for (size_t i = 0; i < puts.size(); ++i) {
    int64_t expected = base + static_cast<int64_t>(i) * stride;
    if (puts[i].getOffsetsAttr().asArrayRef()[0] != expected)
      return false;
  }

  // HW-cap check (mirrors HomogeneousRepeatPattern).
  int64_t N = static_cast<int64_t>(puts.size());
  if (N > std::numeric_limits<uint32_t>::max())
    return false;
  Value producerTile = lookupProducerTile(scope, chanName);
  Value consumerTile = lookupConsumerTile(scope, chanName);
  // Defensive: refuse if the producer side (where producer_dimensions is
  // consumed by Pass C BD emit) is a compute tile.  See isComputeTile
  // comment for the rationale (AIEDialect 3-dim cap on non-MemTile
  // parents would reject Pass C's 4-dim padded form).
  if (isComputeTile(scope, producerTile)) {
    createOp.emitWarning()
        << "canonicalize-loop-unroll-puts: refusing to collapse " << N
        << " puts on @" << chanName
        << " — producer is a compute tile and the outer wrap+stride dim "
           "would emit on a compute-tile dma_bd that the AIEDialect "
           "verifier caps at 3 dims";
    return false;
  }
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

  // HW per-BD data-layout-dim cap.  Collapse will prepend ONE outer
  // wrap+stride dim to producer_dimensions, so the post-collapse literal
  // dim count is K_existing + 1.  Refuse if either side's dim cap would
  // be violated.  K_existing is the literal stamped attr length — see
  // ArithProgressionPattern.cpp:301 (prependDim) for the prepend op, and
  // ConduitToDMALower.cpp BD emit which consumes producer_dimensions
  // verbatim before AIEDialect's dim-count verifier runs.
  //
  // TODO Sprint N+2 canon unification: replace dims.size() with
  // stripTrailingPadding(dims).size() to allow collapse on op7-shape
  // inputs whose effective geometry is 1-dim despite 4-dim literal padding.
  uint32_t worstDimCap = std::numeric_limits<uint32_t>::max();
  bool dimCapKnown = false;
  if (auto pdc = tileBDDimCap(scope, producerTile)) {
    worstDimCap = std::min(worstDimCap, *pdc);
    dimCapKnown = true;
  }
  if (auto cdc = tileBDDimCap(scope, consumerTile)) {
    worstDimCap = std::min(worstDimCap, *cdc);
    dimCapKnown = true;
  }
  size_t kExisting = 0;
  if (auto refDims = mlir::dyn_cast_or_null<AIE::BDDimLayoutArrayAttr>(
          ref.getProducerDimensionsAttr()))
    kExisting = refDims.getValue().size();
  if (dimCapKnown && (kExisting + 1) > static_cast<size_t>(worstDimCap)) {
    createOp.emitWarning()
        << "canonicalize-loop-unroll-puts: refusing to collapse " << N
        << " puts on @" << chanName << " — post-collapse dim count "
        << (kExisting + 1) << " exceeds per-BD dim cap of " << worstDimCap;
    return false;
  }

  // Match holds — perform collapse.
  // Build the outer wrap+stride BDDimLayoutAttr and prepend on (a) the
  // surviving put's producer_dimensions, (b) the channel-level
  // producer_dimensions on conduit.create.
  AIE::BDDimLayoutAttr outer = AIE::BDDimLayoutAttr::get(
      ctx, static_cast<uint32_t>(N), static_cast<uint32_t>(stride));
  AIE::BDDimLayoutArrayAttr putNewDims =
      prependDim(ctx,
                 mlir::dyn_cast_or_null<AIE::BDDimLayoutArrayAttr>(
                     ref.getProducerDimensionsAttr()),
                 outer);
  AIE::BDDimLayoutArrayAttr createNewDims =
      prependDim(ctx,
                 mlir::dyn_cast_or_null<AIE::BDDimLayoutArrayAttr>(
                     createOp.getProducerDimensionsAttr()),
                 outer);

  // Erase puts[1..N-1] and their chains via the rewriter so the greedy
  // driver tracks the deletions.
  for (size_t i = 1; i < puts.size(); ++i) {
    for (WaitAll w : chains[i])
      rewriter.eraseOp(w);
    rewriter.eraseOp(puts[i]);
  }

  rewriter.modifyOpInPlace(ref,
                           [&] { ref.setProducerDimensionsAttr(putNewDims); });
  rewriter.modifyOpInPlace(createOp, [&] {
    // The outer wrap+stride dim encodes BOTH cycle count (N) AND per-cycle
    // offset variation (stride).  DO NOT also set dma_repeat=N — that would
    // double-multiply the transfer count (N × N).  dma_repeat is reserved
    // for the homogeneous case (stride==0) handled by HomogeneousRepeatPattern.
    createOp.setProducerDimensionsAttr(createNewDims);
  });
  return true;
}

// Symmetric: collapse N GetMemrefAsync ops with arith-progression offsets[0].
bool tryCollapseArithGets(Create createOp, PatternRewriter &rewriter) {
  StringRef chanName = createOp.getName();
  MLIRContext *ctx = createOp.getContext();

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
    if (!getsArithCompatible(ref, g))
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

  // Refuse when the per-get sync chain requests per-issue acknowledgment
  // (any wait_all{token=true}).  Symmetric to tryCollapseArithPuts above;
  // see chainHasAwait docstring.
  if (chainHasAwait(refShape)) {
    createOp.emitRemark()
        << "canon: refusing to collapse channel '" << chanName
        << "' — sync chain requests per-issue acknowledgment "
           "(wait_all{token=true}); collapsing N tokens → 1 would "
           "starve the per-chunk consumer-side ack and stall HW";
    return false;
  }

  // Refuse on linked channels.  Symmetric to tryCollapseArithPuts above;
  // see isLinkedChannel docstring.
  if (isLinkedChannel(scope, chanName)) {
    createOp.emitRemark()
        << "canon: refusing to collapse channel '" << chanName
        << "' — participates in aie.objectfifo.link; linked path "
           "wants N separate paced configures, not collapsed dma_repeat";
    return false;
  }

  ArrayRef<int64_t> firstOff = ref.getOffsetsAttr().asArrayRef();
  if (firstOff.empty())
    return false;
  int64_t base = firstOff[0];
  ArrayRef<int64_t> secondOff = gets[1].getOffsetsAttr().asArrayRef();
  int64_t stride = secondOff[0] - base;
  if (stride == 0)
    return false;
  if (stride < 0 || stride > std::numeric_limits<uint32_t>::max())
    return false;
  for (size_t i = 0; i < gets.size(); ++i) {
    int64_t expected = base + static_cast<int64_t>(i) * stride;
    if (gets[i].getOffsetsAttr().asArrayRef()[0] != expected)
      return false;
  }

  int64_t N = static_cast<int64_t>(gets.size());
  if (N > std::numeric_limits<uint32_t>::max())
    return false;
  Value producerTile = lookupProducerTile(scope, chanName);
  Value consumerTile = lookupConsumerTile(scope, chanName);
  // Defensive: refuse if the consumer side (where consumer_dimensions is
  // consumed by Pass C BD emit) is a compute tile.  Mirrors the put-side
  // refuse — see isComputeTile comment for rationale.
  if (isComputeTile(scope, consumerTile)) {
    createOp.emitWarning()
        << "canonicalize-loop-unroll-puts: refusing to collapse " << N
        << " gets on @" << chanName
        << " — consumer is a compute tile and the outer wrap+stride dim "
           "would emit on a compute-tile dma_bd that the AIEDialect "
           "verifier caps at 3 dims";
    return false;
  }
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

  // HW per-BD data-layout-dim cap on the consumer side.  consumer_dimensions
  // is BDDimLayoutArrayArrayAttr — one inner array per consumer.  Collapse
  // prepends ONE outer wrap+stride dim to EACH inner array (see
  // prependDimArrayArray), so the post-collapse literal dim count for each
  // consumer is innerSize + 1.  Refuse if any inner array would overflow.
  // Use the longest inner array as the worst case.
  //
  // TODO Sprint N+2 canon unification: replace dims.size() with
  // stripTrailingPadding(dims).size() to allow collapse on op7-shape
  // inputs whose effective geometry is 1-dim despite 4-dim literal padding.
  uint32_t worstDimCap = std::numeric_limits<uint32_t>::max();
  bool dimCapKnown = false;
  if (auto pdc = tileBDDimCap(scope, producerTile)) {
    worstDimCap = std::min(worstDimCap, *pdc);
    dimCapKnown = true;
  }
  if (auto cdc = tileBDDimCap(scope, consumerTile)) {
    worstDimCap = std::min(worstDimCap, *cdc);
    dimCapKnown = true;
  }
  size_t kExisting = 0;
  if (auto refDims = mlir::dyn_cast_or_null<AIE::BDDimLayoutArrayArrayAttr>(
          ref.getConsumerDimensionsAttr())) {
    for (AIE::BDDimLayoutArrayAttr inner : refDims.getValue())
      kExisting = std::max(kExisting, inner.getValue().size());
  }
  if (dimCapKnown && (kExisting + 1) > static_cast<size_t>(worstDimCap)) {
    createOp.emitWarning()
        << "canonicalize-loop-unroll-puts: refusing to collapse " << N
        << " gets on @" << chanName << " — post-collapse dim count "
        << (kExisting + 1) << " exceeds per-BD dim cap of " << worstDimCap;
    return false;
  }

  AIE::BDDimLayoutAttr outer = AIE::BDDimLayoutAttr::get(
      ctx, static_cast<uint32_t>(N), static_cast<uint32_t>(stride));
  AIE::BDDimLayoutArrayArrayAttr getNewDims = prependDimArrayArray(
      ctx,
      mlir::dyn_cast_or_null<AIE::BDDimLayoutArrayArrayAttr>(
          ref.getConsumerDimensionsAttr()),
      outer);
  AIE::BDDimLayoutArrayArrayAttr createNewDims = prependDimArrayArray(
      ctx,
      mlir::dyn_cast_or_null<AIE::BDDimLayoutArrayArrayAttr>(
          createOp.getConsumerDimensionsAttr()),
      outer);

  for (size_t i = 1; i < gets.size(); ++i) {
    for (WaitAll w : chains[i])
      rewriter.eraseOp(w);
    rewriter.eraseOp(gets[i]);
  }

  rewriter.modifyOpInPlace(ref,
                           [&] { ref.setConsumerDimensionsAttr(getNewDims); });
  rewriter.modifyOpInPlace(createOp, [&] {
    // The outer wrap+stride dim encodes BOTH cycle count (N) AND per-cycle
    // offset variation (stride).  DO NOT also set dma_repeat=N — that would
    // double-multiply the transfer count (N × N).  dma_repeat is reserved
    // for the homogeneous case (stride==0) handled by HomogeneousRepeatPattern.
    createOp.setConsumerDimensionsAttr(createNewDims);
  });
  return true;
}

//===----------------------------------------------------------------------===//
// OpRewritePattern entry points
//===----------------------------------------------------------------------===//

struct CollapseArithPutsPattern : public OpRewritePattern<Create> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Create createOp,
                                PatternRewriter &rewriter) const override {
    if (!tryCollapseArithPuts(createOp, rewriter))
      return failure();
    return success();
  }
};

struct CollapseArithGetsPattern : public OpRewritePattern<Create> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(Create createOp,
                                PatternRewriter &rewriter) const override {
    if (!tryCollapseArithGets(createOp, rewriter))
      return failure();
    return success();
  }
};

} // namespace

namespace xilinx::conduit::detail {

void populateArithProgressionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<CollapseArithPutsPattern, CollapseArithGetsPattern>(ctx);
}

} // namespace xilinx::conduit::detail
