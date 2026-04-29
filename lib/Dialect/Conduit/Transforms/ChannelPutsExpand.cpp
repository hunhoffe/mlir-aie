//===- ChannelPutsExpand.cpp ------------------------------------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --conduit-expand-channel-puts:
//   Inverse of --conduit-canonicalize-channel-puts; replicate the
//   canonicalized form (1 put + dma_repeat=N) back into N copies for
//   downstream passes that need per-batch IR-level mutation.
//
//===----------------------------------------------------------------------===//

#include "CanonicalizeChannelPutsUtils.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITEXPANDCHANNELPUTS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Local helpers — outer-dim strip / offset rewrite for arith-progression
// expand.  (Mirrors the prepend helpers in ArithProgressionPattern.cpp:
// canon prepends an outer wrap+stride dim; expand strips that same dim.)
//===----------------------------------------------------------------------===//

// Strip the leading entry off a BDDimLayoutArrayAttr.  Returns nullptr when
// the result would be empty (caller should `removeXxxAttr()` in that case to
// restore the absent-attr state).
AIE::BDDimLayoutArrayAttr stripOuterDim(mlir::MLIRContext *ctx,
                                        AIE::BDDimLayoutArrayAttr dims) {
  if (!dims || dims.getValue().empty())
    return nullptr;
  auto inner = dims.getValue().drop_front(1);
  if (inner.empty())
    return nullptr;
  return AIE::BDDimLayoutArrayAttr::get(
      ctx, llvm::SmallVector<AIE::BDDimLayoutAttr>(inner.begin(), inner.end()));
}

// Symmetric: strip the leading entry off each inner array of a
// BDDimLayoutArrayArrayAttr.  Returns nullptr if every inner array
// drops to empty (caller restores absent-attr state).
AIE::BDDimLayoutArrayArrayAttr
stripOuterDimArrayArray(mlir::MLIRContext *ctx,
                        AIE::BDDimLayoutArrayArrayAttr dims) {
  if (!dims || dims.getValue().empty())
    return nullptr;
  llvm::SmallVector<AIE::BDDimLayoutArrayAttr> outer;
  for (AIE::BDDimLayoutArrayAttr inner : dims.getValue()) {
    AIE::BDDimLayoutArrayAttr stripped = stripOuterDim(ctx, inner);
    if (stripped)
      outer.push_back(stripped);
  }
  if (outer.empty())
    return nullptr;
  return AIE::BDDimLayoutArrayArrayAttr::get(ctx, outer);
}

// Build a fresh DenseI64ArrayAttr whose [0] is replaced by `newHead`,
// preserving entries [1..].
mlir::DenseI64ArrayAttr makeOffsetsWithHead(mlir::MLIRContext *ctx,
                                            mlir::DenseI64ArrayAttr orig,
                                            int64_t newHead) {
  llvm::ArrayRef<int64_t> ov = orig.asArrayRef();
  llvm::SmallVector<int64_t> v;
  v.reserve(ov.size());
  v.push_back(newHead);
  for (size_t i = 1; i < ov.size(); ++i)
    v.push_back(ov[i]);
  return mlir::DenseI64ArrayAttr::get(ctx, v);
}

// Per-OpTy producer/consumer-dimensions accessors.  Specializations below
// keep expandOne polymorphic without `if constexpr`.
template <typename OpTy>
struct DimsTraits;

template <> struct DimsTraits<PutMemrefAsync> {
  using DimsAttr = AIE::BDDimLayoutArrayAttr;
  static DimsAttr getOpDims(PutMemrefAsync op) {
    return mlir::dyn_cast_or_null<DimsAttr>(op.getProducerDimensionsAttr());
  }
  static void setOpDims(PutMemrefAsync op, DimsAttr v) {
    if (v)
      op.setProducerDimensionsAttr(v);
    else
      op.removeProducerDimensionsAttr();
  }
  static DimsAttr getCreateDims(Create c) {
    return mlir::dyn_cast_or_null<DimsAttr>(c.getProducerDimensionsAttr());
  }
  static void setCreateDims(Create c, DimsAttr v) {
    if (v)
      c.setProducerDimensionsAttr(v);
    else
      c.removeProducerDimensionsAttr();
  }
  static DimsAttr stripOuter(mlir::MLIRContext *ctx, DimsAttr v) {
    return stripOuterDim(ctx, v);
  }
  // Returns the leading BDDimLayoutAttr (the canon-prepended outer dim) iff
  // present.  Used to detect arith-progression form.
  static AIE::BDDimLayoutAttr leadingDim(DimsAttr v) {
    if (!v || v.getValue().empty())
      return nullptr;
    return v.getValue()[0];
  }
};

template <> struct DimsTraits<GetMemrefAsync> {
  using DimsAttr = AIE::BDDimLayoutArrayArrayAttr;
  static DimsAttr getOpDims(GetMemrefAsync op) {
    return mlir::dyn_cast_or_null<DimsAttr>(op.getConsumerDimensionsAttr());
  }
  static void setOpDims(GetMemrefAsync op, DimsAttr v) {
    if (v)
      op.setConsumerDimensionsAttr(v);
    else
      op.removeConsumerDimensionsAttr();
  }
  static DimsAttr getCreateDims(Create c) {
    return mlir::dyn_cast_or_null<DimsAttr>(c.getConsumerDimensionsAttr());
  }
  static void setCreateDims(Create c, DimsAttr v) {
    if (v)
      c.setConsumerDimensionsAttr(v);
    else
      c.removeConsumerDimensionsAttr();
  }
  static DimsAttr stripOuter(mlir::MLIRContext *ctx, DimsAttr v) {
    return stripOuterDimArrayArray(ctx, v);
  }
  // Detect arith-progression marker by inspecting the FIRST inner array's
  // leading dim (canon prepends to every inner array, so any inner is fine
  // — first is convenient).
  static AIE::BDDimLayoutAttr leadingDim(DimsAttr v) {
    if (!v || v.getValue().empty())
      return nullptr;
    AIE::BDDimLayoutArrayAttr inner = v.getValue()[0];
    if (!inner || inner.getValue().empty())
      return nullptr;
    return inner.getValue()[0];
  }
};

//===----------------------------------------------------------------------===//
// Expand: replicate 1 put + dma_repeat=N back to N copies
//===----------------------------------------------------------------------===//

// Templated body for both PutMemrefAsync and GetMemrefAsync cases of
// expandLoopUnrollPuts.  Returns success when expansion happened.
//
// Two collapse forms are inverted here:
//   1. Homogeneous repeat (HomogeneousRepeatPattern): N puts all at the same
//      offsets → 1 put + dma_repeat=N.  Expand: clone N-1 copies with
//      identical offsets.
//   2. Arith progression (ArithProgressionPattern): N puts whose offsets[0]
//      forms `base + i*stride` → 1 put + dma_repeat=N + outer wrap+stride dim
//      prepended to producer/consumer dimensions.  Expand: clone N-1 copies
//      with offsets[0] = base + i*stride and strip the outer dim from both
//      the surviving op and conduit.create.
//
// Detection: arith-progression form leaves the surviving op (and the create)
// with a leading BDDimLayoutAttr whose `size == dma_repeat`.  Homogeneous
// form leaves the producer/consumer dims as canon found them, so the first
// dim — if any — is unrelated to dma_repeat.
template <typename OpTy>
mlir::LogicalResult expandOne(Create createOp, mlir::OpBuilder &builder) {
  using Traits = DimsTraits<OpTy>;
  using DimsAttr = typename Traits::DimsAttr;

  llvm::StringRef chanName = createOp.getName();
  mlir::MLIRContext *ctx = createOp.getContext();
  AIE::DeviceOp dev = detail::findEnclosingDevice(createOp.getOperation());
  mlir::Operation *scope =
      dev ? dev.getOperation()
          : createOp->getParentOfType<mlir::ModuleOp>().getOperation();
  if (!scope)
    return mlir::failure();

  llvm::SmallVector<OpTy> ops;
  scope->walk([&](OpTy o) {
    if (o.getName() == chanName)
      ops.push_back(o);
  });
  if (ops.size() != 1)
    return mlir::failure();

  OpTy origOp = ops.front();
  auto chain = detail::collectSyncChain(origOp.getToken());
  if (!chain)
    return mlir::failure();

  // Detect canonical form via mutually-exclusive markers ("two encodings,
  // two paths" — mirrors stateful's repeat_count vs dimensions split):
  //   - Homogeneous repeat:  conduit.create has dma_repeat = N (> 1);
  //     producer/consumer_dimensions untouched.  Pass C reads dma_repeat
  //     into DMAStartOp::repeat_count (Link.cpp:1807, 1901, 2356).
  //   - Arith-progression:   surviving op's leading BDDimLayoutAttr has
  //     size > 1 AND stride != 0; dma_repeat ABSENT.  Pass C reads the
  //     outer dim as in-BD TAP wrap+stride.
  // Setting BOTH on one channel double-multiplies transfers (N × N), so
  // canon stamps only one marker per channel — expand detects which.
  // Detection order: dma_repeat first (homogeneous wins on ambiguous IR);
  // arith only fires when dma_repeat is absent/1.
  bool isArith = false;
  int64_t N = detail::getDmaRepeatOr1(createOp);
  int64_t base = 0;
  int64_t stride = 0;
  if (N > 1) {
    // Homogeneous form: replicate N copies at identical offsets.
  } else if (AIE::BDDimLayoutAttr lead =
                 Traits::leadingDim(Traits::getOpDims(origOp))) {
    if (lead.getSize() > 1 && lead.getStride() != 0) {
      isArith = true;
      N = static_cast<int64_t>(lead.getSize());
      stride = static_cast<int64_t>(lead.getStride());
      llvm::ArrayRef<int64_t> off = origOp.getOffsetsAttr().asArrayRef();
      if (off.empty()) {
        // Defensive: arith canon requires non-empty offsets.  If the IR is
        // malformed, bail rather than corrupt offsets.
        return mlir::failure();
      }
      base = off[0];
    } else {
      return mlir::failure();
    }
  } else {
    return mlir::failure();
  }

  if (isArith) {
    DimsAttr origDims = Traits::getOpDims(origOp);
    Traits::setOpDims(origOp, Traits::stripOuter(ctx, origDims));
  }

  // Replicate (N-1) more copies AFTER the original op + its chain. Each
  // copy carries an empty deps list and matching attrs.
  mlir::Operation *insertAfter = origOp.getOperation();
  for (WaitAll w : *chain)
    if (w->isBeforeInBlock(insertAfter) == false &&
        w->getBlock() == origOp->getBlock() &&
        insertAfter->isBeforeInBlock(w))
      insertAfter = w.getOperation();

  for (int64_t i = 1; i < N; ++i) {
    builder.setInsertionPointAfter(insertAfter);
    mlir::IRMapping map;
    mlir::Operation *cloned = builder.clone(*origOp.getOperation(), map);
    OpTy clonedTyped = mlir::cast<OpTy>(cloned);
    if (isArith) {
      // Sliding offset: offsets[0] = base + i*stride.  Outer dim was already
      // stripped on origOp, so the clone inherits the stripped dim list —
      // no further dims work needed on the clone.
      clonedTyped.setOffsetsAttr(makeOffsetsWithHead(
          ctx, clonedTyped.getOffsetsAttr(), base + i * stride));
    }
    insertAfter = cloned;
    // Replicate the sync chain on the cloned token.
    for (WaitAll w : *chain) {
      builder.setInsertionPointAfter(insertAfter);
      auto newWa = WaitAll::create(builder, w.getLoc(),
                                   mlir::ValueRange{clonedTyped.getToken()});
      newWa.setTokenAttr(builder.getBoolAttr(w.getToken()));
      insertAfter = newWa.getOperation();
    }
  }

  // Strip the outer dim from conduit.create's producer/consumer dims when
  // the arith-progression marker fired (canon stamps both, so expand must
  // clear both).
  if (isArith) {
    DimsAttr createDims = Traits::getCreateDims(createOp);
    Traits::setCreateDims(createOp, Traits::stripOuter(ctx, createDims));
  }
  // Clear dma_repeat on the create.
  createOp.removeDmaRepeatAttr();
  return mlir::success();
}

struct ConduitExpandChannelPutsPass
    : public impl::ConduitExpandChannelPutsBase<ConduitExpandChannelPutsPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    module.walk([&](Create c) {
      // Try put-side expansion first; if not applicable, try get-side.
      if (mlir::succeeded(expandOne<PutMemrefAsync>(c, builder)))
        return;
      (void)expandOne<GetMemrefAsync>(c, builder);
    });
  }
};

} // namespace

mlir::LogicalResult expandLoopUnrollPuts(Create channel,
                                         mlir::OpBuilder &builder) {
  if (mlir::succeeded(expandOne<PutMemrefAsync>(channel, builder)))
    return mlir::success();
  return expandOne<GetMemrefAsync>(channel, builder);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitExpandChannelPutsPass() {
  return std::make_unique<ConduitExpandChannelPutsPass>();
}

} // namespace xilinx::conduit
