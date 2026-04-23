//===- ConduitDmaTaskToConduit.cpp - dma-task-to-conduit pass ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --dma-task-to-conduit: Convert IRON DMATask ops in aie.runtime_sequence
// into Conduit IR Tier 3 memref-DMA ops.
//
// After Pass A (--objectfifo-to-conduit) runs, the aie.runtime_sequence body
// still contains raw AIEX DMA task ops that program the shim DMA:
//
//   %t = aiex.dma_configure_task_for @chan_shim_alloc {
//     aie.dma_bd(%buf : memref<...>, offset, len, [dims]) {...}
//     aie.end
//   }
//   aiex.dma_start_task(%t)
//   aiex.dma_await_task(%t)
//   aiex.dma_free_task(%t)
//
// This pass replaces them with high-level conduit ops:
//
//   MM2S (shim sends to device) → conduit.put_memref {name = @chan, ...}
//   S2MM (shim receives from device) → conduit.get_memref {name = @chan, ...}
//
// Direction is determined by looking up the aie.shim_dma_allocation that
// the dma_configure_task_for references via its $alloc attribute.
//
// The buffer descriptor parameters (offset, len, dimensions) from the
// aie.dma_bd inside the task body are mapped to the conduit op attributes:
//   num_elems ← len
//   offsets/sizes/strides ← derived from BDDimLayout dimensions
//   producer_dimensions ← raw BDDimLayout (MM2S)
//   consumer_dimensions ← raw BDDimLayout wrapped in a singleton array of
//                         arrays (S2MM)
//   arg_index           ← BlockArgument index of `aie.dma_bd(%argN, …)` in
//                         the enclosing aie.runtime_sequence body
//
// arg_index encodes the EXPLICIT block-arg binding so the reverse rebuild
// in --conduit-to-dma Step 8g does not need to guess via offset heuristics
// (FS7 fix: the heuristic mis-bound output ops with non-zero `0xDEADBEE0`
// patch markers to the input arg).
//
// All dma_start_task, dma_await_task, and dma_free_task ops are erased.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITDMATASKTOCONDUIT
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helper: extract the FlatSymbolRefAttr "alloc" from a dma_configure_task_for
// op (which may be registered or unregistered).
// ---------------------------------------------------------------------------
static mlir::FlatSymbolRefAttr getAllocAttr(mlir::Operation *op) {
  // Try inherent attr first (registered ops store properties this way).
  if (auto optAttr = op->getInherentAttr("alloc"))
    if (auto flat = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(*optAttr))
      return flat;

  // Fallback: discardable attr dict (unregistered ops).
  if (auto flat = op->getAttrOfType<mlir::FlatSymbolRefAttr>("alloc"))
    return flat;

  return {};
}

// ---------------------------------------------------------------------------
// Helper: convert BDDimLayout dimensions into offsets/sizes/strides arrays.
//
// Conduit Tier-3 memref-DMA ops have a verifier requirement that
// `num_elems == product(sizes)` — i.e. `sizes` must describe the unique
// addressable elements that one BD execution transfers (matching the BD's
// `len` parameter).
//
// A `dma_bd`'s `dimensions` array, however, can include extra OUTER dims
// that act as REPEAT factors rather than addressable iteration dims.
// They come in two flavors:
//
//   1. `<size=N, stride=0>` (broadcast). N==1 is a trivial filler; N>1 is
//      an explicit broadcast (read the same address N times). FS5 fix.
//   2. `<size=N, stride=S>` with S != 0, where the iteration wraps over
//      memory already visited by inner dims. Detected by:
//        product(sizes) > len  (channel-element count > BD-pass count).
//      Llama runlist emits this combined with `repeat_count` on the
//      enclosing `aiex.dma_configure_task_for`. FS5 extension fix.
//
// Both flavors are stripped from `sizes/strides`. The full BDDimLayout
// (including the stripped outer dims) is preserved separately:
//   * MM2S — propagated to `producer_dimensions` on `conduit.put_memref`,
//     which downstream MM2S DMA programming consumes for the actual BD
//     chain.
//   * S2MM — the source `repeat_count` attr is the channel-side encoding;
//     `consumer_dimensions` is left null (matching pre-fix behavior).
//
// Outer-dim peeling rules (defensive — only peel when the residual is
// well-defined):
//   * Stop when `product(sizes) == len` (no more repeat factors).
//   * Skip when `len <= 0` (no target to peel against).
//   * Stop if peeling the outermost would drop product BELOW `len`
//     (something else is off — surface via the verifier rather than
//     silently mangle the BD).
//
// The dma_bd scalar offset becomes offsets[0].
// ---------------------------------------------------------------------------
static void dimsToOffsetsStrides(int32_t bdOffset, int64_t len,
                                 AIE::BDDimLayoutArrayAttr dimensions,
                                 llvm::SmallVectorImpl<int64_t> &offsets,
                                 llvm::SmallVectorImpl<int64_t> &sizes,
                                 llvm::SmallVectorImpl<int64_t> &strides) {
  if (dimensions && !dimensions.empty()) {
    for (auto dim : dimensions.getValue()) {
      auto bdDim = mlir::cast<AIE::BDDimLayoutAttr>(dim);
      // Skip all stride=0 dims (broadcast/repeat or trivial filler).
      if (bdDim.getStride() == 0)
        continue;
      sizes.push_back(static_cast<int64_t>(bdDim.getSize()));
      strides.push_back(static_cast<int64_t>(bdDim.getStride()));
    }
  }

  // Peel outer non-zero-stride REPEAT dims while product(sizes) > len.
  // (BDDimLayout convention: index 0 is outermost.)
  if (len > 0 && sizes.size() > 1) {
    int64_t prod = 1;
    for (int64_t s : sizes)
      prod *= s;
    while (sizes.size() > 1 && prod > len) {
      int64_t outer = sizes.front();
      if (outer <= 0)
        break;
      int64_t next = prod / outer;
      // Don't peel if the residual would no longer cover `len`; let the
      // verifier surface the inconsistency rather than mangling the BD.
      if (next < len)
        break;
      sizes.erase(sizes.begin());
      strides.erase(strides.begin());
      prod = next;
    }
  }

  // Ensure at least one dimension.
  if (sizes.empty()) {
    sizes.push_back(len > 0 ? len : 1);
    strides.push_back(1);
  }

  // Offsets: [bdOffset, 0, 0, ...] matching sizes rank.
  offsets.push_back(static_cast<int64_t>(bdOffset));
  for (size_t i = 1; i < sizes.size(); ++i)
    offsets.push_back(0);
}

// ---------------------------------------------------------------------------
// Info extracted from a shim_dma_allocation for a given channel.
// ---------------------------------------------------------------------------
struct ShimAllocInfo {
  AIE::DMAChannelDir dir;
  llvm::StringRef conduitName; // actual conduit channel name
};

// ---------------------------------------------------------------------------
// Main pass.
// ---------------------------------------------------------------------------
struct ConduitDmaTaskToConduitPass
    : impl::ConduitDmaTaskToConduitBase<ConduitDmaTaskToConduitPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());

    module.walk([&](AIE::DeviceOp device) { processDevice(device, builder); });
  }

private:
  void processDevice(AIE::DeviceOp device, mlir::OpBuilder &builder) {
    // Phase 1: Build a map from shim_dma_allocation symbol name to direction
    // and conduit channel name.
    llvm::StringMap<ShimAllocInfo> allocMap;
    device.walk([&](AIE::ShimDMAAllocationOp alloc) {
      auto conduitChannelAttr =
          alloc->getAttrOfType<mlir::FlatSymbolRefAttr>("conduit_channel");
      llvm::StringRef conduitName = conduitChannelAttr
                                        ? conduitChannelAttr.getValue()
                                        : alloc.getSymName();
      allocMap[alloc.getSymName()] = {alloc.getChannelDir(), conduitName};
    });

    if (allocMap.empty())
      return;

    // Phase 2: Walk runtime_sequence ops and convert DMA task ops.
    device.walk([&](AIE::RuntimeSequenceOp rtSeq) {
      processRuntimeSequence(rtSeq, allocMap, builder);
    });
  }

  void processRuntimeSequence(AIE::RuntimeSequenceOp rtSeq,
                              const llvm::StringMap<ShimAllocInfo> &allocMap,
                              mlir::OpBuilder &builder) {

    mlir::MLIRContext *ctx = rtSeq.getContext();

    // The runtime_sequence's body block — used to verify that the dma_bd's
    // BlockArgument actually belongs to THIS runtime_sequence (not some
    // unrelated nested op).
    mlir::Block *rtSeqBlock = &rtSeq.getBody().front();

    // Collect ops to erase, separated by kind for correct erase ordering.
    llvm::SmallVector<mlir::Operation *> configOps;
    llvm::SmallVector<mlir::Operation *> userOps; // start/await/free

    rtSeq.walk([&](mlir::Operation *op) {
      llvm::StringRef opName = op->getName().getStringRef();

      if (opName == "aiex.dma_configure_task_for") {
        // --- Convert dma_configure_task_for → conduit.put/get_memref ---
        auto allocAttr = getAllocAttr(op);
        if (!allocAttr)
          return;

        auto it = allocMap.find(allocAttr.getValue());
        if (it == allocMap.end())
          return;

        AIE::DMAChannelDir dir = it->second.dir;
        llvm::StringRef conduitName = it->second.conduitName;

        // Walk the task body to find aie.dma_bd.
        AIE::DMABDOp dmaBd = nullptr;
        op->walk([&](AIE::DMABDOp bd) { dmaBd = bd; });
        if (!dmaBd)
          return;

        // Extract BD parameters.
        int32_t bdOffset = dmaBd.getOffset();
        std::optional<int32_t> lenOpt = dmaBd.getLen();
        int64_t len = lenOpt ? static_cast<int64_t>(*lenOpt) : 0;
        AIE::BDDimLayoutArrayAttr dimensions = dmaBd.getDimensionsAttr();
        mlir::Value buffer = dmaBd.getBuffer();

        // If len is 0 and we have a shaped memref, compute from shape.
        if (len == 0) {
          auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(buffer.getType());
          if (memrefTy && memrefTy.hasStaticShape())
            len = memrefTy.getNumElements();
        }

        // Resolve the BlockArgument that this BD binds to.  FS7: the
        // round-trip from put/get_memref back to dma_configure_task_for in
        // --conduit-to-dma Step 8g previously relied on an offset==0
        // heuristic to group ops by block arg.  That heuristic mis-binds any
        // op whose offsets[0] is a non-zero patch marker (e.g. `0xDEADBEE0`).
        // We instead capture the binding explicitly here and propagate it
        // via the new arg_index attribute.  Fail loudly if the buffer is not
        // a direct block arg of the enclosing aie.runtime_sequence — that
        // signals an upstream IR shape this pass never validated against.
        auto bufBlockArg = mlir::dyn_cast<mlir::BlockArgument>(buffer);
        if (!bufBlockArg || bufBlockArg.getOwner() != rtSeqBlock) {
          dmaBd.emitError(
              "dma-task-to-conduit: aie.dma_bd buffer must be a direct "
              "BlockArgument of the enclosing aie.runtime_sequence "
              "(arg_index binding required for round-trip to "
              "aiex.dma_configure_task_for in --conduit-to-dma Step 8g)");
          signalPassFailure();
          return;
        }
        auto argIndexAttr = mlir::IntegerAttr::get(
            mlir::IntegerType::get(ctx, 64),
            static_cast<int64_t>(bufBlockArg.getArgNumber()));

        // Build offsets/sizes/strides from dimensions.
        llvm::SmallVector<int64_t> offsets, sizes, strides;
        dimsToOffsetsStrides(bdOffset, len, dimensions, offsets, sizes,
                             strides);

        // Emit conduit.put_memref or conduit.get_memref.
        builder.setInsertionPoint(op);
        auto nameAttr = mlir::FlatSymbolRefAttr::get(ctx, conduitName);
        auto numElemsAttr =
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), len);
        auto offsetsAttr = mlir::DenseI64ArrayAttr::get(ctx, offsets);
        auto sizesAttr = mlir::DenseI64ArrayAttr::get(ctx, sizes);
        auto stridesAttr = mlir::DenseI64ArrayAttr::get(ctx, strides);

        if (dir == AIE::DMAChannelDir::MM2S) {
          // MM2S: shim sends data into the conduit → put_memref.
          // Pass through the BDDimLayout as producer_dimensions.
          PutMemref::create(builder, op->getLoc(), nameAttr, numElemsAttr,
                            offsetsAttr, sizesAttr, stridesAttr, dimensions,
                            argIndexAttr);
        } else {
          // S2MM: shim receives data from the conduit → get_memref.
          // Wrap the single BDDimLayout as a singleton-of-singleton
          // BDDimLayoutArrayArrayAttr so the round-trip rebuild in
          // --conduit-to-dma Step 8g can recover the full S2MM strided
          // write geometry (FS7 sub-fix: previously dropped to nullptr,
          // which lost StridedCopy's e.g. 8×131072 scatter pattern).
          AIE::BDDimLayoutArrayArrayAttr consumerDims;
          if (dimensions && !dimensions.empty())
            consumerDims = AIE::BDDimLayoutArrayArrayAttr::get(ctx, dimensions);
          GetMemref::create(builder, op->getLoc(), nameAttr, numElemsAttr,
                            offsetsAttr, sizesAttr, stridesAttr, consumerDims,
                            argIndexAttr);
        }

        configOps.push_back(op);
      } else if (opName == "aiex.dma_start_task" ||
                 opName == "aiex.dma_await_task" ||
                 opName == "aiex.dma_free_task") {
        userOps.push_back(op);
      }
    });

    // Erase user ops first (they reference the configure op results),
    // then configure ops.
    for (auto *op : userOps)
      op->erase();
    for (auto *op : configOps)
      op->erase();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitDmaTaskToConduitPass() {
  return std::make_unique<ConduitDmaTaskToConduitPass>();
}

} // namespace xilinx::conduit
