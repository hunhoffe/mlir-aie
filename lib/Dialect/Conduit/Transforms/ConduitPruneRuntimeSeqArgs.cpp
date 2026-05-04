//===- ConduitPruneRuntimeSeqArgs.cpp - prune dead runtime_seq args -*-C++-*-=//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --conduit-prune-runtime-seq-args: drop dead block arguments from
// aie.runtime_sequence ops after operator/channel fusion has erased
// intermediate channels.
//
// Background
// ----------
// IRON emits one aie.device per operator, each with its own
// aie.runtime_sequence whose block arguments are bound positionally to host
// kernel-call arguments (via aie.shim_dma_allocation symbols referenced by
// conduit.put_memref / conduit.get_memref ops, with each op carrying an
// `arg_index` attribute that names which seq arg it binds to).
//
// --aie-combine-device{same-tile=true} merges multiple devices' runtime
// sequences by appending devB's args + body to devA's seq, renumbering devB's
// arg_index attrs by + len(devA.args).  --conduit-fuse-core-bodies and
// --conduit-fuse-operators then erase the intermediate-channel
// put/get_memref_async ops + their wait_all chains, leaving the merged
// runtime_sequence with N declared args but only M < N referenced by surviving
// conduit ops.  The unreferenced slots are dead — if they remain, the host's
// positional set_arg(i, bo) calls will land at the wrong runtime arg index
// and surviving ops will read undefined memory.
//
// This pass reconciles seq's block arguments with the surviving
// conduit.put_memref{,_async} / conduit.get_memref{,_async} ops:
//   1. Walk seq body, collect set<int64_t> of arg_index values referenced.
//   2. If every declared arg is referenced, no-op early exit.
//   3. Otherwise, build a renumber map (sorted live old idx → 0..M-1).
//   4. Rewrite each conduit op's `arg_index` attr per the map.
//   5. Erase dead block args in reverse order via Block::eraseArgument.
//
// aie.runtime_sequence has NO separate function_type attribute — args are
// pure block arguments of the body region (see AIEDialect.cpp parse/print),
// so erasing block args is the only signature update needed.
//
// Pipeline placement (aiecc.cpp, --use-conduit branch): AFTER
// `--conduit-fuse-channels` and BEFORE `--conduit-depth-promote,
// --conduit-to-dma`. Running last among the fusion passes means it picks up
// dead args introduced by any earlier fusion step regardless of which one
// created them.
//
// Run with:  aie-opt --conduit-prune-runtime-seq-args <input.mlir>
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITPRUNERUNTIMESEQARGS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// Collect arg_index from a single conduit op (put/get_memref, async or sync).
// Returns std::nullopt if the op has no `arg_index` attribute (which is valid
// for hand-written tests / pre-dma-task-to-conduit IR — leave such ops alone).
static std::optional<int64_t> getArgIndex(mlir::Operation *op) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>("arg_index");
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

// Update a single conduit op's arg_index from oldIdx to newIdx.
static void setArgIndex(mlir::Operation *op, int64_t newIdx) {
  mlir::OpBuilder builder(op->getContext());
  op->setAttr("arg_index", builder.getI64IntegerAttr(newIdx));
}

// Returns true iff op is a conduit memref-DMA op carrying an arg_index attr.
static bool isConduitMemrefDmaOp(mlir::Operation *op) {
  return mlir::isa<PutMemref, GetMemref, PutMemrefAsync, GetMemrefAsync>(op);
}

static void pruneOneRuntimeSeq(xilinx::AIE::RuntimeSequenceOp seq) {
  mlir::Region &body = seq.getRegion();
  if (body.empty())
    return;
  mlir::Block &block = body.front();
  unsigned numArgs = block.getNumArguments();
  if (numArgs == 0)
    return;

  // ------------------------------------------------------------------------
  // Step 1: collect live arg indices.  An arg is "live" if EITHER:
  //   (a) a surviving conduit memref-DMA op references it via arg_index attr
  //       (positional binding — these are what we renumber), OR
  //   (b) any op in the body holds an SSA use of the BlockArgument value
  //       (e.g., bare-conduit IR where aiex.npu.dma_memcpy_nd consumes
  //       `%in` / `%out` directly without going through arg_index, or any
  //       future non-conduit op that references seq args by SSA).
  //
  // Erasing a block arg with surviving SSA uses trips the
  // "Cannot destroy a value that still has uses!" assertion in
  // IRObjectWithUseList::~IRObjectWithUseList; (b) is required for safety.
  // ------------------------------------------------------------------------
  llvm::SmallVector<mlir::Operation *, 16> conduitOps;
  llvm::SmallSet<int64_t, 16> liveSet;
  block.walk([&](mlir::Operation *op) {
    if (!isConduitMemrefDmaOp(op))
      return;
    auto idx = getArgIndex(op);
    if (!idx)
      return;
    conduitOps.push_back(op);
    liveSet.insert(*idx);
  });
  for (unsigned i = 0; i < numArgs; ++i) {
    if (!block.getArgument(i).use_empty())
      liveSet.insert(static_cast<int64_t>(i));
  }

  // ------------------------------------------------------------------------
  // Step 2: bail out cheaply if every declared arg is already referenced.
  // ------------------------------------------------------------------------
  if (liveSet.size() == numArgs)
    return;

  // ------------------------------------------------------------------------
  // Step 3: build renumber map (sorted live old idx → 0..M-1).
  //
  // Sorting preserves the source-order intent: the surviving args keep the
  // same relative ordering they had pre-prune, just packed dense from 0.
  // ------------------------------------------------------------------------
  llvm::SmallVector<int64_t, 16> liveSorted(liveSet.begin(), liveSet.end());
  llvm::sort(liveSorted);

  // Sanity: every live index must be a valid block-arg position.  If a
  // conduit op references an out-of-range arg_index, that signals a bug
  // upstream of this pass — surface it loudly rather than silently masking.
  for (int64_t idx : liveSorted) {
    if (idx < 0 || static_cast<unsigned>(idx) >= numArgs) {
      seq.emitError("conduit-prune-runtime-seq-args: arg_index ")
          << idx << " out of range for runtime_sequence with " << numArgs
          << " args (upstream pipeline bug)";
      return;
    }
  }

  llvm::DenseMap<int64_t, int64_t> renumber;
  for (size_t newIdx = 0; newIdx < liveSorted.size(); ++newIdx)
    renumber[liveSorted[newIdx]] = static_cast<int64_t>(newIdx);

  // ------------------------------------------------------------------------
  // Step 4: rewrite each conduit op's arg_index per the map.
  // ------------------------------------------------------------------------
  for (mlir::Operation *op : conduitOps) {
    auto oldIdx = getArgIndex(op);
    assert(oldIdx && "conduitOps was filtered to ops with arg_index attrs");
    auto it = renumber.find(*oldIdx);
    assert(it != renumber.end() && "live arg missing from renumber map");
    if (it->second != *oldIdx)
      setArgIndex(op, it->second);
  }

  // ------------------------------------------------------------------------
  // Step 5: erase dead block args in reverse order so earlier indices stay
  // valid as later ones are removed.
  // ------------------------------------------------------------------------
  for (int64_t i = static_cast<int64_t>(numArgs) - 1; i >= 0; --i) {
    if (liveSet.contains(i))
      continue;
    block.eraseArgument(static_cast<unsigned>(i));
  }
}

struct ConduitPruneRuntimeSeqArgsPass
    : public impl::ConduitPruneRuntimeSeqArgsBase<
          ConduitPruneRuntimeSeqArgsPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    module.walk([&](xilinx::AIE::DeviceOp deviceOp) {
      deviceOp.walk(
          [&](xilinx::AIE::RuntimeSequenceOp seq) { pruneOneRuntimeSeq(seq); });
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitPruneRuntimeSeqArgsPass() {
  return std::make_unique<ConduitPruneRuntimeSeqArgsPass>();
}

} // namespace xilinx::conduit
