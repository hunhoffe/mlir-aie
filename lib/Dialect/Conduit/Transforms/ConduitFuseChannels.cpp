//===- ConduitFuseChannels.cpp - conduit-fuse-channels pass ------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --conduit-fuse-channels: DMA channel fusion for conduits on the same tile.
//
// Background
// ----------
// Each AIE compute tile exposes a small number of DMA channels (typically 2
// MM2S and 2 S2MM on AIE2).  When a single tile hosts more than two conduit
// channels, compilation fails because Pass C cannot assign unique hardware
// channels.  This is the "DMA channel exhaustion" gap documented in Track A.
//
// Theory
// ------
// Two conduits on the same producer tile can *time-share* one DMA channel
// when their memory-access windows do not overlap in program order.  More
// precisely, conduits A and B are *sequentially non-overlapping* in a basic
// block if every conduit.acquire / conduit.release / conduit.put_memref /
// conduit.get_memref op for A appears either entirely before or entirely after
// the corresponding ops for B within that block.
//
// If A's ops all precede B's ops (or vice versa), the hardware DMA engine is
// guaranteed to have finished A's transfer before it begins B's, so both
// conduits can be serviced by the same physical channel slot — reprogrammed
// between uses.
//
// This is exactly analogous to register allocation across non-overlapping live
// ranges: two variables that are not simultaneously live can share a register.
//
// Formal condition (single-block):
//   Let first(X) = min program-order index of any Conduit op for X in block B
//   Let last(X)  = max program-order index of any Conduit op for X in block B
//   A and B are fuseable iff:  last(A) < first(B)  OR  last(B) < first(A)
//
// Implementation
// --------------
// The pass:
//   1. Groups conduit.create ops by producer_tile = [col, row].
//   2. For each tile with >= 2 conduits, walks all blocks to compute per-block
//      live intervals [first, last] for each conduit name.
//   3. For each block where >= 2 conduits of this tile have activity, runs a
//      greedy linear-scan coloring to assign fuseable conduits to the same
//      group.
//   4. Annotates each conduit.create in a group of size >= 2 with:
//        fused_dma_channel_group = "groupN"
//      where N is globally unique across tiles (so Pass C can distinguish
//      groups on different tiles that happen to share an index).
//   5. Singleton groups (no fuseable partner found) are not annotated.
//
// Pass C (--conduit-to-dma) is responsible for interpreting the group
// annotation and assigning conduits in the same group to the same hardware
// DMA channel slot.  This pass is annotation-only.
//
// Limitations:
//   - Only single-block live intervals are considered.  Cross-block analysis
//     (CFG liveness) is deferred to a future M12 pass.
//   - When the same conduit name appears in multiple blocks with conflicting
//     orderings, the first block analyzed wins.  In practice each conduit's
//     ops appear in exactly one function body block, so this does not arise.
//   - Only producer-tile (MM2S) grouping is implemented; consumer-side (S2MM)
//     fusion is a symmetric future extension.
//   - The pass does not verify that the target tile has a DMA channel budget
//     deficit; that check belongs in Pass C or a resource-check pass.
//
// Run with:  aie-opt --conduit-fuse-channels <input.mlir>
//
// This pass is OPT-IN and NOT part of the default pipeline.  It is safe to
// run speculatively: it adds attributes but does not restructure the IR.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <string>
#include <utility>

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITFUSECHANNELS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Trace a !conduit.window<T> SSA value back to the channel name of its
// defining acquire or wait_window op.  Returns empty StringRef if not
// traceable (e.g., defined by a block argument or an unsupported op).
static llvm::StringRef getWindowChannelName(mlir::Value windowVal) {
  mlir::Operation *def = windowVal.getDefiningOp();
  if (!def)
    return {};
  if (auto acq = mlir::dyn_cast<Acquire>(def))
    return acq.getName();
  if (auto ww = mlir::dyn_cast<WaitWindow>(def))
    return ww.getName();
  return {};
}

// Returns the conduit channel name touched by 'op', or an empty StringRef if
// the op does not reference a named conduit.
//
// Ops with an explicit 'name' attribute: Acquire, AcquireAsync, ReleaseAsync,
//   WaitWindow, PutMemref*, GetMemref*.
// Ops without a 'name' attribute (operand is an SSA window value): Release —
//   the channel name is recovered by tracing the operand's defining op.
static llvm::StringRef getConduitOpName(mlir::Operation *op) {
  // Release takes a !conduit.window<T> operand; trace back to the acquire.
  if (auto relOp = mlir::dyn_cast<Release>(op))
    return getWindowChannelName(relOp.getWindow());

  // All other Tier 2 / Tier 3 activity ops carry an explicit 'name' attribute.
  if (mlir::isa<Acquire, AcquireAsync, ReleaseAsync, WaitWindow,
                PutMemref, GetMemref, PutMemrefAsync, GetMemrefAsync>(op))
    if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name"))
      return nameAttr.getValue();

  return {};
}

// A closed program-order interval [lo, hi] within a single basic block.
struct LiveInterval {
  unsigned lo;
  unsigned hi;
};

// Returns the closed live interval [first, last] of all Conduit ops that
// reference 'conduitName' within 'block', or nullopt if none are found.
static std::optional<LiveInterval>
computeInterval(mlir::Block *block, llvm::StringRef conduitName) {
  std::optional<unsigned> lo, hi;
  unsigned idx = 0;
  for (mlir::Operation &op : *block) {
    if (getConduitOpName(&op) == conduitName) {
      if (!lo)
        lo = idx;
      hi = idx;
    }
    ++idx;
  }
  if (!lo)
    return std::nullopt;
  return LiveInterval{*lo, *hi};
}

// ---------------------------------------------------------------------------
// Per-tile fusion state
// ---------------------------------------------------------------------------

struct ConduitInfo {
  std::string name;
  Create createOp;
};

// Greedy linear-scan interval coloring.
//
// Sorts 'items' in place by interval start, then assigns each to the
// lowest-numbered group whose last-assigned interval ended strictly before
// this interval starts.  Opens a new group when no existing one qualifies.
//
// The greedy scan is optimal for interval graphs (which are perfect graphs):
// the number of groups equals the maximum clique size (maximum concurrency).
//
// Returns a SmallVector of group IDs in the post-sort order of 'items'.
static llvm::SmallVector<unsigned>
assignGroups(llvm::SmallVectorImpl<std::pair<std::string, LiveInterval>> &items) {
  std::sort(items.begin(), items.end(),
            [](const auto &a, const auto &b) { return a.second.lo < b.second.lo; });

  llvm::SmallVector<unsigned> groups(items.size());
  // groupEnd[g] = hi of the last interval assigned to group g.
  llvm::SmallVector<unsigned> groupEnd;

  for (unsigned i = 0; i < items.size(); ++i) {
    unsigned lo = items[i].second.lo;
    // Find the first group whose last interval ends strictly before lo.
    unsigned bestGroup = groupEnd.size(); // sentinel: open a new group
    for (unsigned g = 0; g < groupEnd.size(); ++g) {
      if (groupEnd[g] < lo) {
        bestGroup = g;
        break;
      }
    }
    if (bestGroup == groupEnd.size())
      groupEnd.push_back(items[i].second.hi);
    else
      groupEnd[bestGroup] = items[i].second.hi;
    groups[i] = bestGroup;
  }
  return groups;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct ConduitFuseChannelsPass
    : public impl::ConduitFuseChannelsBase<ConduitFuseChannelsPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Step 1: collect conduit.create ops grouped by producer_tile [col, row].
    // DenseMapInfo for std::pair<int64_t,int64_t> is provided by LLVM.
    llvm::DenseMap<std::pair<int64_t, int64_t>,
                   llvm::SmallVector<ConduitInfo, 4>>
        tileGroups;

    module.walk([&](Create createOp) {
      auto tileAttr = createOp.getProducerTile();
      if (!tileAttr || tileAttr->size() != 2)
        return;
      int64_t col = (*tileAttr)[0];
      int64_t row = (*tileAttr)[1];
      // Shim tiles (row == 0) use a separate DMA model; exclude them.
      if (row == 0)
        return;
      tileGroups[{col, row}].push_back({createOp.getName().str(), createOp});
    });

    // Group IDs are globally unique across tiles so that Pass C can
    // distinguish "group0 on tile [0,2]" from "group0 on tile [1,2]".
    unsigned nextGroupId = 0;

    // Step 2: for each tile with >= 2 conduits, attempt fusion.
    for (auto &[tile, conduits] : tileGroups) {
      if (conduits.size() < 2)
        continue;

      // Build a map from each basic block to the (name, interval) pairs for
      // conduits of this tile that have activity in that block.
      llvm::DenseMap<mlir::Block *,
                     llvm::SmallVector<std::pair<std::string, LiveInterval>, 4>>
          blockConduits;

      module.walk([&](mlir::Block *block) {
        for (auto &ci : conduits) {
          auto iv = computeInterval(block, ci.name);
          if (!iv)
            continue;
          blockConduits[block].push_back({ci.name, *iv});
        }
      });

      // For each block where >= 2 conduits of this tile appear, run greedy
      // interval coloring and record name → group assignments.
      llvm::StringMap<unsigned> nameToGroup;

      for (auto &[block, items] : blockConduits) {
        if (items.size() < 2)
          continue;

        llvm::SmallVector<std::pair<std::string, LiveInterval>, 4> sortable(
            items.begin(), items.end());
        llvm::SmallVector<unsigned> groupIds = assignGroups(sortable);

        // Record assignments; first-block-wins if a name appears in multiple
        // blocks (see Limitations in file header).
        for (unsigned i = 0; i < sortable.size(); ++i) {
          llvm::StringRef n = sortable[i].first;
          if (!nameToGroup.count(n))
            nameToGroup[n] = nextGroupId + groupIds[i];
        }

        unsigned maxGroup = *std::max_element(groupIds.begin(), groupIds.end());
        nextGroupId += maxGroup + 1;
      }

      // Only annotate groups with >= 2 members; singleton groups have no
      // fusion partner and annotating them would mislead Pass C.
      llvm::DenseMap<unsigned, unsigned> groupCount;
      for (auto &[name, gid] : nameToGroup)
        ++groupCount[gid];

      for (auto &ci : conduits) {
        auto it = nameToGroup.find(ci.name);
        if (it == nameToGroup.end())
          continue;
        unsigned gid = it->second;
        if (groupCount[gid] < 2)
          continue;
        std::string label = "group" + std::to_string(gid);
        ci.createOp->setAttr(
            "fused_dma_channel_group",
            mlir::StringAttr::get(module.getContext(), label));
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitFuseChannelsPass() {
  return std::make_unique<ConduitFuseChannelsPass>();
}

} // namespace xilinx::conduit
