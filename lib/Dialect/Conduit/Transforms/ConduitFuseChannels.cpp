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
//        dma_channel_group = "groupN"
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
//   - Both producer-tile (MM2S) and consumer-tile (S2MM) grouping are
//     implemented.  S2MM fusion annotates with dma_channel_group_s2mm.
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

#include "ConduitTileInference.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  if (mlir::isa<Acquire, AcquireAsync, ReleaseAsync, WaitWindow, PutMemref,
                GetMemref, PutMemrefAsync, GetMemrefAsync>(op))
    if (auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name"))
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

// Compute nesting depth of a block (number of ancestor operations).
// Used to sort blocks shallowest-first for deterministic fusion ordering.
static unsigned getBlockDepth(mlir::Block *block) {
  unsigned depth = 0;
  mlir::Operation *parent = block->getParentOp();
  while (parent) {
    ++depth;
    parent = parent->getParentOp();
  }
  return depth;
}

// ---------------------------------------------------------------------------
// Per-tile fusion state
// ---------------------------------------------------------------------------

struct ConduitInfo {
  std::string name;
  Create createOp;
};

// Returns the effective dma_repeat for a Create op.  Absent attribute
// defaults to 1 (DMA fires exactly once per dispatch).
static int64_t getEffectiveDmaRepeat(Create createOp) {
  if (auto rep = createOp.getDmaRepeat())
    return static_cast<int64_t>(*rep);
  return 1;
}

// Validates that all members of the same channel group share the same
// effective dma_repeat value.  Channels with mismatched dma_repeat
// fundamentally cannot share a hardware channel slot because they fire
// BDs at different rates per dispatch.
//
// 'kindLabel' is "MM2S" or "S2MM" for diagnostic clarity.
//
// Returns true if all groups are compatible; false (and emits an error
// on the offending op) if any group has a mismatch.
static bool checkGroupDmaRepeatCompatibility(
    llvm::StringRef kindLabel, llvm::SmallVectorImpl<ConduitInfo> &conduits,
    const llvm::StringMap<unsigned> &nameToGroup,
    const llvm::DenseMap<unsigned, unsigned> &groupCount) {
  llvm::DenseMap<unsigned, int64_t> groupRepeat;
  bool ok = true;
  for (auto &ci : conduits) {
    auto it = nameToGroup.find(ci.name);
    if (it == nameToGroup.end())
      continue;
    unsigned gid = it->second;
    auto countIt = groupCount.find(gid);
    if (countIt == groupCount.end() || countIt->second < 2)
      continue;
    int64_t rep = getEffectiveDmaRepeat(ci.createOp);
    auto gIt = groupRepeat.find(gid);
    if (gIt == groupRepeat.end()) {
      groupRepeat[gid] = rep;
    } else if (gIt->second != rep) {
      ci.createOp.emitError()
          << "fuse-channels: cannot group channels with mismatched "
             "dma_repeat values "
          << gIt->second << " vs " << rep << " (" << kindLabel << " group)";
      ok = false;
    }
  }
  return ok;
}

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
static llvm::SmallVector<unsigned> assignGroups(
    llvm::SmallVectorImpl<std::pair<std::string, LiveInterval>> &items) {
  std::sort(items.begin(), items.end(), [](const auto &a, const auto &b) {
    return a.second.lo < b.second.lo;
  });

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

    auto extractCoord = [](mlir::Value tileVal) -> std::pair<int64_t, int64_t> {
      if (auto tileOp = tileVal.getDefiningOp<AIE::TileOp>())
        return {static_cast<int64_t>(tileOp.getCol()),
                static_cast<int64_t>(tileOp.getRow())};
      return {-1, -1};
    };

    // Group IDs are globally unique across tiles and devices so that Pass C
    // can distinguish groups.
    unsigned nextGroupId = 0;

    // Process each aie.device independently to avoid cross-device name
    // collisions.  When two aie.device blocks share channel names (e.g.
    // flash_attn_air_channel), module-scope walks would create duplicate
    // ConduitInfo entries, causing premature singleton assignments and DMA
    // channel exhaustion.
    module.walk([&](AIE::DeviceOp deviceOp) {
      // Infer tile coordinates scoped to this device.
      auto inferredMap = inferAllTiles(deviceOp);

      // ---------------------------------------------------------------------
      // MM2S Live-Interval Fusion
      //
      // Groups conduit.create ops by producer tile and fuses them using greedy
      // interval coloring when their live intervals are non-overlapping in a
      // basic block.
      // ---------------------------------------------------------------------

      // Step 1: collect conduit.create ops grouped by producer tile [col, row].
      llvm::DenseMap<std::pair<int64_t, int64_t>,
                     llvm::SmallVector<ConduitInfo, 4>>
          tileGroups;

      deviceOp.walk([&](Create createOp) {
        int64_t col = -1, row = -1;
        auto tileIt = inferredMap.find(createOp.getName().str());
        if (tileIt != inferredMap.end() && tileIt->second.producerTile) {
          std::tie(col, row) = extractCoord(tileIt->second.producerTile);
        }
        // Shim tiles (row == 0) use a separate DMA model; exclude them.
        if (col >= 0 && row == 0)
          return;
        // Unknown tile: place in a default group {-1, -1}.
        tileGroups[{col, row}].push_back({createOp.getName().str(), createOp});
      });

      // Step 2: for each tile with >= 2 conduits, attempt fusion.
      for (auto &[tile, conduits] : tileGroups) {
        if (conduits.size() < 2)
          continue;

        // Build a map from each basic block to the (name, interval) pairs for
        // conduits of this tile that have activity in that block.
        llvm::DenseMap<
            mlir::Block *,
            llvm::SmallVector<std::pair<std::string, LiveInterval>, 4>>
            blockConduits;

        deviceOp.walk([&](mlir::Block *block) {
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

        // Track whether any block hosting a conduit has an scf::IfOp parent.
        llvm::StringMap<bool> nameNeedsRuntime;

        // Sort blocks by nesting depth (shallowest first) for deterministic
        // iteration order.  The "first-block-wins" semantics of nameToGroup
        // depend on processing order; non-deterministic DenseMap iteration
        // causes outer-block non-overlapping intervals to be shadowed by
        // inner-block overlapping intervals in some devices but not others.
        llvm::SmallVector<mlir::Block *> sortedBlocks;
        for (auto &kv : blockConduits)
          sortedBlocks.push_back(kv.first);
        llvm::sort(sortedBlocks, [](mlir::Block *a, mlir::Block *b) {
          return getBlockDepth(a) < getBlockDepth(b);
        });

        for (mlir::Block *block : sortedBlocks) {
          auto &items = blockConduits[block];
          bool inIfBlock =
              mlir::isa_and_present<mlir::scf::IfOp>(block->getParentOp());
          for (auto &[name, iv] : items) {
            if (inIfBlock)
              nameNeedsRuntime[name] = true;
            else if (!nameNeedsRuntime.count(name))
              nameNeedsRuntime[name] = false;
          }

          if (items.size() < 2)
            continue;

          llvm::SmallVector<std::pair<std::string, LiveInterval>, 4> sortable(
              items.begin(), items.end());
          llvm::SmallVector<unsigned> groupIds = assignGroups(sortable);

          for (unsigned i = 0; i < sortable.size(); ++i) {
            llvm::StringRef n = sortable[i].first;
            if (!nameToGroup.count(n))
              nameToGroup[n] = nextGroupId + groupIds[i];
          }

          unsigned maxGroup =
              *std::max_element(groupIds.begin(), groupIds.end());
          nextGroupId += maxGroup + 1;
        }

        // Build a set of conduit names that have Tier 3 ops (put_memref /
        // get_memref) for this tile group.
        llvm::StringMap<bool> nameIsTier3;
        for (auto &ci : conduits) {
          bool hasTier3 = false;
          deviceOp.walk([&](mlir::Operation *op) {
            if (!mlir::isa<PutMemref, GetMemref, PutMemrefAsync,
                           GetMemrefAsync>(op))
              return;
            auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
            if (nameAttr && nameAttr.getValue() == ci.name)
              hasTier3 = true;
          });
          nameIsTier3[ci.name] = hasTier3;
        }

        llvm::DenseMap<unsigned, unsigned> groupCount;
        for (auto &[name, gid] : nameToGroup)
          ++groupCount[gid];

        llvm::DenseMap<unsigned, bool> groupNeedsRuntime;
        for (auto &[name, gid] : nameToGroup) {
          auto it = nameNeedsRuntime.find(name);
          if (it != nameNeedsRuntime.end() && it->second)
            groupNeedsRuntime[gid] = true;
          else if (!groupNeedsRuntime.count(gid))
            groupNeedsRuntime[gid] = false;
        }

        // dma_repeat compatibility: members of the same group must share
        // the same effective dma_repeat (absent = 1).  Mismatches are
        // unsafe at the hardware level — separate channels fire BDs at
        // different rates and cannot share a slot.
        if (!checkGroupDmaRepeatCompatibility("MM2S", conduits, nameToGroup,
                                              groupCount)) {
          signalPassFailure();
          continue;
        }

        for (auto &ci : conduits) {
          auto it = nameToGroup.find(ci.name);
          if (it == nameToGroup.end())
            continue;
          unsigned gid = it->second;
          if (groupCount[gid] < 2)
            continue;

          if (nameIsTier3.count(ci.name) && nameIsTier3[ci.name]) {
            int64_t depth = 1;
            if (auto depthOpt = ci.createOp.getDepth())
              depth = *depthOpt;
            if (depth > 1) {
              ci.createOp.emitRemark()
                  << "conduit-fuse-channels: skipping '" << ci.name
                  << "' — Tier 3 channel with depth>1 not supported in fuse "
                     "groups";
              continue;
            }
          }

          std::string label = "group" + std::to_string(gid);
          mlir::MLIRContext *ctx = module.getContext();
          ci.createOp->setAttr("dma_channel_group",
                               mlir::StringAttr::get(ctx, label));
          llvm::StringRef fuseMode =
              groupNeedsRuntime[gid] ? "runtime" : "static";
          ci.createOp->setAttr("fuse_mode",
                               mlir::StringAttr::get(ctx, fuseMode));
        }
      }

      // ---------------------------------------------------------------------
      // S2MM Live-Interval Fusion
      //
      // Symmetric to MM2S: groups conduit.create ops by consumer tile and
      // fuses them using greedy interval coloring when their live intervals
      // are non-overlapping in a basic block.
      // ---------------------------------------------------------------------

      // Step 1: collect conduit.create ops grouped by consumer tile.
      llvm::DenseMap<std::pair<int64_t, int64_t>,
                     llvm::SmallVector<ConduitInfo, 4>>
          s2mmTileGroups;

      deviceOp.walk([&](Create createOp) {
        auto tileIt = inferredMap.find(createOp.getName().str());
        if (tileIt == inferredMap.end())
          return;
        for (auto consTile : tileIt->second.consumerTiles) {
          auto [col, row] = extractCoord(consTile);
          if (col >= 0 && (row == 0 || row == 1))
            continue;
          s2mmTileGroups[{col, row}].push_back(
              {createOp.getName().str(), createOp});
        }
      });

      // Step 2: for each consumer tile with >= 2 conduits, attempt fusion.
      for (auto &[tile, conduits] : s2mmTileGroups) {
        if (conduits.size() < 2)
          continue;

        llvm::DenseMap<
            mlir::Block *,
            llvm::SmallVector<std::pair<std::string, LiveInterval>, 4>>
            blockConduits;

        deviceOp.walk([&](mlir::Block *block) {
          for (auto &ci : conduits) {
            auto iv = computeInterval(block, ci.name);
            if (!iv)
              continue;
            blockConduits[block].push_back({ci.name, *iv});
          }
        });

        llvm::StringMap<unsigned> nameToGroup;
        llvm::StringMap<bool> nameNeedsRuntime;

        // Sort blocks by nesting depth (shallowest first) — same fix as
        // MM2S above for deterministic "first-block-wins" ordering.
        llvm::SmallVector<mlir::Block *> sortedBlocks;
        for (auto &kv : blockConduits)
          sortedBlocks.push_back(kv.first);
        llvm::sort(sortedBlocks, [](mlir::Block *a, mlir::Block *b) {
          return getBlockDepth(a) < getBlockDepth(b);
        });

        for (mlir::Block *block : sortedBlocks) {
          auto &items = blockConduits[block];
          bool inIfBlock =
              mlir::isa_and_present<mlir::scf::IfOp>(block->getParentOp());
          for (auto &[name, iv] : items) {
            if (inIfBlock)
              nameNeedsRuntime[name] = true;
            else if (!nameNeedsRuntime.count(name))
              nameNeedsRuntime[name] = false;
          }

          if (items.size() < 2)
            continue;

          llvm::SmallVector<std::pair<std::string, LiveInterval>, 4> sortable(
              items.begin(), items.end());
          llvm::SmallVector<unsigned> groupIds = assignGroups(sortable);

          for (unsigned i = 0; i < sortable.size(); ++i) {
            llvm::StringRef n = sortable[i].first;
            if (!nameToGroup.count(n))
              nameToGroup[n] = nextGroupId + groupIds[i];
          }

          unsigned maxGroup =
              *std::max_element(groupIds.begin(), groupIds.end());
          nextGroupId += maxGroup + 1;
        }

        llvm::StringMap<bool> nameIsTier3;
        for (auto &ci : conduits) {
          bool hasTier3 = false;
          deviceOp.walk([&](mlir::Operation *op) {
            if (!mlir::isa<PutMemref, GetMemref, PutMemrefAsync,
                           GetMemrefAsync>(op))
              return;
            auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
            if (nameAttr && nameAttr.getValue() == ci.name)
              hasTier3 = true;
          });
          nameIsTier3[ci.name] = hasTier3;
        }

        llvm::DenseMap<unsigned, unsigned> groupCount;
        for (auto &[name, gid] : nameToGroup)
          ++groupCount[gid];

        llvm::DenseMap<unsigned, bool> groupNeedsRuntime;
        for (auto &[name, gid] : nameToGroup) {
          auto it = nameNeedsRuntime.find(name);
          if (it != nameNeedsRuntime.end() && it->second)
            groupNeedsRuntime[gid] = true;
          else if (!groupNeedsRuntime.count(gid))
            groupNeedsRuntime[gid] = false;
        }

        // dma_repeat compatibility (S2MM mirror of MM2S check above).
        if (!checkGroupDmaRepeatCompatibility("S2MM", conduits, nameToGroup,
                                              groupCount)) {
          signalPassFailure();
          continue;
        }

        // Path c (#99): cross-producer S2MM groups would emit duplicate
        // dst-port circuit flows in Pass C routePhase Sub-case 4a (two
        // aie.flow ops targeting the same consumer-tile DMA:N), which
        // aie-routing rejects.  Until packet routing lands (Sprint N+4),
        // restrict the dma_channel_group_s2mm annotation to groups whose
        // members all share the same producer tile.  The producer-tile
        // lookup mirrors the MM2S step's inferredMap + extractCoord
        // pattern (lines 320-324 above).  Iterates over nameToGroup so
        // the predicate domain matches exactly the annotation loop below.
        llvm::DenseMap<unsigned, std::pair<int64_t, int64_t>> groupProducer;
        llvm::DenseMap<unsigned, bool> groupCrossProducer;
        for (auto &[name, gid] : nameToGroup) {
          std::pair<int64_t, int64_t> prod = {-1, -1};
          auto tileIt = inferredMap.find(name);
          if (tileIt != inferredMap.end() && tileIt->second.producerTile)
            prod = extractCoord(tileIt->second.producerTile);
          auto pit = groupProducer.find(gid);
          if (pit == groupProducer.end())
            groupProducer[gid] = prod;
          else if (pit->second != prod)
            groupCrossProducer[gid] = true;
        }

        for (auto &ci : conduits) {
          auto it = nameToGroup.find(ci.name);
          if (it == nameToGroup.end())
            continue;
          unsigned gid = it->second;
          if (groupCount[gid] < 2)
            continue;

          // Path c (#99): skip cross-producer groups; deferred to packet
          // routing (Sprint N+4).
          auto crossIt = groupCrossProducer.find(gid);
          if (crossIt != groupCrossProducer.end() && crossIt->second)
            continue;

          if (nameIsTier3.count(ci.name) && nameIsTier3[ci.name]) {
            int64_t depth = 1;
            if (auto depthOpt = ci.createOp.getDepth())
              depth = *depthOpt;
            if (depth > 1) {
              ci.createOp.emitRemark()
                  << "conduit-fuse-channels: skipping S2MM fusion for '"
                  << ci.name
                  << "' — Tier 3 channel with depth>1 not supported in fuse "
                     "groups";
              continue;
            }
          }

          std::string label = "group" + std::to_string(gid);
          mlir::MLIRContext *ctx = module.getContext();
          ci.createOp->setAttr("dma_channel_group_s2mm",
                               mlir::StringAttr::get(ctx, label));
          llvm::StringRef fuseMode =
              groupNeedsRuntime[gid] ? "runtime" : "static";
          ci.createOp->setAttr("fuse_mode_s2mm",
                               mlir::StringAttr::get(ctx, fuseMode));
        }
      }
    }); // end module.walk over DeviceOp
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitFuseChannelsPass() {
  return std::make_unique<ConduitFuseChannelsPass>();
}

} // namespace xilinx::conduit
