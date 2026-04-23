//===- AIEAssignRuntimeSequenceBDIDs.cpp ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEAssignBufferDescriptorIDs.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

namespace xilinx::AIEX {
#define GEN_PASS_DEF_AIEASSIGNRUNTIMESEQUENCEBDIDS
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"
} // namespace xilinx::AIEX

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIEX;

namespace {

/// Lifetime interval of one aiex.dma_configure_task. The interval spans
/// [startIdx, endIdx], where startIdx is the configure's pre-order position
/// and endIdx is the position of its first matching aiex.dma_free_task. If no
/// free is encountered the interval lives forever (endIdx == max).
struct ConfigureInterval {
  DMAConfigureTaskOp configOp;
  int64_t startIdx;
  int64_t endIdx;
};

} // namespace

struct AIEAssignRuntimeSequenceBDIDsPass
    : xilinx::AIEX::impl::AIEAssignRuntimeSequenceBDIDsBase<
          AIEAssignRuntimeSequenceBDIDsPass> {

  BdIdGenerator &
  getGeneratorForTile(AIE::TileOp tile,
                      std::map<AIE::TileOp, BdIdGenerator> &gens) {
    auto it = gens.find(tile);
    if (it != gens.end())
      return std::get<1>(*it);
    const AIETargetModel &targetModel =
        tile->getParentOfType<AIE::DeviceOp>().getTargetModel();
    auto inserted = gens.insert(std::pair(
        tile, BdIdGenerator(tile.getCol(), tile.getRow(), targetModel)));
    return std::get<1>(*inserted.first);
  }

  /// Reserve user-specified bd_ids on the given configure task in the tile's
  /// generator. Errors if a requested ID is already held by another live
  /// interval (or by an earlier user spec).
  LogicalResult reserveUserSpecified(DMAConfigureTaskOp op,
                                     BdIdGenerator &gen) {
    WalkResult result = op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
      if (!bd_op.getBdId().has_value())
        return WalkResult::advance();
      uint32_t id = bd_op.getBdId().value();
      if (gen.bdIdAlreadyAssigned(id)) {
        op.emitOpError("Specified buffer descriptor ID ")
            << id
            << " is already in use. Emit an aiex.dma_free_task operation to "
               "reuse BDs.";
        return WalkResult::interrupt();
      }
      gen.assignBdId(id);
      return WalkResult::advance();
    });
    return result.wasInterrupted() ? failure() : success();
  }

  /// Allocate bd_ids for all unspecified BDs on the given configure task, out
  /// of the tile's generator. The configure_task's own channel index is used
  /// to honor per-channel BD partitioning (e.g. mem-tile even/odd channels).
  /// Returns the list of IDs allocated (or already user-specified) on this
  /// task in `assigned` so the caller can free them when the interval ends.
  LogicalResult allocateUnspecified(DMAConfigureTaskOp op, BdIdGenerator &gen,
                                    int channelIndex,
                                    SmallVectorImpl<uint32_t> &assigned) {
    WalkResult result = op.walk<WalkOrder::PreOrder>([&](AIE::DMABDOp bd_op) {
      if (bd_op.getBdId().has_value()) {
        assigned.push_back(bd_op.getBdId().value());
        return WalkResult::advance();
      }
      std::optional<uint32_t> next_id = gen.nextBdId(channelIndex);
      if (!next_id) {
        AIE::TileOp tile = op.getTileOp();
        op.emitOpError()
            << "Allocator exhausted available buffer descriptor IDs for "
               "channel "
            << channelIndex << " on tile (" << tile.getCol() << ", "
            << tile.getRow()
            << "). Live BD intervals exceed the per-channel pool capacity; "
               "interleave aiex.dma_await_task / aiex.dma_free_task with "
               "configures to recycle IDs.";
        return WalkResult::interrupt();
      }
      bd_op.setBdId(*next_id);
      assigned.push_back(*next_id);
      return WalkResult::advance();
    });
    return result.wasInterrupted() ? failure() : success();
  }

  void runOnOperation() override {
    // This pass models BD-ID lifetime as the interval
    //
    //     [aiex.dma_configure_task   ...   aiex.dma_free_task]
    //
    // and assigns each configure_task a set of buffer descriptor IDs from the
    // per-(tile, channel) pool such that no two intervals which overlap on
    // the same tile share an ID. IDs are recycled across non-overlapping
    // intervals.
    //
    // The pool to allocate from is selected by the configure_task's own
    // channel index (DMAConfigureTaskOp::getChannel), and BdIdGenerator
    // further filters by AIETargetModel::isBdChannelAccessible so e.g.
    // mem-tile even/odd channels stay in their disjoint BD ranges.
    //
    // Branching / loops within the runtime sequence body are not modelled;
    // intervals are derived from straight-line pre-order positions. A proper
    // liveness analysis would be required for control flow.

    AIE::DeviceOp device = getOperation();

    // Insert a dma_free_task immediately after each dma_await_task. After
    // waiting on a task its BD IDs are guaranteed safe to recycle.
    device.walk([&](DMAAwaitTaskOp op) {
      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);
      DMAFreeTaskOp::create(builder, op.getLoc(), op.getTask());
    });

    // First walk: collect intervals. `pos` is the pre-order position of each
    // op; only configure_task and free_task positions are read, but every op
    // bumps the counter so positions are stable across IR mutation between
    // walks.
    DenseMap<Operation *, size_t> intervalIdx;
    SmallVector<ConfigureInterval> intervals;
    SmallVector<DMAFreeTaskOp> freesToErase;
    bool collectFailure = false;
    int64_t pos = 0;
    device.walk<WalkOrder::PreOrder>([&](Operation *op) {
      int64_t myPos = pos++;
      if (auto cfg = dyn_cast<DMAConfigureTaskOp>(op)) {
        ConfigureInterval iv;
        iv.configOp = cfg;
        iv.startIdx = myPos;
        iv.endIdx = std::numeric_limits<int64_t>::max();
        intervalIdx[cfg.getOperation()] = intervals.size();
        intervals.push_back(iv);
        return;
      }
      if (auto free = dyn_cast<DMAFreeTaskOp>(op)) {
        DMAConfigureTaskOp cfg = free.getTaskOp();
        if (!cfg) {
          auto err = free.emitOpError(
              "does not reference a valid configure_task operation.");
          Operation *defOp = free.getTask().getDefiningOp();
          if (defOp && llvm::isa<DMAStartBdChainOp>(defOp))
            err.attachNote(defOp->getLoc())
                << "Lower this operation first using the "
                   "--aie-materialize-bd-chains pass.";
          if (defOp && llvm::isa<DMAConfigureTaskForOp>(defOp))
            err.attachNote(defOp->getLoc())
                << "Lower this operation first using the "
                   "--aie-substitute-shim-dma-allocations pass.";
          collectFailure = true;
          return;
        }
        auto it = intervalIdx.find(cfg.getOperation());
        if (it != intervalIdx.end()) {
          // Multiple frees on the same task (legal; old behavior treated
          // them as silent double-frees). Keep the earliest endpoint so the
          // BD becomes recyclable as soon as the program permits.
          intervals[it->second].endIdx =
              std::min(intervals[it->second].endIdx, myPos);
        }
        freesToErase.push_back(free);
      }
    });
    if (collectFailure)
      return signalPassFailure();

    // Sort intervals by start position. (Pre-order collection already yields
    // start-sorted order; re-sort defensively so future changes that reorder
    // collection do not silently change semantics.)
    SmallVector<size_t> order(intervals.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return intervals[a].startIdx < intervals[b].startIdx;
    });

    // Per-tile generators and live-interval lists. A live entry holds the
    // interval's endIdx and the BD IDs it is still occupying in the tile's
    // generator.
    std::map<AIE::TileOp, BdIdGenerator> gens;
    DenseMap<Operation *,
             SmallVector<std::pair<int64_t, SmallVector<uint32_t>>>>
        livePerTile;

    // Sweep intervals in start-sorted order. Before allocating IDs for a new
    // interval, reap all live intervals on the same tile whose endIdx is
    // strictly less than the new interval's startIdx; their IDs return to
    // the generator and become candidates for reuse.
    for (size_t i : order) {
      ConfigureInterval &iv = intervals[i];
      AIE::TileOp tile = iv.configOp.getTileOp();
      BdIdGenerator &gen = getGeneratorForTile(tile, gens);
      auto &live = livePerTile[tile.getOperation()];

      SmallVector<std::pair<int64_t, SmallVector<uint32_t>>> stillLive;
      stillLive.reserve(live.size());
      for (auto &entry : live) {
        if (entry.first < iv.startIdx) {
          for (uint32_t id : entry.second)
            gen.freeBdId(id);
        } else {
          stillLive.emplace_back(entry.first, std::move(entry.second));
        }
      }
      live = std::move(stillLive);

      // Reserve user-specified BD IDs first, then auto-allocate the rest.
      // Both steps record IDs in the generator so subsequent intervals see
      // them as live.
      if (failed(reserveUserSpecified(iv.configOp, gen)))
        return signalPassFailure();

      SmallVector<uint32_t> assigned;
      int channelIndex = static_cast<int>(iv.configOp.getChannel());
      if (failed(
              allocateUnspecified(iv.configOp, gen, channelIndex, assigned)))
        return signalPassFailure();

      live.emplace_back(iv.endIdx, std::move(assigned));
    }

    // dma_free_task ops have served their purpose for the allocator and are
    // not consumed by downstream passes; erase them now.
    for (DMAFreeTaskOp free : freesToErase)
      free.erase();
  }
};

std::unique_ptr<OperationPass<AIE::DeviceOp>>
AIEX::createAIEAssignRuntimeSequenceBDIDsPass() {
  return std::make_unique<AIEAssignRuntimeSequenceBDIDsPass>();
}
