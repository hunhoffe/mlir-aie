//===- ConduitCheckChannels.cpp - conduit-check-channels pass ----*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// P1-B: Hardware DMA channel count validation.
//
// Each AIE tile exposes a fixed number of DMA channels:
//   - AIE2 compute tiles: 2 MM2S (source/producer) + 2 S2MM (dest/consumer)
//   - AIE2 MemTiles:      6 MM2S + 6 S2MM
//   - AIE1 compute tiles: 2 MM2S + 2 S2MM
//
// Each conduit requires one MM2S channel on its producer tile and one S2MM
// channel on each of its consumer tiles.  When a program assigns more conduits
// to a tile than it has hardware channels, the resulting configuration is
// invalid — Pass C cannot assign unique channels and the hardware will
// malfunction silently.
//
// This pass validates that no tile exceeds its DMA channel capacity.  It runs
// AFTER Pass A or Pass B (which populate conduit.create attributes) and can
// run either before or after --conduit-fuse-channels.  When fusion annotations
// are present (fused_dma_channel_group attribute), the pass accounts for them:
// conduits sharing a fusion group count as one channel, not N.
//
// The pass queries the target model for each tile's channel limits using
// getNumSourceSwitchboxConnections (MM2S) and getNumDestSwitchboxConnections
// (S2MM) with WireBundle::DMA.
//
// Shim tiles (row == 0) are excluded from this check — they use a separate
// shim DMA model with aie.shim_dma_allocation.
//
// Run with:  aie-opt --conduit-check-channels <input.mlir>
//
// This pass emits hard errors and signals pass failure on the first violation
// found.  It is OPT-IN and NOT part of the default pipeline.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"

#include <string>

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITCHECKCHANNELS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

using TileCoord = std::pair<int64_t, int64_t>;

struct ConduitCheckChannelsPass
    : public impl::ConduitCheckChannelsBase<ConduitCheckChannelsPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Find the first aie.device op (needed for target model).
    AIE::DeviceOp deviceOp;
    module.walk([&](AIE::DeviceOp op) {
      if (!deviceOp)
        deviceOp = op;
    });

    if (!deviceOp) {
      // No device op — nothing to check.
      return;
    }

    const AIE::AIETargetModel &targetModel = AIE::getTargetModel(deviceOp);

    // Per-tile channel usage tracking.
    //
    // Each entry in the StringSet is a "channel ID":
    //   - If the conduit has a fused_dma_channel_group attribute, the channel ID
    //     is the group label (conduits sharing a group share one channel).
    //   - Otherwise, the channel ID is the conduit name (each conduit gets its
    //     own channel).
    //
    // The size of the StringSet after all conduits are processed gives the
    // number of hardware DMA channels required on that tile.
    llvm::DenseMap<TileCoord, llvm::StringSet<>> prodChannels; // MM2S
    llvm::DenseMap<TileCoord, llvm::StringSet<>> consChannels; // S2MM

    // First conduit.create per tile — used for error location reporting.
    llvm::DenseMap<TileCoord, Create> prodFirstCreate;
    llvm::DenseMap<TileCoord, Create> consFirstCreate;

    module.walk([&](Create createOp) {
      std::string name = createOp.getName().str();

      // Check for fusion annotation (set by --conduit-fuse-channels).
      // Conduits in the same group share one hardware channel.
      std::string channelId = name;
      if (auto groupAttr =
              createOp->getAttrOfType<mlir::StringAttr>(
                  "fused_dma_channel_group"))
        channelId = groupAttr.getValue().str();

      // --- Producer tile: needs one MM2S channel ---
      if (auto pt = createOp.getProducerTile()) {
        if (pt->size() >= 2) {
          int64_t col = (*pt)[0], row = (*pt)[1];
          // Shim tiles (row == 0) use shim DMA allocation, not switchbox DMA.
          if (row > 0) {
            TileCoord tc = {col, row};
            prodChannels[tc].insert(channelId);
            if (!prodFirstCreate.count(tc))
              prodFirstCreate[tc] = createOp;
          }
        }
      }

      // --- Consumer tiles: each needs one S2MM channel ---
      // Consumer-side fusion is not yet implemented, so each conduit gets
      // its own S2MM channel regardless of fusion annotations.
      if (auto ct = createOp.getConsumerTiles()) {
        for (size_t i = 0; i + 1 < ct->size(); i += 2) {
          int64_t col = (*ct)[i], row = (*ct)[i + 1];
          if (row > 0) {
            TileCoord tc = {col, row};
            consChannels[tc].insert(name);
            if (!consFirstCreate.count(tc))
              consFirstCreate[tc] = createOp;
          }
        }
      }
    });

    bool anyFailure = false;

    // --- Check MM2S (producer-side) limits ---
    for (auto &[tc, channels] : prodChannels) {
      auto [col, row] = tc;
      uint32_t limit = targetModel.getNumSourceSwitchboxConnections(
          col, row, AIE::WireBundle::DMA);
      uint32_t used = channels.size();
      if (used > limit) {
        prodFirstCreate[tc]->emitError()
            << "DMA channel limit exceeded on tile (" << col << ", " << row
            << "): " << used << " conduits require " << used
            << " MM2S channels, hardware supports " << limit;
        anyFailure = true;
      }
    }

    // --- Check S2MM (consumer-side) limits ---
    for (auto &[tc, channels] : consChannels) {
      auto [col, row] = tc;
      uint32_t limit = targetModel.getNumDestSwitchboxConnections(
          col, row, AIE::WireBundle::DMA);
      uint32_t used = channels.size();
      if (used > limit) {
        consFirstCreate[tc]->emitError()
            << "DMA channel limit exceeded on tile (" << col << ", " << row
            << "): " << used << " conduits require " << used
            << " S2MM channels, hardware supports " << limit;
        anyFailure = true;
      }
    }

    if (anyFailure)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCheckChannelsPass() {
  return std::make_unique<ConduitCheckChannelsPass>();
}

} // namespace xilinx::conduit
