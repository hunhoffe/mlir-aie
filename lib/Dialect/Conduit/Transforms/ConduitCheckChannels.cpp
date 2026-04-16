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
// This pass validates that no tile exceeds its DMA channel slot_elems.  It runs
// AFTER Pass A or Pass B (which populate conduit.create attributes) and can
// run either before or after --conduit-fuse-channels.  When fusion annotations
// are present (dma_channel_group attribute), the pass accounts for them:
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

#include "ConduitTileInference.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include <string>

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITCHECKCHANNELS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

using TileCoord = std::pair<int64_t, int64_t>;

// ---------------------------------------------------------------------------
// P2-E: Convergence hazard check.
//
// After Pass C has emitted all aie.packet_flow ops, walk the module and
// check for convergence hazards: two packet flows sharing the same physical
// source port (tile + bundle + channel) that both route to the SAME
// destination tile with DIFFERENT packet IDs.
//
// When the switchbox arbitrates packets on a shared port, flits from different
// IDs are interleaved.  If two flows target the same consumer through the same
// source port, the consumer sees an unpredictable interleaving of packets from
// both flows — ordering is not guaranteed under sustained load.
//
// The check reconstructs the port → [(flow_id, dest_tile)] map directly from
// the emitted aie.packet_flow ops and flags any port where a collision exists.
//
// This is an OPT-IN post-pass check; it emits warnings (not errors) and does
// not signal pass failure, because the condition is a hazard rather than a
// hard correctness violation (it depends on workload timing).
// ---------------------------------------------------------------------------

// Key identifying a physical source port: (tile_col, tile_row, bundle,
// channel).
struct SourcePortKey {
  int32_t col;
  int32_t row;
  uint32_t bundle; // AIE::WireBundle as uint32_t for DenseMap
  uint32_t channel;

  bool operator==(const SourcePortKey &o) const {
    return col == o.col && row == o.row && bundle == o.bundle &&
           channel == o.channel;
  }
};

struct SourcePortKeyInfo : public llvm::DenseMapInfo<SourcePortKey> {
  static SourcePortKey getEmptyKey() { return {-1, -1, ~0u, ~0u}; }
  static SourcePortKey getTombstoneKey() { return {-2, -2, ~0u - 1, ~0u - 1}; }
  static unsigned getHashValue(const SourcePortKey &k) {
    return llvm::hash_combine(k.col, k.row, k.bundle, k.channel);
  }
  static bool isEqual(const SourcePortKey &a, const SourcePortKey &b) {
    return a == b;
  }
};

// Entry recorded per packet_flow: the flow ID and the destination tile coord.
struct FlowEntry {
  int8_t flowId;
  TileCoord dstTile;
  AIE::PacketFlowOp flowOp; // for diagnostic location
};

// Check all aie.packet_flow ops in the module for convergence hazards.
// Returns true if any warnings were emitted.
static bool checkConvergenceHazards(mlir::ModuleOp module) {
  // Map: source port → list of (flowId, dstTile, flowOp) entries.
  llvm::DenseMap<SourcePortKey, llvm::SmallVector<FlowEntry, 2>,
                 SourcePortKeyInfo>
      portMap;

  module.walk([&](AIE::PacketFlowOp flowOp) {
    int8_t flowId = flowOp.getID();

    // Extract source port from the packet_source op in the flow's region.
    AIE::PacketSourceOp srcOp;
    flowOp.getPorts().walk([&](AIE::PacketSourceOp s) {
      if (!srcOp)
        srcOp = s;
    });

    // Extract all destination tiles from packet_dest ops.
    flowOp.getPorts().walk([&](AIE::PacketDestOp dstOp) {
      if (!srcOp)
        return;

      // Get source tile coordinates.
      mlir::Value srcTileVal = srcOp.getTile();
      auto srcTileOp = srcTileVal.getDefiningOp<AIE::TileOp>();
      if (!srcTileOp)
        return;

      // Get destination tile coordinates.
      mlir::Value dstTileVal = dstOp.getTile();
      auto dstTileOp = dstTileVal.getDefiningOp<AIE::TileOp>();
      if (!dstTileOp)
        return;

      SourcePortKey key{srcTileOp.getCol(), srcTileOp.getRow(),
                        static_cast<uint32_t>(srcOp.getBundle()),
                        static_cast<uint32_t>(srcOp.getChannel())};
      TileCoord dst{dstTileOp.getCol(), dstTileOp.getRow()};
      portMap[key].push_back({flowId, dst, flowOp});
    });
  });

  bool anyWarning = false;

  // For each source port, check if two flows with different IDs share the same
  // destination tile.
  for (auto &[port, entries] : portMap) {
    if (entries.size() < 2)
      continue;

    // Check all pairs.
    for (size_t i = 0; i < entries.size(); ++i) {
      for (size_t j = i + 1; j < entries.size(); ++j) {
        const FlowEntry &a = entries[i];
        FlowEntry b = entries[j]; // non-const copy so emitWarning() is callable
        // Same destination tile but different packet IDs → ordering hazard.
        if (a.dstTile == b.dstTile && a.flowId != b.flowId) {
          b.flowOp.emitWarning()
              << "packet flows with different IDs ("
              << static_cast<int>(a.flowId) << " and "
              << static_cast<int>(b.flowId)
              << ") route to the same consumer tile (" << b.dstTile.first
              << ", " << b.dstTile.second
              << ") through the same switchbox source port on tile ("
              << port.col << ", " << port.row
              << "); ordering is not guaranteed under sustained load";
          anyWarning = true;
        }
      }
    }
  }

  return anyWarning;
}

struct ConduitCheckChannelsPass
    : public impl::ConduitCheckChannelsBase<ConduitCheckChannelsPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Infer tile coordinates from IR structure.
    auto inferredMap = inferAllTiles(module);
    auto extractCoord = [](mlir::Value tileVal) -> std::pair<int64_t, int64_t> {
      if (auto tileOp = tileVal.getDefiningOp<AIE::TileOp>())
        return {static_cast<int64_t>(tileOp.getCol()),
                static_cast<int64_t>(tileOp.getRow())};
      return {-1, -1};
    };

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
    //   - If the conduit has a dma_channel_group attribute, the channel
    //   ID
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
      if (auto groupAttr = createOp->getAttrOfType<mlir::StringAttr>(
              "dma_channel_group"))
        channelId = groupAttr.getValue().str();

      auto tileIt = inferredMap.find(name);

      // --- Producer tile: needs one MM2S channel ---
      // Prefer inference, fallback to attribute for hand-written IR.
      int64_t prodCol = -1, prodRow = -1;
      if (tileIt != inferredMap.end() && tileIt->second.producerTile) {
        std::tie(prodCol, prodRow) = extractCoord(tileIt->second.producerTile);
      }
      if (prodCol >= 0 && prodRow > 0) {
        TileCoord tc = {prodCol, prodRow};
        prodChannels[tc].insert(channelId);
        if (!prodFirstCreate.count(tc))
          prodFirstCreate[tc] = createOp;
      }

      // --- Consumer tiles: each needs one S2MM channel ---
      // Consumer-side fusion is not yet implemented, so each conduit gets
      // its own S2MM channel regardless of fusion annotations.
      // Prefer inference, fallback to attribute.
      llvm::SmallVector<std::pair<int64_t, int64_t>> consCoords;
      if (tileIt != inferredMap.end() &&
          !tileIt->second.consumerTiles.empty()) {
        for (mlir::Value tv : tileIt->second.consumerTiles) {
          auto [c, r] = extractCoord(tv);
          if (c >= 0)
            consCoords.push_back({c, r});
        }
      }
      for (auto [col, row] : consCoords) {
        if (row > 0) {
          TileCoord tc = {col, row};
          consChannels[tc].insert(name);
          if (!consFirstCreate.count(tc))
            consFirstCreate[tc] = createOp;
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

    // -----------------------------------------------------------------------
    // P2-E: Convergence hazard check (post-Pass-C analysis).
    // Walk aie.packet_flow ops in the lowered IR and warn when two flows
    // with different IDs share the same source port and destination tile.
    // This is a warning-only check; it never signals pass failure.
    // -----------------------------------------------------------------------
    checkConvergenceHazards(module);
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCheckChannelsPass() {
  return std::make_unique<ConduitCheckChannelsPass>();
}

} // namespace xilinx::conduit
