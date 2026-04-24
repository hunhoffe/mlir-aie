//===- ConduitInferModes.cpp - conduit-infer-modes pass ----------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// P2-C: mode=any inference pass.
//
// Walks conduit.create ops with routing_mode="any" and resolves them to a
// concrete routing mode by applying the R3 + Step 3.5 decision procedure:
//
//   R1. If routing_mode is not "any": skip (already resolved).
//   R2. If routing_mode is "cascade": skip (hardware-fixed, no inference).
//   R3a. If producer and single consumer are adjacent tiles
//   (isLegalMemAffinity)
//        and forceDMA is not set: resolve to "shared_memory".  Pass C Phase 3c
//        detects routing_mode == SharedMemory and uses shared memory — no DMA
//        flow or channel allocation is needed.
//   R3b. If a circuit-mode MM2S DMA channel is available on the producer tile:
//        resolve to "circuit".
//   Step 3.5. If circuit DMA is exhausted: resolve to "packet" (DMA channel
//        sharing via packet routing).  Requires global packet ID budget > 0.
//   Step 4. If all modes are exhausted: emit a hard error.
//
// The per-tile MM2S budget is simulated in the same order as Pass C Phase 4.5a
// (conduits sorted by name for determinism, matching conduitMap MapVector
// order).
//
// This pass runs AFTER Pass A or Pass B and BEFORE Pass C.
// It is OPT-IN and NOT part of the default pipeline.
//
// Run with:  aie-opt --conduit-infer-modes <input.mlir>
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "ConduitTileInference.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITINFERMODES
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// Maximum packet flow IDs available (same default as PacketIDAllocator).
static constexpr uint8_t kDefaultPacketIDBudget = 32;

struct ConduitInferModesPass
    : public impl::ConduitInferModesBase<ConduitInferModesPass> {

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

    // Find the aie.device op; required for target model queries.
    AIE::DeviceOp deviceOp;
    module.walk([&](AIE::DeviceOp op) {
      if (!deviceOp)
        deviceOp = op;
    });

    if (!deviceOp) {
      // No device op present — nothing to infer (no conduits can have tiles).
      return;
    }

    const AIE::AIETargetModel &targetModel = AIE::getTargetModel(deviceOp);

    // Build tile coordinate cache from aie.tile ops in the device body.
    // Maps (col, row) → tile SSA Value.
    llvm::DenseMap<std::pair<int64_t, int64_t>, mlir::Value> tileCache;
    deviceOp.walk([&](AIE::TileOp tileOp) {
      int64_t col = tileOp.getCol();
      int64_t row = tileOp.getRow();
      tileCache[{col, row}] = tileOp.getResult();
    });

    // Per-tile next-available MM2S channel counter.
    // Models the same allocation order as Pass C Phase 4.5a so that the
    // circuit/packet decision made here will be consistent with Pass C.
    llvm::DenseMap<mlir::Value, int32_t> tileNextMM2S;

    // Global packet flow ID budget countdown.
    // Each "packet" assignment consumes one ID (per consumer tile per conduit).
    uint8_t pktBudget = kDefaultPacketIDBudget;

    // Collect conduit.create ops in source order for deterministic iteration.
    llvm::SmallVector<Create> anyConduits;
    llvm::SmallVector<Create> otherConduits;

    module.walk([&](Create op) {
      auto rm = op.getRoutingMode();
      // Absent routing_mode = unresolved (formerly "any").
      if (!rm.has_value())
        anyConduits.push_back(op);
      else
        otherConduits.push_back(op);
    });

    // Pre-consume MM2S channels already allocated by already-resolved conduits
    // so that the budget seen by unresolved conduits is accurate.
    for (Create op : otherConduits) {
      auto rm = op.getRoutingMode();
      // Cascade and shared-memory conduits consume no DMA channels.
      if (rm && *rm == RoutingMode::Cascade)
        continue;

      // Get producer tile from inference.
      int64_t prodCol = -1, prodRow = -1;
      auto tileIt = inferredMap.find(op.getName().str());
      if (tileIt != inferredMap.end() && tileIt->second.producerTile) {
        std::tie(prodCol, prodRow) = extractCoord(tileIt->second.producerTile);
      }
      if (prodCol < 0 || prodRow < 0)
        continue;
      if (prodRow == 0) // shim producer — no compute tile MM2S consumed
        continue;

      mlir::Value prodTileVal = tileCache.lookup({prodCol, prodRow});
      if (!prodTileVal)
        continue;

      // Check if this conduit uses shared memory (skip DMA channel allocation).
      // Shared memory: single consumer, adjacent tile, via_DMA not set.
      // Get consumer tiles from inference.
      llvm::SmallVector<std::pair<int64_t, int64_t>> consCoords;
      if (tileIt != inferredMap.end() &&
          !tileIt->second.consumerTiles.empty()) {
        for (mlir::Value tv : tileIt->second.consumerTiles) {
          auto [c, r] = extractCoord(tv);
          if (c >= 0)
            consCoords.push_back({c, r});
        }
      }

      if (consCoords.size() == 1) {
        int64_t consCol = consCoords[0].first;
        int64_t consRow = consCoords[0].second;
        auto forceDMAAttr = op->getAttrOfType<mlir::BoolAttr>("forceDMA");
        bool forceDMA = forceDMAAttr && forceDMAAttr.getValue();
        if (!forceDMA) {
          bool adj = targetModel.isLegalMemAffinity(prodCol, prodRow, consCol,
                                                    consRow) ||
                     targetModel.isLegalMemAffinity(consCol, consRow, prodCol,
                                                    prodRow);
          // Cross-column same-row "adjacency" (W-neighbor) is reported legal
          // by AIE2TargetModel but NOT used in the Conduit lowering path
          // when not explicitly requested.  Treat it as non-adjacent here so
          // pre-consumed MM2S budget matches Pass C's actual allocation.
          // Use target-model helpers so future targets (e.g., NPU3) can
          // override the neighbor relationship correctly.
          bool sameRowDifferentCol =
              targetModel.isMemWest(prodCol, prodRow, consCol, consRow) ||
              targetModel.isMemWest(consCol, consRow, prodCol, prodRow);
          if (sameRowDifferentCol)
            adj = false;
          if (adj)
            continue; // shared-memory path, no DMA channel consumed
        }
      }

      // Non-cascade, non-shared-memory, non-shim conduit: consumes one MM2S.
      tileNextMM2S[prodTileVal]++;
    }

    mlir::OpBuilder builder(module.getContext());
    bool failed = false;

    // Now resolve each unresolved conduit (absent routing_mode).
    for (Create op : anyConduits) {
      // Get producer tile from inference.
      int64_t prodCol = -1, prodRow = -1;
      auto tileIt = inferredMap.find(op.getName().str());
      if (tileIt != inferredMap.end() && tileIt->second.producerTile) {
        std::tie(prodCol, prodRow) = extractCoord(tileIt->second.producerTile);
      }
      if (prodCol < 0 || prodRow < 0) {
        // No producer tile info — cannot determine topology; default to
        // circuit.
        op.setRoutingModeAttr(
            RoutingModeAttr::get(module.getContext(), RoutingMode::Circuit));
        continue;
      }

      // Shim producer: always use circuit mode (shim DMA).
      if (prodRow == 0) {
        op.setRoutingModeAttr(
            RoutingModeAttr::get(module.getContext(), RoutingMode::Circuit));
        continue;
      }

      mlir::Value prodTileVal = tileCache.lookup({prodCol, prodRow});
      if (!prodTileVal) {
        // Producer tile not in device: cannot route, default to circuit and
        // let Pass C emit the appropriate error.
        op.setRoutingModeAttr(
            RoutingModeAttr::get(module.getContext(), RoutingMode::Circuit));
        continue;
      }

      // Get consumer tiles from inference.
      llvm::SmallVector<std::pair<int64_t, int64_t>> consCoords;
      if (tileIt != inferredMap.end() &&
          !tileIt->second.consumerTiles.empty()) {
        for (mlir::Value tv : tileIt->second.consumerTiles) {
          auto [c, r] = extractCoord(tv);
          if (c >= 0)
            consCoords.push_back({c, r});
        }
      }

      auto forceDMAAttr = op->getAttrOfType<mlir::BoolAttr>("forceDMA");
      bool forceDMA = forceDMAAttr && forceDMAAttr.getValue();

      // -----------------------------------------------------------------------
      // R3a: Shared memory check.
      //
      // If there is exactly one consumer tile, it is adjacent to the producer
      // (isLegalMemAffinity in either direction), and forceDMA is not set,
      // resolve to SharedMemory explicitly.  Pass C Phase 3c recognizes
      // routing_mode == SharedMemory and skips DMA channel allocation.
      // -----------------------------------------------------------------------
      if (!forceDMA && consCoords.size() == 1) {
        int64_t consCol = consCoords[0].first;
        int64_t consRow = consCoords[0].second;
        bool adj =
            targetModel.isLegalMemAffinity(prodCol, prodRow, consCol,
                                           consRow) ||
            targetModel.isLegalMemAffinity(consCol, consRow, prodCol, prodRow);
        // Cross-column same-row W-neighbor is silently legal in
        // AIE2TargetModel but not NPU-validated through Conduit's shmem
        // lowering.  Do NOT infer "shared_memory" for it; let Pass C take
        // the DMA path (which is exercised + known good).
        // Use target-model helpers so future targets (e.g., NPU3) can override
        // the neighbor relationship correctly.
        bool sameRowDifferentCol =
            targetModel.isMemWest(prodCol, prodRow, consCol, consRow) ||
            targetModel.isMemWest(consCol, consRow, prodCol, prodRow);
        if (sameRowDifferentCol)
          adj = false;
        if (adj) {
          op.setRoutingModeAttr(RoutingModeAttr::get(
              module.getContext(), RoutingMode::SharedMemory));
          continue;
        }
      }

      // -----------------------------------------------------------------------
      // R3a.5: Multicast inference.
      //
      // If the conduit has more than one consumer tile and all consumers use
      // uniform access patterns (consumer_dimensions absent or all sub-arrays
      // identical), resolve to "packet".  Multicast costs only 1 packet ID
      // regardless of N consumers (the switchbox broadcasts).
      // -----------------------------------------------------------------------
      if (consCoords.size() > 1) {
        // consumer_dimensions moved to get_memref ops; treat all multicasts as
        // uniform for routing_mode inference purposes.
        bool uniform = true;
        if (uniform) {
          if (pktBudget >= 1) {
            pktBudget -= 1;
            op.setRoutingModeAttr(
                RoutingModeAttr::get(module.getContext(), RoutingMode::Packet));
            op->emitRemark("conduit-infer-modes: resolved multicast conduit "
                           "to \"packet\" (uniform consumers on tile (")
                << prodCol << "," << prodRow << "))";
            continue;
          }
        }
      }

      // -----------------------------------------------------------------------
      // R3b: Circuit DMA channel available?
      //
      // Query the hardware MM2S channel limit from the target model.
      // If the next-available slot is within bounds, assign Circuit and
      // advance the counter so the next unresolved conduit on the same tile
      // sees the reduced budget.
      // -----------------------------------------------------------------------
      uint32_t maxMM2S =
          static_cast<uint32_t>(targetModel.getNumSourceSwitchboxConnections(
              static_cast<int>(prodCol), static_cast<int>(prodRow),
              AIE::WireBundle::DMA));
      if (maxMM2S == 0)
        maxMM2S = 2; // safe fallback for non-modelled tiles

      int32_t nextCh =
          tileNextMM2S.count(prodTileVal) ? tileNextMM2S[prodTileVal] : 0;

      if (static_cast<uint32_t>(nextCh) < maxMM2S) {
        tileNextMM2S[prodTileVal]++;
        op.setRoutingModeAttr(
            RoutingModeAttr::get(module.getContext(), RoutingMode::Circuit));
        continue;
      }

      // -----------------------------------------------------------------------
      // Step 3.5: Packet DMA fallback.
      //
      // Circuit DMA is exhausted on this tile.  If packet IDs remain, assign
      // Packet.  Each packet assignment per consumer tile costs one packet ID.
      // -----------------------------------------------------------------------
      unsigned numConsumers = consCoords.empty() ? 1 : consCoords.size();
      if (pktBudget >= numConsumers) {
        pktBudget -= static_cast<uint8_t>(numConsumers);
        op.setRoutingModeAttr(
            RoutingModeAttr::get(module.getContext(), RoutingMode::Packet));
        op->emitRemark("conduit-infer-modes: resolved unresolved routing_mode "
                       "to \"packet\" (circuit DMA exhausted on tile (")
            << prodCol << "," << prodRow << "))";
        continue;
      }

      // -----------------------------------------------------------------------
      // Step 4: All modes exhausted — hard error.
      // -----------------------------------------------------------------------
      op->emitError("conduit-infer-modes: cannot resolve routing_mode "
                    "for conduit '")
          << op.getName() << "': circuit DMA MM2S channels exhausted on tile ("
          << prodCol << "," << prodRow
          << ") and packet flow ID budget is also exhausted";
      failed = true;
    }

    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitInferModesPass() {
  return std::make_unique<ConduitInferModesPass>();
}

} // namespace xilinx::conduit
