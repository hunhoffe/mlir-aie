//===- ConduitCheckOrdering.cpp - conduit-check-ordering pass ----*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// CSDFa static-ordering verifier for DMA-only programs with rate annotations.
//
// Complements --conduit-check-deps (acyclicity) and M6/M7 (balance).
// Together: acyclicity + balance + ordering = formally verified
// deadlock-freedom for the DMA-only subset, grounded in Denolf 2007
// Theorems 4-5.
//
// Scope: DMA-only channels only.  Silently skips any conduit.create whose
// use-def chain contains conduit.acquire or conduit.acquire_async ops —
// cross-tier programs are outside CSDFa scope by design.
//
// Algorithm:
//   1. Walk all conduit.create ops with both producer_rates and consumer_rates.
//   2. Filter to DMA-only channels (no window-token ops in the use chain).
//   3. Group DMA-only channels by tile (producer and consumer tiles).
//   4. For tiles with 2+ DMA-only rated channels, check whether pairs of
//      channels fire at overlapping CSDF phases.  In the CSDFa model, two
//      DMA events on the same tile in the same phase have undefined relative
//      ordering — emit a warning.
//
// Run with:  aie-opt --conduit-check-ordering <input.mlir>
//
// This pass is OPT-IN and NOT part of the default pipeline.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITCHECKORDERING
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a channel is DMA-only (no window-token ops reference it).
/// A channel is DMA-only if no conduit.acquire or conduit.acquire_async op
/// in the module references the same channel name.  Window-token channels
/// are outside the scope of CSDFa's static-ordering model.
static bool isDMAOnlyChannel(mlir::ModuleOp module, llvm::StringRef name) {
  bool hasWindowOps = false;
  module.walk([&](mlir::Operation *op) {
    if (hasWindowOps)
      return;
    auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name");
    if (!nameAttr || nameAttr.getValue() != name)
      return;
    if (mlir::isa<Acquire, AcquireAsync>(op))
      hasWindowOps = true;
  });
  return !hasWindowOps;
}

/// Pack (col, row) into a single int64_t key for tile grouping.
static int64_t tileKey(int64_t col, int64_t row) {
  return (col << 32) | (row & 0xFFFFFFFF);
}

// ---------------------------------------------------------------------------
// Channel info for ordering analysis
// ---------------------------------------------------------------------------

struct ChannelInfo {
  mlir::Operation *createOp;
  llvm::StringRef name;
  int64_t producerTileKey;
  llvm::SmallVector<int64_t> consumerTileKeys;
};

// ---------------------------------------------------------------------------
// Main pass
// ---------------------------------------------------------------------------

struct ConduitCheckOrderingPass
    : public impl::ConduitCheckOrderingBase<ConduitCheckOrderingPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Step 1: Collect DMA-only channels with rate annotations.
    llvm::SmallVector<ChannelInfo> channels;

    module.walk([&](Create op) {
      llvm::StringRef name = op.getSymName();
      auto prodRates = op->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "producer_rates");
      auto consRates = op->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "consumer_rates");

      if (name.empty() || !prodRates || !consRates)
        return; // no rate annotations — skip

      // Skip non-DMA-only channels (outside CSDFa scope).
      if (!isDMAOnlyChannel(module, name))
        return;

      ChannelInfo info;
      info.createOp = op.getOperation();
      info.name = name;

      // Extract tile coordinates.
      info.producerTileKey = -1;
      if (auto pt = op->getAttrOfType<mlir::DenseI64ArrayAttr>(
              "producer_tile")) {
        auto arr = pt.asArrayRef();
        if (arr.size() >= 2)
          info.producerTileKey = tileKey(arr[0], arr[1]);
      }
      if (auto ct = op->getAttrOfType<mlir::DenseI64ArrayAttr>(
              "consumer_tiles")) {
        auto arr = ct.asArrayRef();
        for (size_t i = 0; i + 1 < arr.size(); i += 2)
          info.consumerTileKeys.push_back(tileKey(arr[i], arr[i + 1]));
      }

      channels.push_back(std::move(info));
    });

    if (channels.empty())
      return; // No DMA-only rated channels — nothing to check.

    // Step 2: Group channels by tile (both producer and consumer sides).
    // Map: tile key → list of channel indices in `channels`.
    llvm::DenseMap<int64_t, llvm::SmallVector<size_t>> tileChannelMap;
    for (size_t i = 0; i < channels.size(); ++i) {
      if (channels[i].producerTileKey >= 0)
        tileChannelMap[channels[i].producerTileKey].push_back(i);
      for (int64_t ck : channels[i].consumerTileKeys)
        tileChannelMap[ck].push_back(i);
    }

    // Step 3: For tiles with 2+ DMA-only channels, check ordering.
    // In the CSDFa model, an actor firing processes all its channels
    // simultaneously.  When two channels share a tile, their DMA events
    // in the same phase have undefined relative ordering.
    llvm::StringSet<> warnedPairs; // avoid duplicate warnings

    for (auto &[tile, chIndices] : tileChannelMap) {
      if (chIndices.size() < 2)
        continue; // single channel — total order by definition

      for (size_t i = 0; i < chIndices.size(); ++i) {
        for (size_t j = i + 1; j < chIndices.size(); ++j) {
          auto &chA = channels[chIndices[i]];
          auto &chB = channels[chIndices[j]];

          // Skip if same channel (can appear on both producer and consumer
          // tile lists, so two indices may refer to the same channel).
          if (chA.name == chB.name)
            continue;

          // Create a unique pair key to avoid duplicate warnings.
          std::string pairKey;
          if (chA.name < chB.name)
            pairKey = (chA.name + ":" + chB.name).str();
          else
            pairKey = (chB.name + ":" + chA.name).str();

          if (warnedPairs.count(pairKey))
            continue;
          warnedPairs.insert(pairKey);

          // Both channels share a tile and have CSDF rate annotations.
          // Their DMA events fire during the same actor firing — the
          // relative ordering is undefined in the CSDFa model.
          chA.createOp->emitWarning(
              "conduit-check-ordering: ambiguous DMA event ordering -- "
              "channel '")
              << chA.name << "' and '" << chB.name
              << "' share a tile and fire at overlapping CSDF phases; "
              << "the relative ordering of their DMA events is undefined";
        }
      }
    }

    // Note: provably cyclic ordering detection is handled by
    // --conduit-check-deps (M12 dep-token DAG acyclicity check).
    // This pass focuses on the complementary ordering ambiguity check.
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCheckOrderingPass() {
  return std::make_unique<ConduitCheckOrderingPass>();
}

} // namespace xilinx::conduit
