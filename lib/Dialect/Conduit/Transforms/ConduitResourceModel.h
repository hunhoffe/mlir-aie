//===- ConduitResourceModel.h - Per-tile resource accounting --*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Per-tile resource accounting (locks, BDs, data-memory bytes) shared across
// Conduit transform passes that need to reason about tile capacity.
//
// Today this header is consumed only by --conduit-depth-promote, which lifted
// the constants + pre-walk from the body of ConduitDepthPromotion.cpp without
// behavioral change.  Future fusion / scheduling passes will reuse the same
// model so capacity decisions stay consistent across the pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITRESOURCEMODEL_H
#define AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITRESOURCEMODEL_H

#include "ConduitTileInference.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <utility>

namespace xilinx::conduit {

// ---------------------------------------------------------------------------
// Constants for budget heuristics
// ---------------------------------------------------------------------------

// AIE1 has 16 locks per tile.  Each depth-2 conduit uses 2 locks (prod+cons),
// so promoting adds 1 lock.  We refuse if the tile already uses >= this many.
static constexpr int64_t kAIE1MaxLocksPerTile = 16;

// Maximum number of BD slots per tile.  Promoting adds 1 BD per consumer.
static constexpr int64_t kMaxBDSlotsPerTile = 16;

// Memory budget per compute tile (bytes).  Promoting doubles buffer usage.
// 32 KiB is the typical AIE tile data memory size.
static constexpr int64_t kDefaultTileMemoryBytes = 32 * 1024;

// ---------------------------------------------------------------------------
// Inline tile-coordinate helpers
// ---------------------------------------------------------------------------

/// Pack (col, row) into the per-tile DenseMap key used by ConduitResourceModel.
inline int64_t tileKey(int64_t col, int64_t row) {
  return (col << 32) | (row & 0xFFFFFFFF);
}

/// Extract (col, row) from a tile SSA Value.  Returns (-1, -1) when the
/// defining op is not an aie.tile (e.g., an unplaced aie.logical_tile).
inline std::pair<int64_t, int64_t> extractCoord(mlir::Value tileVal) {
  if (auto tileOp = tileVal.getDefiningOp<AIE::TileOp>())
    return {static_cast<int64_t>(tileOp.getCol()),
            static_cast<int64_t>(tileOp.getRow())};
  return {-1, -1};
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Estimate single-slot buffer size in bytes from a channel element type.
/// Falls back to 4 bytes for non-memref or dynamically-shaped element types.
int64_t estimateSingleSlotBytes(mlir::Type elemType);

// ---------------------------------------------------------------------------
// ConduitResourceModel — per-tile counters keyed by tileKey(col, row).
// ---------------------------------------------------------------------------

struct ConduitResourceModel {
  /// Total locks claimed on this tile across all conduit channels touching it.
  llvm::DenseMap<int64_t, int64_t> lockCount;

  /// Total BD slots claimed on this tile.
  llvm::DenseMap<int64_t, int64_t> bdCount;

  /// Total data-memory bytes claimed on this tile.
  llvm::DenseMap<int64_t, int64_t> memUsed;
};

/// Walk every conduit.create in `module` and accumulate per-tile resource
/// usage into `model`, using `inferredMap` to resolve producer / consumer
/// tiles for each channel.
///
/// Behavior matches the original inline pre-walk in ConduitDepthPromotion:
///   * Cascade conduits contribute zero resources (no FIFO, no buffer).
///   * Each non-cascade channel adds 2 locks, `depth` BDs, and
///     `estimateSingleSlotBytes(elementType) * depth` bytes per resolved
///     consumer tile.
///   * Producer tile contributes the same when it is a non-shim tile
///     (row != 0) and was inferred.
///   * Tile coordinates that cannot be extracted (extractCoord returns
///     col == -1) are skipped silently.
void populateConduitResourceModel(
    mlir::ModuleOp module, const llvm::StringMap<InferredTiles> &inferredMap,
    ConduitResourceModel &model);

} // namespace xilinx::conduit

#endif // AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITRESOURCEMODEL_H
