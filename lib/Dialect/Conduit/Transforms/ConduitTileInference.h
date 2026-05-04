//===- ConduitTileInference.h - Tile inference from IR structure
//-*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Shared utility for inferring conduit producer/consumer tiles from IR
// structure (aie.core walks, aie.shim_dma_allocation, relay ops) instead
// of reading producer_tile / consumer_tiles attributes on conduit.create.
//
// Used by: ConduitToDMACollect (Pass C), ConduitDepthPromotion,
//          ConduitCheckOrdering, ConduitFuseChannels, ConduitFuseOperators,
//          ConduitOps verifier.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITTILEINFERENCE_H
#define AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITTILEINFERENCE_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

namespace xilinx::conduit {

// ---------------------------------------------------------------------------
// Per-conduit inferred tiles.
//
// Stores mlir::Value tile op references (aie.tile or aie.logical_tile
// results) rather than raw (col, row) coordinates.  This is forward-
// compatible with unplaced aie.logical_tile ops (TileLike interface).
//
// Populated by inferAllTiles() from IR structure walks.
//
// Precondition for passes requiring concrete coordinates: tiles must be
// placed (aie.tile ops with col/row).  Works with unplaced aie.logical_tile
// for passes that only need tile identity.
// ---------------------------------------------------------------------------
struct InferredTiles {
  /// Producer tile SSA value (aie.tile / aie.logical_tile result).
  /// Null if no producer tile inferred.
  /// Sources: Acquire(Port::Produce) inside aie.core,
  ///          aie.shim_dma_allocation with MM2S direction,
  ///          aie.put_cascade with conduit_channel attr,
  ///          PutMemrefAsync inside aie.core (Source 8a), or
  ///          air_producer_tile attr on conduit.create (Source 8b).
  mlir::Value producerTile;

  /// Non-shim consumer tiles.
  /// Sources: Acquire(Port::Consume) or GetMemrefAsync inside aie.core,
  ///          aie.get_cascade with conduit_channel attr, or
  ///          air_consumer_tiles attr on conduit.create (Source 8b, row>0).
  llvm::SmallVector<mlir::Value> consumerTiles;

  /// Shim consumer tiles (row == 0).
  /// Sources: aie.shim_dma_allocation with S2MM direction, matched via
  ///          conduit_channel attr, sym_name, or _shim_alloc suffix, or
  ///          air_consumer_tiles attr on conduit.create (Source 8b, row==0).
  llvm::SmallVector<mlir::Value> shimConsumerTiles;

  /// Relay MemTile intermediaries (from conduit.scatter/gather $memtile attr).
  llvm::SmallVector<mlir::Value> relayTiles;
};

// ---------------------------------------------------------------------------
// Bulk inference: walk the given scope (DeviceOp or ModuleOp) once and
// return a map from conduit name to inferred tiles.
//
// The walk covers:
//   1. aie.core → Acquire(Port::Produce) → producer tile
//   2. aie.core → Acquire(Port::Consume) / GetMemrefAsync → consumer tiles
//   3. aie.shim_dma_allocation (MM2S) → shim producer tile
//   4. aie.shim_dma_allocation (S2MM) → shim consumer tiles
//   5. conduit.scatter / conduit.gather / conduit.transpose → relay MemTiles
//   6. aie.core → PutCascade(conduit_channel) → cascade producer tile
//      aie.core → GetCascade(conduit_channel) → cascade consumer tiles
//   7. conduit.create with dma_channel_group + no producer after 1–6 →
//      standalone MemTile producer (exactly one unused MemTile in scope)
//   8. Air-channel origin inference (from Pass B via --air-channel-to-conduit):
//      a. aie.core → PutMemrefAsync → producer tile
//      b. conduit.create with air_producer_tile / air_consumer_tiles attrs
//         (persisted by Pass B Phase 2b.7 from hierarchy context)
//
// For efficiency, call this once per pass and reuse the result map.
// ---------------------------------------------------------------------------
llvm::StringMap<InferredTiles> inferAllTiles(mlir::Operation *scope);

} // namespace xilinx::conduit

#endif // AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITTILEINFERENCE_H
