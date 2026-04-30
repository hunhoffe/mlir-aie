//===- ConduitTileInference.cpp - Tile inference from IR structure --------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Implements inferAllTiles(): walks IR structure to determine which tiles
// produce/consume each conduit channel, returning mlir::Value tile op
// references.  Replaces the need for producer_tile / consumer_tiles
// attributes on conduit.create.
//
//===----------------------------------------------------------------------===//

#include "ConduitTileInference.h"

#include "llvm/ADT/StringSet.h"

namespace xilinx::conduit {

llvm::StringMap<InferredTiles> inferAllTiles(mlir::Operation *scope) {
  llvm::StringMap<InferredTiles> result;

  // Helper: check if a Value is already in a SmallVector<Value>.
  auto contains = [](const llvm::SmallVector<mlir::Value> &vec,
                     mlir::Value v) -> bool {
    return llvm::find(vec, v) != vec.end();
  };

  // Build a (col, row) → aie.tile Value cache by walking the scope once.
  // Used by Source 7 (MemTile enumeration) and Source 8b (air_*_tile attr
  // lookup) to resolve coordinates to existing aie.tile SSA values.  Mirrors
  // the pattern in ConduitInferModes.cpp.
  llvm::DenseMap<std::pair<int64_t, int64_t>, mlir::Value> tileCache;
  scope->walk([&](AIE::TileOp tileOp) {
    tileCache[{tileOp.getCol(), tileOp.getRow()}] = tileOp.getResult();
  });

  // -------------------------------------------------------------------------
  // Source 1 & 2: Walk aie.core ops for Acquire and GetMemrefAsync.
  //
  // Acquire(Port::Produce) inside aie.core → producer tile.
  // Acquire(Port::Consume) inside aie.core → consumer tile.
  // GetMemrefAsync inside aie.core → consumer tile (Tier 3 DMA receive).
  // -------------------------------------------------------------------------
  scope->walk([&](AIE::CoreOp coreOp) {
    mlir::Value tileVal = coreOp.getTile();
    if (!tileVal)
      return;

    coreOp.walk([&](Acquire acqOp) {
      std::string name = acqOp.getName().str();
      auto &entry = result[name];
      if (acqOp.getPort() == Port::Produce) {
        entry.producerTile = tileVal;
      } else {
        if (!contains(entry.consumerTiles, tileVal))
          entry.consumerTiles.push_back(tileVal);
      }
    });

    // AcquireAsync has the same name/port attrs as Acquire.
    coreOp.walk([&](AcquireAsync acqOp) {
      std::string name = acqOp.getName().str();
      auto &entry = result[name];
      if (acqOp.getPort() == Port::Produce) {
        entry.producerTile = tileVal;
      } else {
        if (!contains(entry.consumerTiles, tileVal))
          entry.consumerTiles.push_back(tileVal);
      }
    });

    // ReleaseAsync carries name/port; infer tile from port direction.
    coreOp.walk([&](ReleaseAsync relOp) {
      std::string name = relOp.getName().str();
      auto &entry = result[name];
      if (relOp.getPort() == Port::Produce) {
        entry.producerTile = tileVal;
      } else {
        if (!contains(entry.consumerTiles, tileVal))
          entry.consumerTiles.push_back(tileVal);
      }
    });

    coreOp.walk([&](GetMemrefAsync getOp) {
      std::string name = getOp.getName().str();
      auto &entry = result[name];
      if (!contains(entry.consumerTiles, tileVal))
        entry.consumerTiles.push_back(tileVal);
    });
  });

  // -------------------------------------------------------------------------
  // Source 3 & 4: Walk aie.shim_dma_allocation ops.
  //
  // MM2S direction → shim producer tile (the shim tile sends data).
  // S2MM direction → shim consumer tile (the shim tile receives data).
  //
  // Matching conduit name via three methods (priority order):
  //   Case 0: conduit_channel = @name attr (set by Pass A/B, preferred)
  //   Case 1: sym_name direct match
  //   Case 2: sym_name = "<name>_shim_alloc" suffix convention
  // -------------------------------------------------------------------------
  struct ShimAllocInfo {
    mlir::Value tileVal;
    AIE::DMAChannelDir dir;
    std::string symName;
    std::string conduitChannel; // from conduit_channel attr, or empty
  };
  llvm::SmallVector<ShimAllocInfo> shimAllocs;

  scope->walk([&](AIE::ShimDMAAllocationOp shimOp) {
    mlir::Value tileVal = shimOp.getTile();
    if (!tileVal)
      return;
    ShimAllocInfo info;
    info.tileVal = tileVal;
    info.dir = shimOp.getChannelDir();
    info.symName = shimOp.getSymName().str();
    if (auto ccAttr =
            shimOp->getAttrOfType<mlir::FlatSymbolRefAttr>("conduit_channel"))
      info.conduitChannel = ccAttr.getValue().str();
    shimAllocs.push_back(std::move(info));
  });

  // Collect conduit names from the scope for matching.
  llvm::StringSet<> conduitNames;
  scope->walk(
      [&](Create createOp) { conduitNames.insert(createOp.getName().str()); });

  auto addShimTile = [&](llvm::StringRef conduitName,
                         const ShimAllocInfo &alloc) {
    auto &entry = result[conduitName];
    if (alloc.dir == AIE::DMAChannelDir::MM2S) {
      if (!entry.producerTile)
        entry.producerTile = alloc.tileVal;
    } else {
      if (!contains(entry.shimConsumerTiles, alloc.tileVal))
        entry.shimConsumerTiles.push_back(alloc.tileVal);
    }
  };

  for (const auto &alloc : shimAllocs) {
    // Case 0: conduit_channel attr — preferred path.
    if (!alloc.conduitChannel.empty() &&
        conduitNames.count(alloc.conduitChannel)) {
      addShimTile(alloc.conduitChannel, alloc);
      continue;
    }
    // Case 1: sym_name directly matches a conduit name.
    if (conduitNames.count(alloc.symName)) {
      addShimTile(alloc.symName, alloc);
      continue;
    }
    // Case 2: sym_name = "<conduitName>_shim_alloc" suffix convention.
    llvm::StringRef symRef = alloc.symName;
    if (symRef.ends_with("_shim_alloc")) {
      llvm::StringRef stripped =
          symRef.drop_back(llvm::StringLiteral("_shim_alloc").size());
      if (conduitNames.count(stripped)) {
        addShimTile(stripped, alloc);
        continue;
      }
    }
  }

  // -------------------------------------------------------------------------
  // Source 5: Walk conduit.scatter, conduit.gather, and conduit.transpose
  // for relay MemTiles.
  //
  // The $memtile operand is the SSA result of an aie.tile op (Index type).
  // -------------------------------------------------------------------------

  scope->walk([&](ScatterOp scatterOp) {
    mlir::Value memtileVal = scatterOp.getMemtile();
    if (!memtileVal)
      return;

    // MemTile is a relay for the src channel and a consumer of src.
    std::string srcName = scatterOp.getSrc().str();
    auto &srcEntry = result[srcName];
    if (!contains(srcEntry.relayTiles, memtileVal))
      srcEntry.relayTiles.push_back(memtileVal);
    if (!contains(srcEntry.consumerTiles, memtileVal))
      srcEntry.consumerTiles.push_back(memtileVal);

    // MemTile is the producer of each dst channel.
    for (auto dstAttr : scatterOp.getDsts()) {
      std::string dstName =
          mlir::cast<mlir::FlatSymbolRefAttr>(dstAttr).getValue().str();
      auto &dstEntry = result[dstName];
      if (!dstEntry.producerTile)
        dstEntry.producerTile = memtileVal;
      if (!contains(dstEntry.relayTiles, memtileVal))
        dstEntry.relayTiles.push_back(memtileVal);
    }
  });

  scope->walk([&](GatherOp gatherOp) {
    mlir::Value memtileVal = gatherOp.getMemtile();
    if (!memtileVal)
      return;

    // MemTile is the producer of the dst channel.
    std::string dstName = gatherOp.getDst().str();
    auto &dstEntry = result[dstName];
    if (!dstEntry.producerTile)
      dstEntry.producerTile = memtileVal;
    if (!contains(dstEntry.relayTiles, memtileVal))
      dstEntry.relayTiles.push_back(memtileVal);

    // MemTile is a consumer of each src channel.
    for (auto srcAttr : gatherOp.getSrcs()) {
      std::string srcName =
          mlir::cast<mlir::FlatSymbolRefAttr>(srcAttr).getValue().str();
      auto &srcEntry = result[srcName];
      if (!contains(srcEntry.consumerTiles, memtileVal))
        srcEntry.consumerTiles.push_back(memtileVal);
      if (!contains(srcEntry.relayTiles, memtileVal))
        srcEntry.relayTiles.push_back(memtileVal);
    }
  });

  // -------------------------------------------------------------------------
  // Source 6: Walk aie.put_cascade and aie.get_cascade ops for cascade
  // channels.
  //
  // Cascade conduits (routing_mode = "cascade") do not use
  // Acquire/Release ops inside aie.core, so Sources 1/2 miss them.
  // Instead, aie.put_cascade / aie.get_cascade carry an optional
  // conduit_channel = @name back-reference.
  //
  // PutCascade(conduit_channel = @name) inside aie.core → producer tile.
  // GetCascade(conduit_channel = @name) inside aie.core → consumer tile.
  // -------------------------------------------------------------------------
  scope->walk([&](AIE::CoreOp coreOp) {
    mlir::Value tileVal = coreOp.getTile();
    if (!tileVal)
      return;

    coreOp.walk([&](AIE::PutCascadeOp putOp) {
      auto cc = putOp.getConduitChannel();
      if (!cc)
        return;
      std::string name = cc->str();
      auto &entry = result[name];
      entry.producerTile = tileVal;
    });

    coreOp.walk([&](AIE::GetCascadeOp getOp) {
      auto cc = getOp.getConduitChannel();
      if (!cc)
        return;
      std::string name = cc->str();
      auto &entry = result[name];
      if (!contains(entry.consumerTiles, tileVal))
        entry.consumerTiles.push_back(tileVal);
    });
  });

  // TransposeOp: N:M relay with $srcs (array) and $dsts (array).
  scope->walk([&](TransposeOp transposeOp) {
    mlir::Value memtileVal = transposeOp.getMemtile();
    if (!memtileVal)
      return;

    // MemTile is a consumer of each src channel.
    for (auto srcAttr : transposeOp.getSrcs()) {
      std::string srcName =
          mlir::cast<mlir::FlatSymbolRefAttr>(srcAttr).getValue().str();
      auto &srcEntry = result[srcName];
      if (!contains(srcEntry.relayTiles, memtileVal))
        srcEntry.relayTiles.push_back(memtileVal);
      if (!contains(srcEntry.consumerTiles, memtileVal))
        srcEntry.consumerTiles.push_back(memtileVal);
    }

    // MemTile is the producer of each dst channel.
    for (auto dstAttr : transposeOp.getDsts()) {
      std::string dstName =
          mlir::cast<mlir::FlatSymbolRefAttr>(dstAttr).getValue().str();
      auto &dstEntry = result[dstName];
      if (!dstEntry.producerTile)
        dstEntry.producerTile = memtileVal;
      if (!contains(dstEntry.relayTiles, memtileVal))
        dstEntry.relayTiles.push_back(memtileVal);
    }
  });

  // -------------------------------------------------------------------------
  // Source 7: Fused MemTile standalone producer inference.
  //
  // For conduit.create ops that have a dma_channel_group attr but still
  // have no inferred producerTile after Sources 1–6, infer the producer from
  // MemTile aie.tile declarations.  This handles the case where a MemTile is
  // the standalone producer (no scatter/gather relay op, no aie.core with
  // Acquire(Produce)).
  //
  // Algorithm: collect all MemTile aie.tile ops in the device scope, exclude
  // any that are already a consumerTile or relayTile for this channel.  If
  // exactly one MemTile candidate remains, use it as the producerTile.
  // -------------------------------------------------------------------------

  // Collect all MemTile tile Values from the tile cache.
  llvm::SmallVector<mlir::Value> memTileValues;
  for (auto &[coords, tileVal] : tileCache) {
    auto tileOp = llvm::dyn_cast<AIE::TileOp>(tileVal.getDefiningOp());
    if (tileOp && tileOp.isMemTile())
      memTileValues.push_back(tileVal);
  }

  if (!memTileValues.empty()) {
    scope->walk([&](Create createOp) {
      std::string name = createOp.getName().str();
      auto &entry = result[name];

      // Only act on channels with dma_channel_group and no producer yet.
      if (entry.producerTile)
        return;
      if (!createOp->getAttrOfType<mlir::StringAttr>("dma_channel_group"))
        return;

      // Filter out MemTiles already used as consumer or relay for this channel.
      llvm::SmallVector<mlir::Value> candidates;
      for (mlir::Value mt : memTileValues) {
        if (contains(entry.consumerTiles, mt))
          continue;
        if (contains(entry.relayTiles, mt))
          continue;
        candidates.push_back(mt);
      }

      // Unambiguous: exactly one MemTile candidate → infer as producer.
      if (candidates.size() == 1)
        entry.producerTile = candidates[0];
    });
  }

  // -------------------------------------------------------------------------
  // Source 8: Air-channel origin producer/consumer inference.
  //
  // Channels lowered from air.channel through --air-hierarchy-to-aie +
  // --air-channel-to-conduit (Pass B) use PutMemrefAsync/GetMemrefAsync
  // instead of Acquire/Release ops.  Source 2 already handles
  // GetMemrefAsync inside aie.core for consumer tiles, but PutMemrefAsync
  // inside aie.core is not covered by any prior source.
  //
  // Additionally, Pass B (Phase 2b.7) persists inferred tile coordinates
  // as discardable attrs (air_producer_tile, air_consumer_tiles) on
  // conduit.create ops.  These capture tile info from enclosing aie.core
  // and memory-space-based inference that would otherwise be lost when the
  // original air.channel put/get ops are erased.
  //
  // Sub-sources:
  //   8a: PutMemrefAsync inside aie.core → producer tile.
  //   8b: air_producer_tile / air_consumer_tiles attrs on conduit.create.
  // -------------------------------------------------------------------------

  // 8a: PutMemrefAsync inside aie.core → producer tile.
  scope->walk([&](AIE::CoreOp coreOp) {
    mlir::Value tileVal = coreOp.getTile();
    if (!tileVal)
      return;
    coreOp.walk([&](PutMemrefAsync putOp) {
      std::string name = putOp.getName().str();
      auto &entry = result[name];
      if (!entry.producerTile)
        entry.producerTile = tileVal;
    });
  });

  // 8b: Read air_producer_tile / air_consumer_tiles attrs from
  // conduit.create ops.  These are set by Pass B (AirChannelToConduit
  // Phase 2b.7) to persist tile coordinate information from aie.core
  // enclosure and memory-space inference.
  scope->walk([&](Create createOp) {
    std::string name = createOp.getName().str();
    auto &entry = result[name];

    // Producer tile from air_producer_tile = [col, row].
    if (!entry.producerTile) {
      if (auto attr = createOp->getAttrOfType<mlir::DenseI64ArrayAttr>(
              "air_producer_tile")) {
        if (attr.size() >= 2) {
          auto it = tileCache.find({attr[0], attr[1]});
          if (it != tileCache.end())
            entry.producerTile = it->second;
        }
      }
    }

    // Consumer tiles from air_consumer_tiles = [col0, row0, col1, row1, ...].
    // Tiles at row 0 are dispatched to shimConsumerTiles; others to
    // consumerTiles.
    if (auto attr = createOp->getAttrOfType<mlir::DenseI64ArrayAttr>(
            "air_consumer_tiles")) {
      for (int64_t i = 0; i + 1 < attr.size(); i += 2) {
        int64_t col = attr[i], row = attr[i + 1];
        auto it = tileCache.find({col, row});
        if (it == tileCache.end())
          continue;
        mlir::Value tileVal = it->second;
        if (row == 0) {
          if (!contains(entry.shimConsumerTiles, tileVal))
            entry.shimConsumerTiles.push_back(tileVal);
        } else {
          if (!contains(entry.consumerTiles, tileVal))
            entry.consumerTiles.push_back(tileVal);
        }
      }
    }
  });

  return result;
}

} // namespace xilinx::conduit
