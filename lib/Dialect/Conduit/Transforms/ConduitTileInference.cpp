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
  // The $memtile attribute is a string "tile(col,row)".  We resolve it to
  // the aie.tile SSA Value via a tile cache built from the scope.
  // -------------------------------------------------------------------------

  // Build tile cache: (col, row) → tile Value.
  llvm::DenseMap<std::pair<int64_t, int64_t>, mlir::Value> tileCache;
  scope->walk([&](AIE::TileOp tileOp) {
    int64_t col = static_cast<int64_t>(tileOp.getCol());
    int64_t row = static_cast<int64_t>(tileOp.getRow());
    tileCache[{col, row}] = tileOp.getResult();
  });

  // Helper: resolve a "tile(col,row)" string to a Value via the cache.
  auto resolveTileStr = [&](llvm::StringRef memtileStr) -> mlir::Value {
    auto [col, row] = parseTileCoord(memtileStr);
    if (col < 0)
      return {};
    auto it = tileCache.find({col, row});
    if (it == tileCache.end())
      return {};
    return it->second;
  };

  scope->walk([&](ScatterOp scatterOp) {
    mlir::Value memtileVal = resolveTileStr(scatterOp.getMemtile());
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
    mlir::Value memtileVal = resolveTileStr(gatherOp.getMemtile());
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
    mlir::Value memtileVal = resolveTileStr(transposeOp.getMemtile());
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

  return result;
}

} // namespace xilinx::conduit
