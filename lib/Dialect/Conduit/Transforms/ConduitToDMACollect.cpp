//===- ConduitToDMACollect.cpp - Phase 1-2.5: metadata collection --*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Phase 1: Walk conduit.create ops and populate conduitMap with ConduitInfo.
// Phase 2: Find the aie.device op, build tile cache, determine aieArch.
// Phase 2.5: Compute effectiveDepth for producer-side buffer optimization.
// Also: collect link source names, consumer acquire names.
//
//===----------------------------------------------------------------------===//

#include "ConduitToDMACommon.h"

namespace xilinx::conduit {

void collectPhase(ConduitToDMAState &state) {
  mlir::ModuleOp module = state.module;

  // -----------------------------------------------------------------------
  // Phase 1: Collect ConduitInfo from conduit.create typed attributes.
  //
  // conduit.create carries all metadata as typed attributes:
  //   producer_tile   : DenseI64Array [col, row]
  //   consumer_tiles  : DenseI64Array [col0, row0, col1, row1, ...]
  //   element_type    : TypeAttr
  //   depth           : I64Attr
  // -----------------------------------------------------------------------

  module.walk([&](Create op) {
    ConduitInfo info;
    info.slotElems = op.getSlotElems();

    if (auto depthOpt = op.getDepth())
      info.depth = static_cast<int64_t>(*depthOpt);

    if (auto etOpt = op.getElementType())
      info.elemType = *etOpt;

    // Producer tile coordinates.
    if (auto pt = op.getProducerTile()) {
      if (pt->size() >= 2) {
        int64_t col = (*pt)[0], row = (*pt)[1];
        info.producerTileCoord = std::make_pair(col, row);
        std::string s;
        llvm::raw_string_ostream os(s);
        os << "tile(" << col << "," << row << ")";
        info.producerTileStr = os.str();
      }
    }

    // Compute consumer tile coordinates (non-shim, row > 0).
    if (auto ct = op.getConsumerTiles()) {
      for (size_t i = 0; i + 1 < ct->size(); i += 2) {
        int64_t col = (*ct)[i], row = (*ct)[i + 1];
        info.consumerTileCoords.push_back(std::make_pair(col, row));
        std::string s;
        llvm::raw_string_ostream os(s);
        os << "tile(" << col << "," << row << ")";
        info.consumerTileStrs.push_back(os.str());
      }
    }

    // Shim consumer tile coordinates (row == 0).
    if (auto sct = op.getShimConsumerTiles()) {
      for (size_t i = 0; i + 1 < sct->size(); i += 2) {
        int64_t col = (*sct)[i], row = (*sct)[i + 1];
        info.shimConsumerTileCoords.push_back(std::make_pair(col, row));
      }
    }

    // Cyclostatic (CSDF) access pattern.
    if (auto ap = op.getAccessPattern()) {
      for (int64_t v : *ap)
        info.accessPattern.push_back(v);
    }

    // Routing mode (enum; absent = unresolved — treated as "any" in Pass C
    // so that Step 3.5 packet fallback still applies when no explicit mode
    // has been set via --conduit-infer-modes).
    if (auto rm = op.getRoutingMode())
      info.routingMode = stringifyRoutingMode(*rm).str();
    else
      info.routingMode = "any"; // absent = unresolved; let Pass C decide

    // Core stream port for routing_mode="stream".
    if (auto aspAttr =
            op->getAttrOfType<mlir::IntegerAttr>("aie_stream_port"))
      info.aieStreamPort = static_cast<int32_t>(aspAttr.getInt());

    // Alloc tile delegate coordinates.
    if (auto at = op.getAllocTile()) {
      if (at->size() >= 2) {
        info.hasAllocTile = true;
        info.allocTileCoord = std::make_pair((*at)[0], (*at)[1]);
      }
    }

    // Fused DMA channel group label.
    if (auto fuseAttr =
            op->getAttrOfType<mlir::StringAttr>("fused_dma_channel_group")) {
      info.fuseGroup = fuseAttr.getValue().str();

      // Reject fuse_mode="runtime" — static BD chain lowering is incorrect
      // for conditional programs; the DMA engine runs both members
      // unconditionally even when the scf.if branch is not taken.
      if (auto modeAttr =
              op->getAttrOfType<mlir::StringAttr>("fuse_mode")) {
        if (modeAttr.getValue() == "runtime") {
          op.emitError(
              "conduit-to-dma: fuse_mode=\"runtime\" is not yet supported "
              "(control-packet BD reprogramming path unimplemented); "
              "conduit ops inside scf.if branches cannot be fused safely "
              "with static BD chains — remove the fused_dma_channel_group "
              "annotation or restructure the program to avoid conditional "
              "fusion");
          state.passFailed = true;
          return;
        }
      }
    }

    // New feature attributes.
    if (auto attr = op.getDisableSynchronization())
      if (*attr)
        info.disableSynchronization = true;
    if (auto attr = op.getViaDMA())
      if (*attr)
        info.viaDMA = true;
    if (auto plioAttr = op.getPlio())
      if (*plioAttr)
        info.plio = true;
    if (auto attr = op.getDmaRepeat())
      info.dmaRepeat = static_cast<int64_t>(*attr);
    if (auto attr = op.getBdRepeat())
      info.bdRepeat = static_cast<int64_t>(*attr);
    // Note: time_multiplex_count has been removed from conduit.create.
    // Pass C infers BD chain length from putCount (Phase 1 put_memref_async walk).
    if (auto attr = op.getProducerDimensions()) {
      if (auto typed = mlir::dyn_cast<AIE::BDDimLayoutArrayAttr>(*attr))
        info.producerDimensions = typed;
    }
    if (auto attr = op.getConsumerDimensions()) {
      if (auto typed = mlir::dyn_cast<AIE::BDDimLayoutArrayArrayAttr>(*attr)) {
        for (auto dims : typed.getValue())
          info.consumerDimensions.push_back(dims);
      }
    }

    // Cascade depth assertion: cascade conduits must have depth = 1.
    // The hardware cascade stream is a blocking register (rendezvous channel),
    // not a FIFO. depth = 2 cannot be implemented; emitting it would produce
    // a silently incorrect program.
    if (info.routingMode == "cascade" && info.depth != 1) {
      op.emitError("cascade conduit must have depth = 1; hardware has no FIFO "
                   "on the cascade stream");
      state.passFailed = true;
      return;
    }

    state.conduitMap[op.getName().str()] = std::move(info);
  });

  if (state.passFailed)
    return;

  // -----------------------------------------------------------------------
  // Phase 2: Find aie.device ops, build unified tile cache, determine arch.
  //
  // Multi-device support: collect ALL DeviceOps from the module.
  // --conduit-fuse-operators offsets tile coordinates in device B so there
  // are no coordinate conflicts.  The unified tile cache covers all devices.
  // state.deviceOp is set to the first device (for legacy single-device code).
  // state.deviceOps holds all devices in module order for multi-device paths.
  // -----------------------------------------------------------------------

  module.walk([&](AIE::DeviceOp op) {
    state.deviceOps.push_back(op);
    if (!state.deviceOp)
      state.deviceOp = op;
  });

  // -----------------------------------------------------------------------
  // Phase 5a: Infer tile coordinates from IR structure.
  //
  // Primary source: walk aie.core ops to find which conduit channels are
  // used (via acquire/release/put_memref/get_memref) on each tile, and walk
  // aie.shim_dma_allocation ops for the shim consumer tile map.
  //
  // Fallback: if the IR walk finds nothing for a field, the attribute-read
  // values already populated above (producer_tile / consumer_tiles /
  // shim_consumer_tiles) remain in place.  This preserves all existing tests.
  // -----------------------------------------------------------------------

  // channelToProducerTile: channel name → (col, row) of producer tile.
  // Source: Acquire with Port::Produce inside aie.core (Tier 2 window model).
  // Note: Release has no $name attr; Acquire with port=Produce is the
  //   unambiguous producer marker. put_memref_async is NOT used — it can
  //   appear on either tile in loopback tests, making direction ambiguous.
  llvm::StringMap<std::pair<int64_t, int64_t>> channelToProducerTile;

  // channelToConsumerTiles: channel name → list of (col, row) consumer tiles.
  // Source: Acquire with Port::Consume inside aie.core (Tier 2 window model).
  llvm::StringMap<llvm::SmallVector<std::pair<int64_t, int64_t>>>
      channelToConsumerTiles;

  // Walk every aie.core. Only Acquire ops (which carry both $name and $port)
  // provide unambiguous producer/consumer tile identification.
  for (AIE::DeviceOp dev : state.deviceOps) {
    dev.walk([&](AIE::CoreOp coreOp) {
      AIE::TileOp tileOp =
          coreOp.getTile().getDefiningOp<AIE::TileOp>();
      if (!tileOp)
        return;
      int64_t col = static_cast<int64_t>(tileOp.getCol());
      int64_t row = static_cast<int64_t>(tileOp.getRow());

      coreOp.walk([&](Acquire acqOp) {
        std::string name = acqOp.getName().str();
        if (acqOp.getPort() == Port::Produce) {
          channelToProducerTile[name] = {col, row};
        } else {
          // Port::Consume → consumer tile.
          auto &vec = channelToConsumerTiles[name];
          std::pair<int64_t, int64_t> coord = {col, row};
          if (llvm::find(vec, coord) == vec.end())
            vec.push_back(coord);
        }
      });
      // GetMemrefAsync inside aie.core → consumer tile (Tier 3 DMA receive).
      // This handles the case where a compute core drives DMA receives directly
      // without Tier 2 acquire/release (e.g., conduit_to_dma_tier3_bd_length).
      coreOp.walk([&](GetMemrefAsync getOp) {
        std::string name = getOp.getName().str();
        auto &vec = channelToConsumerTiles[name];
        std::pair<int64_t, int64_t> coord = {col, row};
        if (llvm::find(vec, coord) == vec.end())
          vec.push_back(coord);
      });
    });
  }

  // shimAllocationMap: shim sym_name → (col, row) of the shim tile.
  llvm::StringMap<std::pair<int64_t, int64_t>> shimAllocationMap;
  for (AIE::DeviceOp dev : state.deviceOps) {
    dev.walk([&](AIE::ShimDMAAllocationOp shimOp) {
      // The tile operand is an Index value produced by aie.tile(col, row).
      mlir::Value tileVal = shimOp.getTile();
      AIE::TileOp tileOp = tileVal.getDefiningOp<AIE::TileOp>();
      if (!tileOp)
        return;
      int64_t col = static_cast<int64_t>(tileOp.getCol());
      int64_t row = static_cast<int64_t>(tileOp.getRow());
      shimAllocationMap[shimOp.getSymName()] = {col, row};
    });
  }

  // Apply inferred coordinates to conduitMap, using fallback to existing values.
  for (auto &[name, info] : state.conduitMap) {
    // Producer tile: IR walk wins if found.
    auto prodIt = channelToProducerTile.find(name);
    if (prodIt != channelToProducerTile.end()) {
      int64_t col = prodIt->second.first;
      int64_t row = prodIt->second.second;
      info.producerTileCoord = {col, row};
      std::string s;
      llvm::raw_string_ostream os(s);
      os << "tile(" << col << "," << row << ")";
      info.producerTileStr = os.str();
    }
    // For relay ops (scatter/gather/transpose), producer tile is the memtile
    // attr on the relay op itself; that is handled in linkPhase already.

    // Consumer tiles: IR walk wins if found.
    auto consIt = channelToConsumerTiles.find(name);
    if (consIt != channelToConsumerTiles.end() &&
        !consIt->second.empty()) {
      info.consumerTileCoords.clear();
      info.consumerTileStrs.clear();
      for (auto [col, row] : consIt->second) {
        info.consumerTileCoords.push_back({col, row});
        std::string s;
        llvm::raw_string_ostream os(s);
        os << "tile(" << col << "," << row << ")";
        info.consumerTileStrs.push_back(os.str());
      }
    }

    // Shim consumer tiles: match shim_dma_allocation ops to conduit channels.
    //
    // Two cases handled in priority order:
    //
    // Case 1 — Pass A post-rewrite: Pass A renames conduit.create @chan to
    //   @chan_shim_alloc, so the conduitMap key IS the shim alloc sym_name.
    //   Direct lookup: shimAllocationMap["chan_shim_alloc"] → found.
    //
    // Case 2 — Direct conduit IR convention: hand-written programs use the
    //   standard suffix convention: shim alloc sym = "<chan>_shim_alloc" with
    //   conduit.create sym = "<chan>" (different symbols, no name conflict).
    //   Suffix-strip lookup: strip "_shim_alloc" from shimAllocationMap keys
    //   and check if the stripped name matches the channel name.
    //
    // Multi-consumer allocations (@chan_shim_alloc_0, _1, …): handled by
    //   Case 1 if Pass A renames the primary channel to @chan_shim_alloc,
    //   and secondary channels to @chan_shim_alloc_0 etc. (Pass A currently
    //   only renames the primary; secondary channels remain unmatched until
    //   Pass C Phase 5c scatter lowering, which uses the shim alloc directly).

    // Shim endpoint: the shim_dma_allocation sym_name IS the conduit channel
    // name after Pass A's rewrite (@chan → @chan_shim_alloc on conduit.create).
    // Direct lookup: if conduitMap key matches the shim alloc sym_name, this
    // channel's shim tile is the producer/consumer.
    auto shimIt = shimAllocationMap.find(name);
    if (shimIt != shimAllocationMap.end()) {
      auto [col, row] = shimIt->second;
      std::pair<int64_t, int64_t> coord = {col, row};
      if (llvm::find(info.shimConsumerTileCoords, coord) ==
          info.shimConsumerTileCoords.end())
        info.shimConsumerTileCoords.push_back(coord);
    }
  }

  // Hard error on depth = 0 after collection: caller must have run
  // --conduit-depth-promote before --conduit-to-dma.
  for (auto &[name, info] : state.conduitMap) {
    if (info.depth == 0) {
      // Emit a diagnostic on the conduit.create op for this channel.
      module.walk([&](Create createOp) {
        if (createOp.getName().str() == name) {
          createOp.emitError(
              "conduit-to-dma: channel @" + name +
              " has depth = 0 — run --conduit-depth-promote before "
              "--conduit-to-dma");
          state.passFailed = true;
        }
      });
    }
  }

  if (state.passFailed)
    return;

  if (!state.deviceOp) {
    module.emitWarning(
        "conduit-to-dma: no aie.device found; skipping lowering");
    return;
  }

  state.targetModel = &AIE::getTargetModel(state.deviceOp);
  state.aieArch = state.targetModel->getTargetArch();
  state.acqAction = state.isAIE2Plus()
                        ? AIE::LockAction::AcquireGreaterEqual
                        : AIE::LockAction::Acquire;

  // Build unified tile cache across all devices.
  // Each device's tiles have unique coordinates after --conduit-fuse-operators.
  for (AIE::DeviceOp dev : state.deviceOps) {
    dev.walk([&](AIE::TileOp tile) {
      state.tileCache[{tile.getCol(), tile.getRow()}] = tile;
    });
  }

  // Set device body reference and insertion point (primary device only;
  // multi-device emission uses getDeviceForTile to select the right device).
  state.deviceBody = &state.deviceOp.getBodyRegion().front();
  for (mlir::Operation &op : *state.deviceBody) {
    if (mlir::isa<AIE::TileOp>(op))
      state.insertAfterTile = &op;
  }

  // Pre-populate lock ID counters from existing locks across all devices.
  for (AIE::DeviceOp dev : state.deviceOps) {
    dev.walk([&](AIE::LockOp existingLock) {
      if (!existingLock.getLockID().has_value())
        return;
      mlir::Value tileVal = existingLock.getTile();
      int existingId = static_cast<int>(existingLock.getLockID().value());
      int &counter = state.lockIdCounter[tileVal];
      if (existingId + 1 > counter)
        counter = existingId + 1;
    });
  }

  // -----------------------------------------------------------------------
  // Collect link source names before allocation runs.
  //
  // Distribute sources: Phase 5 allocates its own per-slice lock pairs on
  // the MemTile, so Phase 3 skips lock allocation for these.
  //
  // Join sources: Phase 5 uses the existing per-source lock pairs from
  // Phase 3. Tracked separately for Phase 3 producer-tile reallocation.
  // -----------------------------------------------------------------------
  module.walk([&](Distribute distOp) {
    for (auto s : distOp.getSrcs())
      state.linkSrcNamesEarly.insert(
          mlir::cast<mlir::FlatSymbolRefAttr>(s).getValue());
    for (auto d : distOp.getDsts())
      state.linkDstNames.insert(mlir::cast<mlir::FlatSymbolRefAttr>(d).getValue());
  });
  module.walk([&](Join joinOp) {
    for (auto s : joinOp.getSrcs())
      state.linkJoinSrcNames.insert(
          mlir::cast<mlir::FlatSymbolRefAttr>(s).getValue());
    for (auto d : joinOp.getDsts())
      state.linkDstNames.insert(mlir::cast<mlir::FlatSymbolRefAttr>(d).getValue());
  });
  module.walk([&](Forward fwdOp) {
    for (auto s : fwdOp.getSrcs())
      state.linkSrcNamesEarly.insert(
          mlir::cast<mlir::FlatSymbolRefAttr>(s).getValue());
    for (auto d : fwdOp.getDsts())
      state.linkDstNames.insert(mlir::cast<mlir::FlatSymbolRefAttr>(d).getValue());
  });
  module.walk([&](ScatterOp scatterOp) {
    state.linkSrcNamesEarly.insert(scatterOp.getSrc());
    for (auto d : scatterOp.getDsts())
      state.linkDstNames.insert(
          mlir::cast<mlir::FlatSymbolRefAttr>(d).getValue());
  });
  module.walk([&](GatherOp gatherOp) {
    for (auto s : gatherOp.getSrcs())
      state.linkJoinSrcNames.insert(
          mlir::cast<mlir::FlatSymbolRefAttr>(s).getValue());
    state.linkDstNames.insert(gatherOp.getDst());
  });

  // Collect numElems from put/get_memref_async ops.
  // For Tier 3 channels (shim↔compute via DMA), slotElems encodes the slot
  // count (typically 1), but BD length must be the per-transfer element count.
  // Take the maximum num_elems seen across all puts and gets for each channel.
  module.walk([&](PutMemrefAsync op) {
    llvm::StringRef name = op.getName();
    auto it = state.conduitMap.find(name.str());
    if (it != state.conduitMap.end()) {
      int64_t n = static_cast<int64_t>(op.getNumElems());
      if (n > it->second.numElems)
        it->second.numElems = n;
    }
  });
  // Count put_memref_async ops per channel for annotation-free BD chain
  // length inference. Tier-3 channels with N sequential token-chained puts
  // (after --conduit-fuse-channels TM merge) need an N-entry linear chain.
  module.walk([&](PutMemrefAsync op) {
    llvm::StringRef name = op.getName();
    auto it = state.conduitMap.find(name.str());
    if (it != state.conduitMap.end())
      ++it->second.putCount;
  });
  module.walk([&](GetMemrefAsync op) {
    llvm::StringRef name = op.getName();
    auto it = state.conduitMap.find(name.str());
    if (it != state.conduitMap.end()) {
      int64_t n = static_cast<int64_t>(op.getNumElems());
      if (n > it->second.numElems)
        it->second.numElems = n;
    }
  });

  // Conduit names with at least one Consume-port acquire op (for rotation
  // counter allocation in Phase 3).
  module.walk([&](Acquire acqOp) {
    if (acqOp.getPort() == Port::Consume)
      state.conduitNamesWithConsumerAcquire.insert(acqOp.getName());
    else if (acqOp.getPort() == Port::Produce)
      state.conduitNamesWithProducerAcquire.insert(acqOp.getName());
  });
  module.walk([&](AcquireAsync acqOp) {
    // AcquireAsync is always consumer-side.
    state.conduitNamesWithConsumerAcquire.insert(acqOp.getName());
  });

  // -----------------------------------------------------------------------
  // Phase 2.5: Compute effectiveDepth for producer-side buffer optimization.
  //
  // For each conduit, find the maximum Produce-port acquire count
  // (maxProdAcquire). The producer only needs min(depth, maxProdAcquire+1)
  // buffers on the producer side.
  // -----------------------------------------------------------------------
  {
    llvm::DenseMap<llvm::StringRef, int64_t> maxProdAcquire;
    module.walk([&](Acquire acqOp) {
      if (acqOp.getPort() == Port::Produce) {
        int64_t count = static_cast<int64_t>(acqOp.getCount());
        auto &cur = maxProdAcquire[acqOp.getName()];
        if (count > cur)
          cur = count;
      }
    });
    for (auto &[name, info] : state.conduitMap) {
      int64_t depth = info.depth > 0 ? info.depth : 1;
      auto it = maxProdAcquire.find(name);
      if (it != maxProdAcquire.end()) {
        int64_t effDepth = std::min(depth, it->second + 1);
        info.effectiveDepth = effDepth;
      } else {
        info.effectiveDepth = depth;
      }
    }
  }

  // -----------------------------------------------------------------------
  // Phase 2.6: Compute partial-release buffer adjustment.
  //
  // For the sliding-window pattern, one port acquires N elements but releases
  // fewer than N per step, holding onto the remainder.
  // The DMA ring must have extra buffer slots: max(depth, maxAcquire + 1).
  //
  // Strategy: scan acquire/release pairs (via window SSA def-use chain) and
  // record the maximum acquire count across all pairs where acqCount > relCount,
  // separately for Consume-port and Produce-port.
  //
  // ReleaseAsync ops are omitted (async pattern never forms a sliding window
  // in practice; and they lack a direct window SSA operand).
  // -----------------------------------------------------------------------
  {
    // Per-conduit: maximum acquire count across all Consume-port
    // acquire/release pairs where acquireCount > releaseCount.
    llvm::DenseMap<llvm::StringRef, int64_t> maxConsAcquire;
    // Per-conduit: maximum acquire count across all Produce-port
    // acquire/release pairs where acquireCount > releaseCount.
    llvm::DenseMap<llvm::StringRef, int64_t> maxProdAcquirePartial;

    module.walk([&](Release relOp) {
      // Recover the paired acquire op from the window SSA operand.
      auto *defOp = relOp.getWindow().getDefiningOp();
      if (!defOp)
        return;
      auto acqOp = mlir::dyn_cast<Acquire>(defOp);
      if (!acqOp)
        return; // WaitWindow or other — not a simple acquire/release pair
      llvm::StringRef conduitName = acqOp.getName();
      int64_t acqCount = static_cast<int64_t>(acqOp.getCount());
      int64_t relCount = static_cast<int64_t>(relOp.getCount());
      if (acqCount <= relCount)
        return; // Full release — not a sliding window
      if (relOp.getPort() == Port::Consume) {
        auto &cur = maxConsAcquire[conduitName];
        if (acqCount > cur)
          cur = acqCount;
      } else if (relOp.getPort() == Port::Produce) {
        auto &cur = maxProdAcquirePartial[conduitName];
        if (acqCount > cur)
          cur = acqCount;
      }
    });

    for (auto &[name, info] : state.conduitMap) {
      auto consIt = maxConsAcquire.find(name);
      if (consIt != maxConsAcquire.end())
        info.maxConsumerAcquire = consIt->second;
      auto prodIt = maxProdAcquirePartial.find(name);
      if (prodIt != maxProdAcquirePartial.end())
        info.maxProduceAcquire = prodIt->second;
    }
  }

  // -----------------------------------------------------------------------
  // Phase 1.5: Collect conduit.register_external_buffers.
  //
  // Records the external buffer SSA values and tile coordinates into
  // ConduitInfo so that Phase 3 can skip internal buffer allocation and
  // Phase 5.5 can build the shim_dma BD chain using the external buffers.
  // -----------------------------------------------------------------------
  module.walk([&](RegisterExternalBuffers regOp) {
    llvm::StringRef conduitName = regOp.getName();
    ConduitInfo *cinfo = state.lookupConduit(conduitName);
    if (!cinfo) {
      regOp.emitWarning(
          "conduit-to-dma: register_external_buffers references unknown "
          "conduit '" + conduitName.str() + "'; ignoring");
      return;
    }
    // Record external buffer SSA values.
    for (mlir::Value extBuf : regOp.getExternalBuffers())
      cinfo->externalBuffers.push_back(extBuf);
    // Record the tile coordinate.
    auto tc = regOp.getTileCoord();
    if (tc.size() >= 2)
      cinfo->externalBufferTileCoord = {tc[0], tc[1]};
  });
}

} // namespace xilinx::conduit
