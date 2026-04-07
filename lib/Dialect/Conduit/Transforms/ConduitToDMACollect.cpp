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
  // Phase 2: Find aie.device op, build tile cache, determine architecture.
  // -----------------------------------------------------------------------

  module.walk([&](AIE::DeviceOp op) {
    if (!state.deviceOp)
      state.deviceOp = op;
  });

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

  // Build tile cache.
  state.deviceOp.walk([&](AIE::TileOp tile) {
    state.tileCache[{tile.getCol(), tile.getRow()}] = tile;
  });

  // Set device body reference and insertion point.
  state.deviceBody = &state.deviceOp.getBodyRegion().front();
  for (mlir::Operation &op : *state.deviceBody) {
    if (mlir::isa<AIE::TileOp>(op))
      state.insertAfterTile = &op;
  }

  // Pre-populate lock ID counters from existing locks.
  state.deviceOp.walk([&](AIE::LockOp existingLock) {
    if (!existingLock.getLockID().has_value())
      return;
    mlir::Value tileVal = existingLock.getTile();
    int existingId = static_cast<int>(existingLock.getLockID().value());
    int &counter = state.lockIdCounter[tileVal];
    if (existingId + 1 > counter)
      counter = existingId + 1;
  });

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
