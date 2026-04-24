//===- ConduitToDMACommon.cpp - Shared helpers for ConduitToDMA pass -*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Method bodies for PacketIDAllocator, ConduitInfo, and ConduitToDMAState.
// Declarations remain in ConduitToDMACommon.h.
//
//===----------------------------------------------------------------------===//

#include "ConduitToDMACommon.h"

namespace xilinx::conduit {

// ---------------------------------------------------------------------------
// Conduit-specific shared-memory feasibility predicate.
// See ConduitToDMACommon.h for the rationale behind the two Conduit-lowering
// adjustments wrapped here.
// ---------------------------------------------------------------------------
bool isConduitFeasibleSharedMemory(const AIE::AIETargetModel &targetModel,
                                   int64_t prodCol, int64_t prodRow,
                                   int64_t consCol, int64_t consRow,
                                   std::optional<RoutingMode> routingMode) {
  bool explicitSharedMem = (routingMode == RoutingMode::SharedMemory);
  bool rightAdj =
      targetModel.isLegalMemAffinity(prodCol, prodRow, consCol, consRow);
  bool leftAdj =
      targetModel.isLegalMemAffinity(consCol, consRow, prodCol, prodRow);
  bool sameRowDifferentCol =
      targetModel.isMemWest(prodCol, prodRow, consCol, consRow) ||
      targetModel.isMemWest(consCol, consRow, prodCol, prodRow);
  if (sameRowDifferentCol && !explicitSharedMem) {
    rightAdj = false;
    leftAdj = false;
  }
  return explicitSharedMem || rightAdj || leftAdj;
}

// ---------------------------------------------------------------------------
// PacketIDAllocator
// ---------------------------------------------------------------------------

std::optional<uint8_t> PacketIDAllocator::allocate(mlir::Value domain) {
  uint8_t &next = nextPerDomain[domain];
  // Start from 1: packet ID 0 can false-match aie.rule {mask=28, value=0},
  // causing xclbin generation failures. Hardware supports 0-31 per domain;
  // reserving ID 0 leaves 31 usable IDs — sufficient for all current designs.
  if (next == 0)
    next = 1;
  if (next >= limit) {
    module.emitError(
        "packet flow ID exhausted in MemTile domain: design requires "
        "more than ")
        << (unsigned)(limit - 1) << " distinct packet flows per MemTile";
    return std::nullopt;
  }
  return next++;
}

std::optional<uint8_t> PacketIDAllocator::allocateBlock(mlir::Value domain,
                                                        unsigned count) {
  if (count == 0)
    return std::nullopt;
  // Single ID: use the normal sequential allocator.
  if (count == 1)
    return allocate(domain);
  uint8_t &next = nextPerDomain[domain];
  if (next == 0)
    next = 1;
  // Compute P = next power of 2 >= count.
  unsigned p = 1;
  while (p < count)
    p <<= 1;
  // Align `next` up to the next multiple of P.
  uint8_t aligned = static_cast<uint8_t>(((next + p - 1) / p) * p);
  // If aligned is 0 due to wraparound, bump to p.
  if (aligned == 0)
    aligned = static_cast<uint8_t>(p);
  if (static_cast<unsigned>(aligned) + count > limit) {
    module.emitError(
        "packet flow ID exhausted in MemTile domain: need aligned block of ")
        << count << " IDs (aligned to " << p << ") but only "
        << (unsigned)(limit - next) << " IDs remain";
    return std::nullopt;
  }
  uint8_t startID = aligned;
  // Reserve the full power-of-2 block so unused slots are not reused.
  next = aligned + static_cast<uint8_t>(p);
  return startID;
}

// ---------------------------------------------------------------------------
// ConduitInfo
// ---------------------------------------------------------------------------

ConduitInfo::ResolvedTileResources
ConduitInfo::resolveForTile(mlir::Operation *op) {
  ResolvedTileResources res;
  res.prodLock = prodLock;
  res.consLock = consLock;
  res.buffers = &buffers;
  res.rotationBuf = rotationBuf;
  res.rotationBufSlot = rotationBufSlot;
  res.producerRotationBuf = producerRotationBuf;
  res.producerRotationBufSlot = producerRotationBufSlot;

  // B-11: Walk up the parent chain to find the enclosing CoreOp.
  // Stop at DeviceOp (sentinel) — ops placed directly inside aie.device
  // (but outside aie.core) are not core-body ops and have no per-tile
  // resources in the resolved form.  Without this sentinel, the walk would
  // continue past DeviceOp into ModuleOp and then nullptr, which is harmless
  // but wasteful and could mask future issues if non-device ancestors exist.
  res.coreOp = op->getParentOp();
  while (res.coreOp && !mlir::isa<AIE::CoreOp>(res.coreOp)) {
    if (mlir::isa<AIE::DeviceOp>(res.coreOp)) {
      // Op is inside the device but not inside any core — no CoreOp found.
      res.coreOp = nullptr;
      break;
    }
    res.coreOp = res.coreOp->getParentOp();
  }
  if (!res.coreOp)
    return res;

  mlir::Value coreTile = mlir::cast<AIE::CoreOp>(res.coreOp).getTile();
  auto lockIt = consumerTileLocks.find(coreTile);
  if (lockIt != consumerTileLocks.end()) {
    res.prodLock = lockIt->second.first;
    res.consLock = lockIt->second.second;
  }
  auto bufIt = consumerTileBuffers.find(coreTile);
  if (bufIt != consumerTileBuffers.end())
    res.buffers = &bufIt->second;
  auto rotIt = consumerTileRotationBufs.find(coreTile);
  if (rotIt != consumerTileRotationBufs.end())
    res.rotationBuf = rotIt->second;
  auto rotSlotIt = consumerTileRotationBufSlots.find(coreTile);
  if (rotSlotIt != consumerTileRotationBufSlots.end())
    res.rotationBufSlot = rotSlotIt->second;
  auto prodRotIt = producerTileRotationBufs.find(coreTile);
  if (prodRotIt != producerTileRotationBufs.end())
    res.producerRotationBuf = prodRotIt->second;
  auto prodRotSlotIt = producerTileRotationBufSlots.find(coreTile);
  if (prodRotSlotIt != producerTileRotationBufSlots.end())
    res.producerRotationBufSlot = prodRotSlotIt->second;
  return res;
}

// ---------------------------------------------------------------------------
// ConduitToDMAState
// ---------------------------------------------------------------------------

mlir::Value ConduitToDMAState::getMemTileDomain(mlir::Value tileVal) {
  auto tileOp = tileVal.getDefiningOp<AIE::TileOp>();
  if (!tileOp)
    return tileVal;
  int col = static_cast<int>(tileOp.getCol());
  int row = static_cast<int>(tileOp.getRow());
  // If this tile is already a MemTile, use it directly.
  if (targetModel && targetModel->isMemTile(col, row))
    return tileVal;
  // Look up the MemTile at (col, 1) in the tile cache.
  AIE::TileOp memTile = lookupTileByCoord(col, 1);
  if (memTile)
    return memTile.getResult();
  // No MemTile found — fall back to the tile itself as domain key.
  return tileVal;
}

AIE::TileOp ConduitToDMAState::lookupTileByCoord(int64_t col, int64_t row) {
  // Multi-device: use per-device cache to avoid cross-device collisions.
  if (activeDevIdx >= 0 &&
      activeDevIdx < static_cast<int>(perDevTileCache.size())) {
    auto &devCache = perDevTileCache[activeDevIdx];
    auto it = devCache.find({col, row});
    if (it != devCache.end())
      return it->second;
    return {};
  }
  // Single-device fallback: global cache.
  auto it = tileCache.find({col, row});
  if (it == tileCache.end())
    return {};
  return it->second;
}

mlir::Location ConduitToDMAState::getLocForTile(mlir::Value tileVal) {
  if (!tileVal)
    return deviceOp.getLoc();
  AIE::TileOp tileOp = tileVal.getDefiningOp<AIE::TileOp>();
  if (!tileOp)
    return deviceOp.getLoc();
  int64_t col = static_cast<int64_t>(tileOp.getCol());
  int64_t row = static_cast<int64_t>(tileOp.getRow());
  AIE::DeviceOp dev = getDeviceForTile(col, row);
  if (!dev)
    return deviceOp.getLoc();
  return dev.getLoc();
}

AIE::DeviceOp ConduitToDMAState::getDeviceForTile(int64_t col,
                                                  int64_t row) const {
  // Multi-device: use active device index.
  if (activeDevIdx >= 0 && activeDevIdx < static_cast<int>(deviceOps.size()))
    return deviceOps[activeDevIdx];
  auto cacheIt = tileCache.find({col, row});
  if (cacheIt == tileCache.end())
    return {};
  AIE::TileOp tile = cacheIt->second;
  // Walk parent chain: tile → DeviceOp.
  mlir::Operation *parent = tile->getParentOp();
  while (parent) {
    if (auto dev = mlir::dyn_cast<AIE::DeviceOp>(parent))
      return dev;
    parent = parent->getParentOp();
  }
  return {};
}

void ConduitToDMAState::switchToDeviceIndex(int devIdx) {
  if (devIdx < 0 || devIdx >= static_cast<int>(deviceOps.size()))
    return;
  activeDevIdx = devIdx;
  AIE::DeviceOp dev = deviceOps[devIdx];
  // FS2: keep state.deviceOp in sync with the active device.  Without this,
  // SymbolTable lookups in routePhase (e.g. checking for a pre-existing
  // shim_dma_allocation) always ran against device 0, missed real matches in
  // non-first devices, and emitted duplicate symbols → "redefinition of
  // symbol" verifier crash.
  deviceOp = dev;
  mlir::Block *body = &dev.getBodyRegion().front();
  if (body == deviceBody)
    return; // already pointing at the correct device
  deviceBody = body;
  insertAfterTile = nullptr;
  for (mlir::Operation &op : *deviceBody) {
    if (mlir::isa<AIE::TileOp>(op))
      insertAfterTile = &op;
  }
}

void ConduitToDMAState::switchDeviceForTile(int64_t col, int64_t row) {
  // Multi-device: use activeDevIdx (already set by switchToDeviceIndex).
  if (activeDevIdx >= 0) {
    switchToDeviceIndex(activeDevIdx);
    return;
  }
  AIE::DeviceOp dev = getDeviceForTile(col, row);
  if (!dev)
    return;
  mlir::Block *body = &dev.getBodyRegion().front();
  if (body == deviceBody)
    return; // already pointing at the correct device
  deviceBody = body;
  insertAfterTile = nullptr;
  for (mlir::Operation &op : *deviceBody) {
    if (mlir::isa<AIE::TileOp>(op))
      insertAfterTile = &op;
  }
}

void ConduitToDMAState::emitFlow(std::optional<RoutingMode> routingMode,
                                 mlir::Value srcTile, AIE::WireBundle srcBundle,
                                 int32_t srcChan, mlir::Value dstTile,
                                 AIE::WireBundle dstBundle, int32_t dstChan) {
  // Use the loc from the device that owns srcTile so that multi-device
  // modules assign correct source locations to emitted flow ops.
  mlir::Location loc = getLocForTile(srcTile);
  if (routingMode == RoutingMode::Packet) {
    // Allocate a packet flow ID; fail gracefully if budget is exhausted.
    if (!packetIDAllocator) {
      module.emitError("internal error: packetIDAllocator not initialized "
                       "before emitFlow");
      passFailed = true;
      return;
    }
    mlir::Value domain = getMemTileDomain(srcTile);
    std::optional<uint8_t> pktID = packetIDAllocator->allocate(domain);
    if (!pktID) {
      passFailed = true;
      return;
    }
    auto pktFlow =
        builder->create<AIE::PacketFlowOp>(loc, static_cast<int8_t>(*pktID),
                                           /*keep_pkt_header=*/mlir::BoolAttr{},
                                           /*priority_route=*/mlir::BoolAttr{});
    mlir::Region &region = pktFlow.getPorts();
    mlir::Block *block = builder->createBlock(&region);
    builder->setInsertionPointToStart(block);
    builder->create<AIE::PacketSourceOp>(loc, srcTile, srcBundle,
                                         static_cast<int32_t>(srcChan));
    builder->create<AIE::PacketDestOp>(loc, dstTile, dstBundle,
                                       static_cast<int32_t>(dstChan));
    builder->create<AIE::EndOp>(loc);
    builder->setInsertionPointAfter(pktFlow);
  } else {
    builder->create<AIE::FlowOp>(loc, srcTile, srcBundle, srcChan, dstTile,
                                 dstBundle, dstChan);
  }
}

ConduitToDMAState::AllocatedLocks
ConduitToDMAState::allocateLockPair(mlir::Value tileVal, llvm::StringRef prefix,
                                    int64_t depth, int64_t prodInit) {
  if (prodInit < 0)
    prodInit = depth;
  // Use the loc from the device that owns tileVal for correct multi-device
  // source location attribution on emitted lock ops.
  mlir::Location loc = getLocForTile(tileVal);
  AllocatedLocks locks;

  // Check lock ID budget before allocation. The hardware has a finite
  // number of lock IDs per tile (e.g., 16 on AIE2 compute tiles, 64 on
  // MemTiles). Exceeding this limit produces a verifier error:
  //   "aie.lock op lock assigned invalid id (maximum is N)"
  // Emit a diagnostic and set passFailed here so the error is actionable.
  if (targetModel) {
    auto tileOp = tileVal.getDefiningOp<AIE::TileOp>();
    if (tileOp) {
      uint32_t maxLocks = targetModel->getNumLocks(
          static_cast<int>(tileOp.getCol()), static_cast<int>(tileOp.getRow()));
      int locksNeeded = isAIE2Plus() ? 2 : static_cast<int>(depth);
      int currentUsed =
          lockIdCounter.count(tileVal) ? lockIdCounter[tileVal] : 0;
      if (currentUsed + locksNeeded > static_cast<int>(maxLocks)) {
        module.emitError(
            llvm::Twine("conduit-to-dma: lock ID exhausted on tile (") +
            llvm::Twine(tileOp.getCol()) + "," + llvm::Twine(tileOp.getRow()) +
            "): need " + llvm::Twine(locksNeeded) + " locks for '" + prefix +
            "' but only " +
            llvm::Twine(static_cast<int>(maxLocks) - currentUsed) + " of " +
            llvm::Twine(maxLocks) + " remain");
        passFailed = true;
        return locks;
      }
    }
  }

  if (isAIE2Plus()) {
    {
      int lockIdx = lockIdCounter[tileVal]++;
      std::string symName = (prefix + "_prod_lock_0").str();
      AIE::LockOp lk = builder->create<AIE::LockOp>(loc, tileVal, lockIdx,
                                                    static_cast<int>(prodInit));
      lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
      locks.prodLock = lk;
    }
    {
      int lockIdx = lockIdCounter[tileVal]++;
      std::string symName = (prefix + "_cons_lock_0").str();
      AIE::LockOp lk = builder->create<AIE::LockOp>(loc, tileVal, lockIdx,
                                                    static_cast<int>(0));
      lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
      locks.consLock = lk;
    }
  } else {
    for (int64_t i = 0; i < depth; ++i) {
      int lockIdx = lockIdCounter[tileVal]++;
      std::string symName = (prefix + "_lock_" + llvm::Twine(i)).str();
      AIE::LockOp lk = builder->create<AIE::LockOp>(loc, tileVal, lockIdx,
                                                    static_cast<int>(0));
      lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
      locks.aie1Locks.push_back(lk);
    }
    if (!locks.aie1Locks.empty()) {
      locks.prodLock = locks.aie1Locks[0];
      locks.consLock = locks.aie1Locks[0];
    }
  }
  return locks;
}

void ConduitToDMAState::emitBDBlock(mlir::Location loc, mlir::Block *block,
                                    mlir::Value acqLock, int32_t acqVal,
                                    mlir::Value buffer, int64_t offset,
                                    int64_t len, mlir::Value relLock,
                                    int32_t relVal,
                                    AIE::BDDimLayoutArrayAttr dims, int pktID) {
  if (!buffer) {
    mlir::emitError(loc,
                    "conduit-to-dma: emitBDBlock called with null buffer — "
                    "internal allocation error in Phase 3");
    return;
  }
  builder->setInsertionPointToEnd(block);
  if (acqLock)
    builder->create<AIE::UseLockOp>(loc, acqLock, acqAction, acqVal);
  if (pktID >= 0)
    builder->create<AIE::DMABDPACKETOp>(loc, /*pkt_type=*/0, pktID);
  if (buffer) {
    if (dims && !dims.getValue().empty())
      builder->create<AIE::DMABDOp>(loc, buffer, static_cast<int>(offset),
                                    static_cast<int>(len), dims);
    else
      builder->create<AIE::DMABDOp>(loc, buffer, static_cast<int>(offset),
                                    static_cast<int>(len));
  }
  if (relLock)
    builder->create<AIE::UseLockOp>(loc, relLock, AIE::LockAction::Release,
                                    relVal);
}

llvm::SmallVector<AIE::BufferOp>
ConduitToDMAState::allocateBuffers(mlir::Value tileVal, llvm::StringRef prefix,
                                   mlir::Type bufTy, int64_t count) {
  llvm::SmallVector<AIE::BufferOp> bufs;
  for (int64_t i = 0; i < count; ++i) {
    std::string symName = (prefix + "_buff_" + llvm::Twine(i)).str();
    auto buf =
        builder->create<AIE::BufferOp>(getLocForTile(tileVal), bufTy, tileVal,
                                       mlir::StringAttr::get(ctx, symName),
                                       /*address=*/mlir::IntegerAttr{},
                                       /*initial_value=*/mlir::ElementsAttr{},
                                       /*mem_bank=*/mlir::IntegerAttr{});
    bufs.push_back(buf);
  }
  return bufs;
}

std::string
ConduitToDMAState::makeConduitKey(llvm::StringRef name,
                                  mlir::Operation *contextOp) const {
  if (!isMultiDevice())
    return name.str();
  auto dev = contextOp->getParentOfType<AIE::DeviceOp>();
  if (!dev)
    return name.str();
  return name.str() + "__d" + std::to_string(getDeviceIndex(dev));
}

ConduitInfo *ConduitToDMAState::lookupConduit(mlir::StringRef name,
                                              mlir::Operation *contextOp) {
  if (!contextOp || !isMultiDevice())
    return lookupConduit(name);
  std::string key = makeConduitKey(name, contextOp);
  auto it = conduitMap.find(key);
  if (it != conduitMap.end())
    return &it->second;
  // Fallback to unqualified name (single-device or non-colliding).
  return lookupConduit(name);
}

} // namespace xilinx::conduit
