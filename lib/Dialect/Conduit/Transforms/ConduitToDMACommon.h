//===- ConduitToDMACommon.h - Shared types for ConduitToDMA split files
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
// Shared types, structures, and helper declarations used across all split
// files of the ConduitToDMA pass (Pass C).
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITTODMACOMMON_H
#define AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITTODMACOMMON_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

namespace xilinx::conduit {

// ---------------------------------------------------------------------------
// Helper: parse "tile(col,row)" → (col, row).  Returns {-1,-1} on failure.
// ---------------------------------------------------------------------------
inline std::pair<int64_t, int64_t> parseTileCoord(llvm::StringRef s) {
  if (!s.starts_with("tile("))
    return {-1, -1};
  s = s.drop_front(5); // drop "tile("
  s = s.drop_back(1);  // drop ")"
  auto [colStr, rowStr] = s.split(',');
  int64_t col, row;
  if (colStr.getAsInteger(10, col) || rowStr.getAsInteger(10, row))
    return {-1, -1};
  return {col, row};
}

// ---------------------------------------------------------------------------
// Per-conduit info gathered from conduit.create typed attributes.
//
// Populated incrementally across phases:
//   Phase 1 (Collect): producerTileCoord, consumerTileCoords,
//       shimConsumerTileCoords, depth, capacity, elemType, accessPattern,
//       routingMode, hasAllocTile, allocTileCoord, fuseGroup,
//       producerTileStr, consumerTileStrs
//   Phase 2.5 (Collect): effectiveDepth
//   Phase 3 (Alloc): buffers, prodLock, consLock, aie1Locks,
//       consumerTileLocks, consumerTileBuffers, consumerTileAIE1Locks,
//       rotationBuf, consumerTileRotationBufs, sharedMemory
// ---------------------------------------------------------------------------
struct ConduitInfo {
  // --- Populated by Phase 1 (collectConduitMap). ---

  // Tile coordinates parsed from the typed Create attributes.
  std::pair<int64_t, int64_t> producerTileCoord = {-1, -1};
  llvm::SmallVector<std::pair<int64_t, int64_t>> consumerTileCoords;
  // Shim consumer tiles (row==0): DMA endpoints, no local memory.
  llvm::SmallVector<std::pair<int64_t, int64_t>> shimConsumerTileCoords;
  int64_t depth = 1;
  int64_t capacity = 0;
  mlir::Type elemType; // actual element memref type (may be null)
  // Cyclostatic (CSDF) access pattern from conduit.create access_pattern attr.
  // Empty = uniform SDF; non-empty = CSDF per-iteration acquire counts.
  llvm::SmallVector<int64_t> accessPattern;
  // Routing mode: "circuit" (default) or "packet".
  std::string routingMode = "circuit";
  // Alloc tile override from objectfifo.allocate delegate tile.
  bool hasAllocTile = false;
  std::pair<int64_t, int64_t> allocTileCoord = {-1, -1};
  // Legacy string form for Link memtile lookup.
  std::string producerTileStr; // "tile(col,row)"
  llvm::SmallVector<std::string> consumerTileStrs;
  // DMA channel fusion group label (from --conduit-fuse-channels annotation).
  std::string fuseGroup;

  // --- Populated by Phase 2.5 (computeEffectiveDepth). ---

  // Producer-side effective depth: min(depth, maxProdAcquire+1).
  // 0 means "use raw depth" (no optimization).
  int64_t effectiveDepth = 0;

  // --- Populated by Phase 3 (allocateBuffersAndLocks). ---

  // Shared memory flag: set when producer and consumer are adjacent tiles.
  // When true, buffers/locks go on the producer (or alloc) tile; no DMA.
  bool sharedMemory = false;

  // Hardware SSA values:
  llvm::SmallVector<AIE::BufferOp> buffers; // depth-many on consumer_tile[0]

  // Per-consumer-tile lock pairs for multi-consumer (broadcast) correctness.
  // Key: tile SSA Value.  Read by Phase 5.5 and Phase 6.
  // NOTE: for sharedMemory conduits, LockOps are physically on the producer
  // tile but keyed on the consumer tile for Phase 6 lookup.
  llvm::DenseMap<mlir::Value, std::pair<AIE::LockOp, AIE::LockOp>>
      consumerTileLocks; // tile → (prodLock, consLock)

  // Per-consumer-tile buffer vectors for SubviewAccess resolution.
  llvm::DenseMap<mlir::Value, llvm::SmallVector<AIE::BufferOp>>
      consumerTileBuffers; // tile → [buff_0, ..., buff_{depth-1}]

  // Convenience accessors for single-consumer and link phase.
  AIE::LockOp prodLock; // prod lock (init=depth)
  AIE::LockOp consLock; // cons lock (init=0)

  // AIE1 per-slot locks: one lock per buffer slot (depth-many).
  // Empty for AIE2.
  llvm::SmallVector<AIE::LockOp> aie1Locks;

  // Per-consumer-tile AIE1 lock vectors (for multi-consumer broadcast).
  llvm::DenseMap<mlir::Value, llvm::SmallVector<AIE::LockOp>>
      consumerTileAIE1Locks; // tile → [lock_0, ..., lock_{depth-1}]

  // For depth>1: rotation counter buffer on the consumer tile.
  AIE::BufferOp rotationBuf;
  llvm::DenseMap<mlir::Value, AIE::BufferOp>
      consumerTileRotationBufs; // tile → rotation counter buffer

  // --- Helper methods ---

  // Result of resolving per-tile resources from the enclosing CoreOp.
  struct ResolvedTileResources {
    AIE::LockOp prodLock;
    AIE::LockOp consLock;
    llvm::SmallVector<AIE::BufferOp> *buffers = nullptr;
    AIE::BufferOp rotationBuf;
    mlir::Operation *coreOp = nullptr;
  };

  // Resolve per-tile locks, buffers, and rotation counter for an op
  // inside a CoreOp.  Walks the parent chain to find the enclosing CoreOp,
  // then looks up per-tile overrides in consumerTileLocks/Buffers/RotationBufs.
  ResolvedTileResources resolveForTile(mlir::Operation *op) {
    ResolvedTileResources res;
    res.prodLock = prodLock;
    res.consLock = consLock;
    res.buffers = &buffers;
    res.rotationBuf = rotationBuf;

    res.coreOp = op->getParentOp();
    while (res.coreOp && !mlir::isa<AIE::CoreOp>(res.coreOp))
      res.coreOp = res.coreOp->getParentOp();
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
    return res;
  }
};

// ---------------------------------------------------------------------------
// Metadata for an acquire_async op, recorded before erasure so that
// wait_window and wait_all can look up lock info after the op is gone.
// Populated by Phase 8a, read by Phase 8b/8c.
// ---------------------------------------------------------------------------
struct AsyncAcquireInfo {
  std::string conduitName;
  Port port;
  int64_t count;
};

// ---------------------------------------------------------------------------
// Shared pass state passed to all phase functions.
//
// Owns the conduitMap and all auxiliary data structures that must survive
// across phases.  Each phase function takes a reference to this struct and
// modifies it in place.
// ---------------------------------------------------------------------------
struct ConduitToDMAState {
  // Module and device references.
  mlir::ModuleOp module;
  AIE::DeviceOp deviceOp;
  mlir::OpBuilder *builder;
  mlir::MLIRContext *ctx;

  // Target model queries.
  const AIE::AIETargetModel *targetModel = nullptr;
  AIE::AIEArch aieArch = AIE::AIEArch::AIE1;
  AIE::LockAction acqAction;

  // Convenience: true for AIE2 and AIE2p (all non-AIE1 architectures).
  bool isAIE2Plus() const { return aieArch != AIE::AIEArch::AIE1; }

  // AIE1 BD block lock value constants.
  // S2MM: acquire(0) [empty], release(1) [full].
  // MM2S: acquire(1) [full], release(0) [empty].
  static constexpr int32_t aie1S2MMAcqVal = 0;
  static constexpr int32_t aie1S2MMRelVal = 1;
  static constexpr int32_t aie1MM2SAcqVal = 1;
  static constexpr int32_t aie1MM2SRelVal = 0;

  // Conduit metadata map.  MapVector preserves insertion order (= source
  // order) for deterministic iteration in allocation and BD generation.
  // Key: conduit name (std::string — owning, safe across op erasure).
  // Uses StringMap<unsigned> for the index (DenseMap<std::string, ...> lacks
  // DenseMapInfo specialization in LLVM).
  llvm::MapVector<std::string, ConduitInfo,
                  llvm::StringMap<unsigned>> conduitMap;

  // Tile cache: (col, row) → TileOp SSA value.
  llvm::DenseMap<std::pair<int64_t, int64_t>, AIE::TileOp> tileCache;

  // Device body reference and insertion point after last tile op.
  mlir::Block *deviceBody = nullptr;
  mlir::Operation *insertAfterTile = nullptr;

  // Per-tile lock ID counter to avoid collisions.
  llvm::DenseMap<mlir::Value, int> lockIdCounter;

  // Per-tile DMA channel counters for flow emission and BD chain creation.
  llvm::DenseMap<mlir::Value, int32_t> tileNextMM2SChannel;
  llvm::DenseMap<mlir::Value, int32_t> tileNextS2MMChannel;

  // Per-conduit assigned channel indices.
  llvm::StringMap<int32_t> conduitMM2SChannel;
  // Per-conduit per-consumer S2MM channel.
  // Key: {conduit_name, consumer_index}.
  std::map<std::pair<std::string, unsigned>, int32_t> conduitConsS2MMChannel;

  // Link source names for skip logic.
  llvm::StringSet<> linkSrcNamesEarly;  // distribute sources only
  llvm::StringSet<> linkJoinSrcNames;   // join sources
  llvm::StringSet<> linkSrcNames;       // all link sources (both)

  // Conduit names with at least one Consume-port acquire op.
  llvm::StringSet<> conduitNamesWithConsumerAcquire;

  // Shim conduit names for Phase 4.5 symbol rewriting.
  llvm::StringSet<> shimConduitNames;

  // Packet flow ID counter.
  int packetFlowID = 0;

  // Fuse group tracking for Phase 4.5a and Phase 5.5.
  llvm::StringMap<int32_t> fuseGroupMM2SChannel;
  llvm::StringMap<llvm::SmallVector<std::string, 4>> fuseGroupMembers;

  // Pre-computed used DMA channels per tile (populated before Phase 5.5).
  llvm::DenseMap<mlir::Value, llvm::DenseSet<int32_t>> preUsedMM2SChannels;
  llvm::DenseMap<mlir::Value, llvm::DenseSet<int32_t>> preUsedS2MMChannels;

  // BD range tracking for fused channel groups (Phase 5.5 post-pass).
  llvm::StringMap<std::pair<mlir::Block *, mlir::Block *>> conduitBDRange;

  // Async acquire metadata for Phase 8.
  llvm::DenseMap<mlir::Value, AsyncAcquireInfo> asyncAcquireMap;

  // Error flag: set by any phase to signal pass failure.
  bool passFailed = false;

  // --- Helper methods ---

  AIE::TileOp lookupTile(llvm::StringRef coord) {
    auto [col, row] = parseTileCoord(coord);
    if (col < 0)
      return {};
    auto it = tileCache.find({col, row});
    if (it == tileCache.end())
      return {};
    return it->second;
  }

  AIE::TileOp lookupTileByCoord(int64_t col, int64_t row) {
    auto it = tileCache.find({col, row});
    if (it == tileCache.end())
      return {};
    return it->second;
  }

  // Emit a circuit or packet flow between two tiles.
  void emitFlow(llvm::StringRef routingMode, mlir::Value srcTile,
                AIE::WireBundle srcBundle, int32_t srcChan,
                mlir::Value dstTile, AIE::WireBundle dstBundle,
                int32_t dstChan) {
    if (routingMode == "packet") {
      auto pktFlow = builder->create<AIE::PacketFlowOp>(
          deviceOp.getLoc(),
          static_cast<int8_t>(packetFlowID++ & 0x7F),
          /*keep_pkt_header=*/mlir::BoolAttr{},
          /*priority_route=*/mlir::BoolAttr{});
      mlir::Region &region = pktFlow.getPorts();
      mlir::Block *block = builder->createBlock(&region);
      builder->setInsertionPointToStart(block);
      builder->create<AIE::PacketSourceOp>(deviceOp.getLoc(), srcTile,
                                           srcBundle,
                                           static_cast<int32_t>(srcChan));
      builder->create<AIE::PacketDestOp>(deviceOp.getLoc(), dstTile,
                                         dstBundle,
                                         static_cast<int32_t>(dstChan));
      builder->create<AIE::EndOp>(deviceOp.getLoc());
      builder->setInsertionPointAfter(pktFlow);
    } else {
      builder->create<AIE::FlowOp>(deviceOp.getLoc(), srcTile, srcBundle,
                                   srcChan, dstTile, dstBundle, dstChan);
    }
  }

  // Result of allocating a lock pair (AIE2) or per-slot locks (AIE1).
  struct AllocatedLocks {
    AIE::LockOp prodLock;
    AIE::LockOp consLock;
    llvm::SmallVector<AIE::LockOp> aie1Locks;
  };

  // Allocate a producer/consumer lock pair on the given tile.
  // AIE2: emits prod_lock (init=depth) + cons_lock (init=0).
  // AIE1: emits depth-many per-slot locks (init=0); prod=cons=locks[0].
  AllocatedLocks allocateLockPair(mlir::Value tileVal,
                                  llvm::StringRef prefix, int64_t depth) {
    AllocatedLocks locks;
    if (isAIE2Plus()) {
      {
        int lockIdx = lockIdCounter[tileVal]++;
        std::string symName = (prefix + "_prod_lock_0").str();
        AIE::LockOp lk = builder->create<AIE::LockOp>(
            deviceOp.getLoc(), tileVal, lockIdx, static_cast<int>(depth));
        lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        locks.prodLock = lk;
      }
      {
        int lockIdx = lockIdCounter[tileVal]++;
        std::string symName = (prefix + "_cons_lock_0").str();
        AIE::LockOp lk = builder->create<AIE::LockOp>(
            deviceOp.getLoc(), tileVal, lockIdx, static_cast<int>(0));
        lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        locks.consLock = lk;
      }
    } else {
      for (int64_t i = 0; i < depth; ++i) {
        int lockIdx = lockIdCounter[tileVal]++;
        std::string symName =
            (prefix + "_lock_" + llvm::Twine(i)).str();
        AIE::LockOp lk = builder->create<AIE::LockOp>(
            deviceOp.getLoc(), tileVal, lockIdx, static_cast<int>(0));
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

  // Allocate `count` buffers of type `bufTy` on the given tile.
  llvm::SmallVector<AIE::BufferOp> allocateBuffers(
      mlir::Value tileVal, llvm::StringRef prefix,
      mlir::Type bufTy, int64_t count) {
    llvm::SmallVector<AIE::BufferOp> bufs;
    for (int64_t i = 0; i < count; ++i) {
      std::string symName =
          (prefix + "_buff_" + llvm::Twine(i)).str();
      auto buf = builder->create<AIE::BufferOp>(
          deviceOp.getLoc(), bufTy, tileVal,
          mlir::StringAttr::get(ctx, symName),
          /*address=*/mlir::IntegerAttr{},
          /*initial_value=*/mlir::ElementsAttr{},
          /*mem_bank=*/mlir::IntegerAttr{});
      bufs.push_back(buf);
    }
    return bufs;
  }
};

// ---------------------------------------------------------------------------
// Phase function declarations.  Each phase function modifies state in place.
// If a phase detects an error, it sets state.passFailed = true.
// ---------------------------------------------------------------------------

/// Phase 1: Collect ConduitInfo from conduit.create ops into conduitMap.
/// Phase 2: Find aie.device, build tile cache, determine aieArch.
/// Phase 2.5: Compute effectiveDepth for producer-side buffer optimization.
/// Also collects link source names and consumer acquire name sets.
void collectPhase(ConduitToDMAState &state);

/// Phase 3: Allocate aie.buffer + aie.lock pairs for each conduit.
/// Covers Phase 3b (shim consumer), Phase 3c (shared memory),
/// Phase 3j (join sources), Phase 3d (non-adjacent producer-side).
void allocPhase(ConduitToDMAState &state);

/// Phase 4: Shim DMA allocation + flow emission.
/// Phase 4.5: Symbol rewriting for shim_dma_allocation.
/// Phase 4.5a: Non-adjacent conduit flow emission.
void routePhase(ConduitToDMAState &state);

/// Phase 5: Lower conduit.link → MemTile DMA BD chain.
/// Phase 5.5: Generate aie.mem BD chains for simple (non-link) conduits.
/// Phase 5.5 post-pass: Link fused BD chains.
void linkPhase(ConduitToDMAState &state);

/// Phase 6: Lower conduit.acquire/release → aie.use_lock.
/// Phase 7: Erase remaining Conduit ops (create, wait, wait_all_async).
/// Phase 8: Lower async acquire/release/wait_window/wait_all.
/// Steps 8e-8h: Erase put/get memref ops.
void lowerPhase(ConduitToDMAState &state);

} // namespace xilinx::conduit

#endif // AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITTODMACOMMON_H
