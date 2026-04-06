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

#include <optional>
#include <set>
#include <string>

namespace xilinx::conduit {

// ---------------------------------------------------------------------------
// PacketIDAllocator: compile-time packet flow ID counter with exhaustion check.
//
// AIE hardware has a finite number of distinct packet flow IDs. AIE1 supports
// up to 32 IDs (5-bit field); AIE2 also supports up to 32 IDs. If more than
// `limit` packet flows are emitted, data corruption occurs silently at runtime
// because the hardware reuses IDs. This allocator enforces the limit at
// compile time, emitting a hard error when the budget is exceeded.
//
// Instantiated in ConduitToDMAPass.cpp with the architecture-specific limit
// (queried from AIETargetModel if available; defaults to 32).
// ---------------------------------------------------------------------------
struct PacketIDAllocator {
  mlir::ModuleOp module;
  uint8_t next = 0;
  uint8_t limit; // from AIETargetModel or default 32

  explicit PacketIDAllocator(mlir::ModuleOp mod, uint8_t lim = 32)
      : module(mod), limit(lim) {}

  std::optional<uint8_t> allocate() {
    if (next >= limit) {
      module.emitError("packet flow ID exhausted: design requires more than ")
          << (unsigned)limit << " distinct packet flows";
      return std::nullopt;
    }
    return next++;
  }

  uint8_t remaining() const { return limit - next; }
};

// ---------------------------------------------------------------------------
// PacketChannelState: module-level state for Step 3.5 packet DMA fallback.
//
// Tracks two pieces of information needed for safe packet-mode selection
// when circuit DMA channels are exhausted (mode=any fallback):
//
//   isPacketChannel: for each (tile_op_ptr, mm2s_channel_index) pair, whether
//     that physical MM2S channel has been designated for packet use.  Once
//     designated, the channel is shared by multiple logical packet flows (each
//     with a distinct flow ID); circuit-mode flows may not use it.
//
//   portOccupancy: for each packet-mode MM2S channel (identified by an
//     int64_t key combining tile ptr and channel index), the list of
//     (flow_id, dst_tile_op*) pairs already routed through it.  Used for the
//     convergence hazard check (Step 3.5d): two packet flows on the same
//     physical channel that route to the same consumer tile D create an
//     ordering hazard under sustained load.
//
// Initialized in ConduitToDMAPass.cpp at Pass C entry.
// Maintained across all conduits during Phase 4.5a flow emission.
// ---------------------------------------------------------------------------
struct PacketChannelState {
  // Per (tile_op_ptr, channel_index): is this MM2S channel packet-mode?
  llvm::DenseMap<std::pair<mlir::Operation *, int>, bool> isPacketChannel;

  // Per packet-mode MM2S port: (flow_id, dst_tile_op*) pairs routing through.
  // Key: portKey(tileOp, channel).
  llvm::DenseMap<int64_t,
                 llvm::SmallVector<std::pair<uint8_t, mlir::Operation *>>>
      portOccupancy;

  // Build a stable int64_t key for portOccupancy from a (tile, channel) pair.
  // Uses the lower 56 bits of the tile pointer + 8 bits of channel index.
  // Collision probability is negligible for designs with <256^7 tiles.
  static int64_t portKey(mlir::Operation *tileOp, int channel) {
    auto addr = reinterpret_cast<uintptr_t>(tileOp);
    return static_cast<int64_t>((addr & 0x00FFFFFFFFFFFFFFULL) << 8)
           | static_cast<int64_t>(channel & 0xFF);
  }
};

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
  // Element count per DMA transfer, from put/get_memref_async {num_elems=N}.
  // Populated by Phase 1 collect; used by Phase 5.5 BD chain for Tier 3
  // channels where capacity encodes slot count (not element count).
  int64_t numElems = 0;
  mlir::Type elemType; // actual element memref type (may be null)
  // Cyclostatic (CSDF) access pattern from conduit.create access_pattern attr.
  // Empty = uniform SDF; non-empty = CSDF per-iteration acquire counts.
  llvm::SmallVector<int64_t> accessPattern;
  // Routing mode: "circuit" (default), "packet", "cascade", or "stream".
  std::string routingMode = "circuit";
  // Core stream port index for routing_mode="stream" (-1 if not stream).
  int32_t aieStreamPort = -1;
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

  // Partial-release buffer count adjustment.
  // Populated by Phase 2.6 in collectPhase.
  //
  // maxConsumerAcquire: maximum acquire count seen across all Consume-port
  //   acquire/release pairs where acquireCount > releaseCount (sliding window).
  //   0 = no sliding window (normal SDF or CSDF with full release per step).
  //
  // nConsumerBuffers() uses: max(depth, maxConsumerAcquire + 1).
  // Derivation: a K-tap sliding window needs K+1 physical buffers minimum
  // (K held by consumer + 1 being filled by DMA). If depth > K+1, use depth.
  int64_t maxConsumerAcquire = 0;

  // maxProduceAcquire: maximum acquire count seen across all Produce-port
  //   acquire/release pairs where acquireCount > releaseCount (producer
  //   sliding window). 0 = no producer sliding window.
  //
  // nProducerBuffers() uses: max(depth, maxProduceAcquire + 1).
  // Derivation: same as consumer — a sliding-window producer that holds K
  // output buffers simultaneously needs K+1 physical slots so the DMA engine
  // can drain one slot while the core holds the rest.
  int64_t maxProduceAcquire = 0;

  // --- Populated by Phase 3 (allocateBuffersAndLocks). ---

  // Shared memory flag: set when producer and consumer are adjacent tiles.
  // When true, buffers/locks go on the producer (or alloc) tile; no DMA.
  bool sharedMemory = false;

  // External buffers (from conduit.register_external_buffers).
  // When non-empty, the shim DMA BD chain uses these instead of allocated
  // buffers, and internal buffer allocation for the shim tile is skipped.
  llvm::SmallVector<mlir::Value> externalBuffers;
  // Tile coordinate of the shim endpoint associated with externalBuffers.
  std::pair<int64_t, int64_t> externalBufferTileCoord = {-1, -1};

  // Shim-tile locks for shim producer conduits (Phase 4a → Phase 5.5).
  // Populated by Phase 4a when the producer tile is a shim.
  AIE::LockOp shimProdLock;
  AIE::LockOp shimConsLock;

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
  AIE::LockOp prodLock; // prod lock (init=depth*repeatN for compute tiles;
                        // init=0 for shim tiles — managed by host runtime)
  AIE::LockOp consLock; // cons lock (init=0)

  // AIE1 per-slot locks: one lock per buffer slot (depth-many).
  // Empty for AIE2.
  llvm::SmallVector<AIE::LockOp> aie1Locks;

  // Per-consumer-tile AIE1 lock vectors (for multi-consumer broadcast).
  llvm::DenseMap<mlir::Value, llvm::SmallVector<AIE::LockOp>>
      consumerTileAIE1Locks; // tile → [lock_0, ..., lock_{depth-1}]

  // For depth>1: rotation counter — shared per-tile alloca + slot index.
  // Multiple conduits on the same tile share one memref<N xi32> alloca
  // (stack-allocated inside the core body); each conduit is assigned a
  // unique slot index within it.  Using alloca instead of aie.buffer
  // eliminates spurious buffer entries in the device-level IR.
  mlir::Value rotationBuf;           // shared tile alloca (consumer direction)
  int64_t rotationBufSlot = 0;       // slot index within that alloca
  llvm::DenseMap<mlir::Value, mlir::Value>
      consumerTileRotationBufs;      // tile → shared rotation alloca
  llvm::DenseMap<mlir::Value, int64_t>
      consumerTileRotationBufSlots;  // tile → slot index for this conduit

  // For depth>1 produce-mode: rotation counter on the producer tile.
  mlir::Value producerRotationBuf;   // shared tile alloca (producer direction)
  int64_t producerRotationBufSlot = 0; // slot index within that alloca
  llvm::DenseMap<mlir::Value, mlir::Value>
      producerTileRotationBufs;      // tile → shared rotation alloca
  llvm::DenseMap<mlir::Value, int64_t>
      producerTileRotationBufSlots;  // tile → slot index for this conduit

  // --- New feature flags (populated by Phase 1 from conduit.create attrs) ---

  // disable_synchronization: suppress all lock allocation and use_lock ops.
  bool disableSynchronization = false;
  // via_DMA: force DMA routing even for adjacent tiles (skip shared-mem path).
  bool viaDMA = false;
  // plio: when true, the shim endpoint uses Platform I/O instead of DMA.
  // Flows use WireBundle::PLIO and shim_dma_allocation carries {plio = true}.
  bool plio = false;
  // iter_count: number of DMA iterations (> 0 → DMAStartOp repeat_count = K-1,
  // non-circular BD chain with terminal aie.end).
  int64_t iterCount = 0;
  // bdChainRepeatCount: from objectfifo repeat_count.  Number of times each BD
  // fires before advancing to the next buffer.  0/1 = once (no unrolling).
  int64_t bdChainRepeatCount = 0;
  // Producer-side BDDimLayout descriptor (may be null/empty).
  AIE::BDDimLayoutArrayAttr producerDimensions;
  // Per-consumer BDDimLayout descriptors (parallel to consumerTileCoords).
  llvm::SmallVector<AIE::BDDimLayoutArrayAttr> consumerDimensions;

  // --- Helper methods ---

  // Compute the consumer-side buffer count for this conduit.
  // For the sliding-window pattern (acquire_count > release_count on a paired
  // acquire/release), extra buffer slots are needed to hold the unreleased
  // elements while the DMA pre-fills the next slot.
  // Formula: max(depth, maxConsumerAcquire + 1)
  // Derivation: a K-tap sliding window needs K+1 physical buffers minimum
  // (K held by consumer + 1 for DMA). If depth > K+1, depth is used.
  // maxConsumerAcquire = 0 for normal SDF/CSDF (full release per step).
  int64_t nConsumerBuffers() const {
    int64_t d = depth > 0 ? depth : 1;
    if (maxConsumerAcquire <= 0)
      return d;
    return std::max(d, maxConsumerAcquire + 1);
  }

  // Compute the producer-side buffer count for this conduit.
  // Mirrors nConsumerBuffers() for the Produce port: when a producer acquires
  // K output slots but releases fewer than K per step, extra slots must be
  // allocated so the DMA engine can drain one slot while the core holds the rest.
  // Formula: max(depth, maxProduceAcquire + 1)
  // maxProduceAcquire = 0 for normal (non-sliding-window) producers.
  int64_t nProducerBuffers() const {
    int64_t d = depth > 0 ? depth : 1;
    if (maxProduceAcquire <= 0)
      return d;
    return std::max(d, maxProduceAcquire + 1);
  }

  // Result of resolving per-tile resources from the enclosing CoreOp.
  struct ResolvedTileResources {
    AIE::LockOp prodLock;
    AIE::LockOp consLock;
    llvm::SmallVector<AIE::BufferOp> *buffers = nullptr;
    mlir::Value rotationBuf;             // shared tile alloca (consumer dir)
    int64_t rotationBufSlot = 0;         // slot index within that alloca
    mlir::Value producerRotationBuf;     // shared tile alloca (producer dir)
    int64_t producerRotationBufSlot = 0; // slot index within that alloca
    mlir::Operation *coreOp = nullptr;
  };

  // Resolve per-tile locks, buffers, and rotation counters for an op
  // inside a CoreOp.  Walks the parent chain to find the enclosing CoreOp,
  // then looks up per-tile overrides in consumerTileLocks/Buffers/RotationBufs.
  ResolvedTileResources resolveForTile(mlir::Operation *op) {
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

  /// Lock acquire value for DMA BD chains and core-side operations.
  /// Port::Produce (S2MM / core-produces): AIE1 acquires empty slot (0).
  /// Port::Consume (MM2S / core-consumes): AIE1 acquires full slot (1).
  /// AIE2+: always returns count (AcquireGreaterEqual semantics).
  int32_t lockAcqValue(Port port, int32_t count) const {
    if (isAIE2Plus())
      return count;
    return (port == Port::Consume) ? 1 : 0;
  }

  /// Lock release value for DMA BD chains and core-side operations.
  /// Port::Produce (S2MM / core-produces): AIE1 releases full slot (1).
  /// Port::Consume (MM2S / core-consumes): AIE1 releases empty slot (0).
  /// AIE2+: always returns count.
  int32_t lockRelValue(Port port, int32_t count = 1) const {
    if (isAIE2Plus())
      return count;
    return (port == Port::Consume) ? 0 : 1;
  }

  // Conduit metadata map.  MapVector preserves insertion order (= source
  // order) for deterministic iteration in allocation and BD generation.
  // Key: conduit name (std::string — owning, safe across op erasure).
  // Uses StringMap<unsigned> for the index (DenseMap<std::string, ...> lacks
  // DenseMapInfo specialization in LLVM).
  llvm::MapVector<std::string, ConduitInfo, llvm::StringMap<unsigned>>
      conduitMap;

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
  llvm::StringSet<> linkSrcNamesEarly; // distribute sources only
  llvm::StringSet<> linkJoinSrcNames;  // join sources
  llvm::StringSet<> linkSrcNames;      // all link sources (both)
  llvm::StringSet<> linkDstNames;      // all link destinations (both)

  // Conduit names with at least one Consume-port acquire op.
  llvm::StringSet<> conduitNamesWithConsumerAcquire;
  // Conduit names with at least one Produce-port acquire op (for producer
  // rotation counter allocation when depth > 1).
  llvm::StringSet<> conduitNamesWithProducerAcquire;

  // Per-tile shared rotation counter alloca pool.
  // Populated by allocPhase() pre-scan; each tile that needs N rotation
  // counters gets one memref<N xi32> alloca (inside the core body) shared
  // across all conduits on that tile.
  llvm::DenseMap<mlir::Value, mlir::Value> tileRotationBuf;
  llvm::DenseMap<mlir::Value, int64_t> tileRotationBufNextSlot;

  // Shim conduit names for Phase 4.5 symbol rewriting.
  llvm::StringSet<> shimConduitNames;

  // Packet flow ID allocator (replaces raw counter; initialized in Pass shell).
  // Use std::optional so the state struct can be default-constructed before
  // the module and architecture limit are known.
  std::optional<PacketIDAllocator> packetIDAllocator;

  // Packet channel state for Step 3.5 mode=any fallback.
  // Tracks which MM2S channels have been designated for packet use, and which
  // (flow_id, dst_tile) pairs are routed through each packet-mode channel.
  PacketChannelState pktChannelState;

  // Per-tile BD budget used (number of BD slots consumed so far).
  // Incremented by `depth` whenever a conduit allocates BD chains on a tile.
  // Used by Step 3.5b to check whether the BD budget allows a new flow.
  llvm::DenseMap<mlir::Value, int32_t> tileBDUsed;

  // Fuse group tracking for Phase 4.5a and Phase 5.5.
  llvm::StringMap<int32_t> fuseGroupMM2SChannel;
  llvm::StringMap<llvm::SmallVector<std::string, 4>> fuseGroupMembers;

  // Pre-computed used DMA channels per tile (populated before Phase 5.5).
  llvm::DenseMap<mlir::Value, llvm::DenseSet<int32_t>> preUsedMM2SChannels;
  llvm::DenseMap<mlir::Value, llvm::DenseSet<int32_t>> preUsedS2MMChannels;

  // BD range tracking for fused channel groups (Phase 5.5 post-pass).
  llvm::StringMap<std::pair<mlir::Block *, mlir::Block *>> conduitBDRange;

  // Per-conduit packet flow ID for MM2S BD packet headers (aie.dma_bd_packet).
  // For packet-mode channels, each MM2S BD needs a packet header matching the
  // packet flow ID so the switchbox can route data to the correct destination.
  // Key: conduit name; Value: packet ID (0-31).
  // Populated by routePhase. Read by linkPhase for Phase 5.5 BD emission.
  llvm::StringMap<uint8_t> conduitPacketID;

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
  // For packet flows, the packet ID is allocated from packetIDAllocator.
  // If the ID budget is exhausted, passFailed is set and the flow is not
  // emitted (the error is reported by the allocator on the module op).
  void emitFlow(llvm::StringRef routingMode, mlir::Value srcTile,
                AIE::WireBundle srcBundle, int32_t srcChan, mlir::Value dstTile,
                AIE::WireBundle dstBundle, int32_t dstChan) {
    if (routingMode == "packet") {
      // Allocate a packet flow ID; fail gracefully if budget is exhausted.
      if (!packetIDAllocator) {
        module.emitError(
            "internal error: packetIDAllocator not initialized before emitFlow");
        passFailed = true;
        return;
      }
      std::optional<uint8_t> pktID = packetIDAllocator->allocate();
      if (!pktID) {
        passFailed = true;
        return;
      }
      auto pktFlow = builder->create<AIE::PacketFlowOp>(
          deviceOp.getLoc(),
          static_cast<int8_t>(*pktID),
          /*keep_pkt_header=*/mlir::BoolAttr{},
          /*priority_route=*/mlir::BoolAttr{});
      mlir::Region &region = pktFlow.getPorts();
      mlir::Block *block = builder->createBlock(&region);
      builder->setInsertionPointToStart(block);
      builder->create<AIE::PacketSourceOp>(
          deviceOp.getLoc(), srcTile, srcBundle, static_cast<int32_t>(srcChan));
      builder->create<AIE::PacketDestOp>(deviceOp.getLoc(), dstTile, dstBundle,
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
  // AIE2: emits prod_lock (init=prodInit) + cons_lock (init=0).
  //   prodInit defaults to depth; pass a different value for repeat_count
  //   scaling.
  // AIE1: emits depth-many per-slot locks (init=0); prod=cons=locks[0].
  AllocatedLocks allocateLockPair(mlir::Value tileVal, llvm::StringRef prefix,
                                  int64_t depth, int64_t prodInit = -1) {
    if (prodInit < 0)
      prodInit = depth;
    AllocatedLocks locks;
    if (isAIE2Plus()) {
      {
        int lockIdx = lockIdCounter[tileVal]++;
        std::string symName = (prefix + "_prod_lock_0").str();
        AIE::LockOp lk = builder->create<AIE::LockOp>(
            deviceOp.getLoc(), tileVal, lockIdx, static_cast<int>(prodInit));
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
        std::string symName = (prefix + "_lock_" + llvm::Twine(i)).str();
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

  /// Look up a conduit by name in the conduitMap.
  /// Returns nullptr if not found.
  ConduitInfo *lookupConduit(mlir::StringRef name) {
    auto it = conduitMap.find(name.str());
    if (it == conduitMap.end())
      return nullptr;
    return &it->second;
  }

  /// Emit DMA BD block content into an existing block:
  ///   1. UseLockOp (acquire) — skipped if acqLock is null
  ///   2. DMABDPACKETOp — skipped if pktID < 0; sets packet header for
  ///      packet-switched DMA routing
  ///   3. DMABDOp — skipped if buffer is null; emits BDDimLayout if dims
  ///   non-empty
  ///   4. UseLockOp (release) — skipped if relLock is null
  /// Sets the builder insertion point to the end of the block.
  /// NextBDOp is NOT emitted; the caller controls ring linkage.
  void emitBDBlock(mlir::Location loc, mlir::Block *block, mlir::Value acqLock,
                   int32_t acqVal, mlir::Value buffer, int64_t offset,
                   int64_t len, mlir::Value relLock, int32_t relVal,
                   AIE::BDDimLayoutArrayAttr dims = {},
                   int pktID = -1) {
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

  // Allocate `count` buffers of type `bufTy` on the given tile.
  llvm::SmallVector<AIE::BufferOp> allocateBuffers(mlir::Value tileVal,
                                                   llvm::StringRef prefix,
                                                   mlir::Type bufTy,
                                                   int64_t count) {
    llvm::SmallVector<AIE::BufferOp> bufs;
    for (int64_t i = 0; i < count; ++i) {
      std::string symName = (prefix + "_buff_" + llvm::Twine(i)).str();
      auto buf =
          builder->create<AIE::BufferOp>(deviceOp.getLoc(), bufTy, tileVal,
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

/// Phase 5: Lower conduit.distribute/join/forward → MemTile DMA BD chain.
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
