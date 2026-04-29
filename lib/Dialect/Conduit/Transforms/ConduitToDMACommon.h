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
// files of the --conduit-to-dma pass.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITTODMACOMMON_H
#define AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITTODMACOMMON_H

#include "ConduitTileInference.h"

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
// AIE hardware has a finite number of distinct packet flow IDs per MemTile
// domain. AIE1 and AIE2 both support up to 32 IDs (5-bit field, values 0–31)
// per domain. Packet IDs are scoped per-MemTile: each MemTile gets its own
// 0–31 ID space, so designs with multiple MemTile columns can reuse IDs
// across columns without conflict.
//
// Callers pass a `domain` Value (typically the MemTile tile value) to
// allocate() and remaining(). Flows that route through the same MemTile
// share one 0–31 budget; flows through different MemTiles are independent.
//
// Instantiated in ConduitToDMAPass.cpp with the architecture-specific limit
// (queried from AIETargetModel if available; defaults to 32).
// ---------------------------------------------------------------------------
struct PacketIDAllocator {
  mlir::ModuleOp module;
  uint8_t limit; // from AIETargetModel or default 32

  // Per-MemTile-domain counters.
  llvm::DenseMap<mlir::Value, uint8_t> nextPerDomain;

  explicit PacketIDAllocator(mlir::ModuleOp mod, uint8_t lim = 32)
      : module(mod), limit(lim) {}

  std::optional<uint8_t> allocate(mlir::Value domain);

  // Allocate a power-of-2-aligned block of `count` consecutive packet IDs.
  //
  // The downstream AIECreatePathFindFlows pass computes mask/value rules for
  // groups of packet flows that share a source port and destination set.
  // If the IDs in a group are not a power-of-2-aligned contiguous block, the
  // computed mask can be overly broad and accidentally match IDs from other
  // groups (e.g., mask=16 value=0 matches ALL IDs 0-15).
  //
  // This method guarantees correctness by:
  //   1. Rounding `count` up to P = nextPow2(count).
  //   2. Aligning the starting ID to the next multiple of P.
  //   3. Reserving the full P-sized block (even if count < P), so unused
  //      IDs in the block cannot be assigned to a different group.
  //
  // Returns the starting ID of the block.  Individual members are at
  // startID, startID+1, ..., startID+count-1.
  std::optional<uint8_t> allocateBlock(mlir::Value domain, unsigned count);

  uint8_t remaining(mlir::Value domain) const {
    auto it = nextPerDomain.find(domain);
    if (it == nextPerDomain.end())
      return limit - 1; // ID 0 is reserved
    return limit - it->second;
  }
};

// ---------------------------------------------------------------------------
// PacketChannelState: module-level state for the packet-DMA fallback used
// when circuit DMA channels are exhausted (mode=any selection).
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
//     convergence hazard check: two packet flows on the same physical channel
//     that route to the same consumer tile D create an ordering hazard under
//     sustained load.
//
// Initialized in ConduitToDMAPass.cpp at pass entry.
// Maintained across all conduits during non-adjacent flow emission.
// ---------------------------------------------------------------------------
struct PacketChannelState {
  // Per (tile_op_ptr, channel_index): is this MM2S channel packet-mode?
  llvm::DenseMap<std::pair<mlir::Operation *, int>, bool> isPacketChannel;

  // Per packet-mode MM2S port: (flow_id, dst_tile_op*) pairs routing through.
  // Key: (tile_op_ptr, channel_index) — same composite key type as
  // isPacketChannel.
  llvm::DenseMap<std::pair<mlir::Operation *, int>,
                 llvm::SmallVector<std::pair<uint8_t, mlir::Operation *>>>
      portOccupancy;
};

// parseTileCoord is defined in ConduitTileInference.h (included above).

// ---------------------------------------------------------------------------
// Per-conduit info gathered from conduit.create typed attributes.
//
// Populated incrementally across pass phases:
//   collectPhase:   producerTileCoord, consumerTileCoords,
//                   shimConsumerTileCoords, depth, capacity, elemType,
//                   accessPattern, routingMode, fuseGroup, effectiveDepth
//   allocPhase:     buffers, prodLock, consLock, aie1Locks,
//                   consumerTileLocks, consumerTileBuffers,
//                   consumerTileAIE1Locks, rotationBuf,
//                   consumerTileRotationBufs, sharedMemory
// ---------------------------------------------------------------------------
struct ConduitInfo {
  // --- Populated by collectPhase. ---

  // Original (unqualified) channel name from conduit.create sym_name.
  // When multiple aie.device ops have identically-named channels, the
  // conduitMap key is device-qualified ("name#devIdx") to prevent
  // overwrites.  origName retains the IR-level name for matching against
  // attribute references and local auxiliary maps.
  std::string origName;

  // Device index within state.deviceOps (-1 = unknown / single-device).
  int deviceIndex = -1;

  // Tile coordinates parsed from the typed Create attributes.
  std::pair<int64_t, int64_t> producerTileCoord = {-1, -1};
  llvm::SmallVector<std::pair<int64_t, int64_t>> consumerTileCoords;
  // Shim consumer tiles (row==0): DMA endpoints, no local memory.
  llvm::SmallVector<std::pair<int64_t, int64_t>> shimConsumerTileCoords;
  int64_t depth = 1;
  // Element count per DMA transfer, from put/get_memref_async {num_elems=N}.
  // Populated by collectPhase; used by linkPhase BD-chain emission for Tier 3
  // channels to determine BD length from the async transfer descriptor.
  int64_t numElems = 0;
  mlir::Type elemType; // actual element memref type (may be null)
  // Cyclostatic (CSDF) access pattern from conduit.create access_pattern attr.
  // Empty = uniform SDF; non-empty = CSDF per-iteration acquire counts.
  llvm::SmallVector<int64_t> accessPattern;
  // Routing mode enum. std::nullopt = "any" (absent / unresolved — resolved
  // inline by --conduit-to-dma's routePhase). Present values: Circuit, Packet,
  // Cascade, Stream, SharedMemory, DMA.
  std::optional<RoutingMode> routingMode;
  // Core stream port index for routing_mode="stream" (-1 if not stream).
  int32_t aieStreamPort = -1;
  // DMA channel fusion group label (from --conduit-fuse-channels annotation).
  std::string fuseGroup;
  // S2MM channel fusion group label (from --conduit-fuse-channels annotation).
  std::string fuseGroupS2MM;

  // Number of put_memref_async ops referencing this channel, inferred by
  // collectPhase.  When putCount > 1 and dmaRepeat == 0, --conduit-to-dma
  // uses putCount as the BD chain length and emits a linear (one-shot) chain.
  // The --conduit-fuse-channels pass rewrites all non-canonical put/get ops
  // to the canonical channel name, so putCount naturally equals N for an
  // N-way temporal-multiplex group.  No explicit annotation is needed.
  int64_t putCount = 0;

  // --- Populated by computeEffectiveDepth (within collectPhase). ---

  // Producer-side effective depth: min(depth, maxProdAcquire+1).
  // 0 means "use raw depth" (no optimization).
  int64_t effectiveDepth = 0;

  // Partial-release buffer count adjustment.
  // Populated by collectPhase (consumer-side acquire scan).
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

  // --- Populated by allocPhase (allocateBuffersAndLocks). ---

  // Shared memory flag: set when producer and consumer are adjacent tiles.
  // When true, buffers/locks go on the producer (or alloc) tile; no DMA.
  bool sharedMemory = false;

  // Shim-tile locks for shim producer conduits (allocated during routePhase
  // shim-endpoint handling, consumed by linkPhase BD-chain emission).
  // Populated by routePhase when the producer tile is a shim.
  AIE::LockOp shimProdLock;
  AIE::LockOp shimConsLock;

  // Hardware SSA values:
  llvm::SmallVector<AIE::BufferOp> buffers; // depth-many on consumer_tile[0]

  // Per-consumer-tile lock pairs for multi-consumer (broadcast) correctness.
  // Key: tile SSA Value.  Read by linkPhase (BD chains) and lowerPhase
  // (use_lock). NOTE: for sharedMemory conduits, LockOps are physically on the
  // producer tile but keyed on the consumer tile for lowerPhase use_lock
  // lookup.
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

  // For depth>1: rotation counter — shared per-tile aie.buffer + slot index.
  // Multiple conduits on the same tile share one memref<N xi32> aie.buffer
  // (device-level, matching stateful transform's _anonymous pattern); each
  // conduit is assigned a unique slot index within it.
  mlir::Value rotationBuf;     // shared tile buffer (consumer direction)
  int64_t rotationBufSlot = 0; // slot index within that buffer
  llvm::DenseMap<mlir::Value, mlir::Value>
      consumerTileRotationBufs; // tile → shared rotation buffer
  llvm::DenseMap<mlir::Value, int64_t>
      consumerTileRotationBufSlots; // tile → slot index for this conduit

  // For depth>1 produce-mode: rotation counter on the producer tile.
  mlir::Value producerRotationBuf; // shared tile buffer (producer direction)
  int64_t producerRotationBufSlot = 0; // slot index within that buffer
  llvm::DenseMap<mlir::Value, mlir::Value>
      producerTileRotationBufs; // tile → shared rotation buffer
  llvm::DenseMap<mlir::Value, int64_t>
      producerTileRotationBufSlots; // tile → slot index for this conduit

  // --- New feature flags (populated by collectPhase from conduit.create attrs)
  // ---

  // noLocks: suppress all lock allocation and use_lock ops.
  // Set when sync_mode == None on conduit.create.
  bool noLocks = false;
  // forceDMA: force DMA routing even for adjacent tiles (skip shared-mem path).
  // Set when routing_mode == Circuit explicitly.
  bool forceDMA = false;
  // plio: when true, the shim endpoint uses Platform I/O instead of DMA.
  // Flows use WireBundle::PLIO and shim_dma_allocation carries {plio = true}.
  bool plio = false;
  // dma_repeat: number of times the DMA engine runs the whole BD chain
  // (> 0 → DMAStartOp repeat_count = K-1, non-circular BD chain with terminal
  // aie.end).
  int64_t dmaRepeat = 0;
  // bdRepeat: from objectfifo bd_repeat (formerly repeat_count).  BD-level
  // unroll factor: number of times each BD fires before advancing.  0/1 = once.
  int64_t bdRepeat = 0;
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
    // Annotation-free inference: putCount > 1 with no dma_repeat means N
    // sequential puts were merged (by --conduit-fuse-channels or by
    // --air-channel-to-conduit's per-channel merge step). dma_repeat wins if
    // set (ObjectFIFO task-queue loops have putCount=1).
    if (putCount > 1 && dmaRepeat == 0)
      return putCount;
    int64_t d = depth > 0 ? depth : 1;
    if (maxConsumerAcquire <= 0)
      return d;
    return std::max(d, maxConsumerAcquire + 1);
  }

  // Compute the producer-side buffer count for this conduit.
  // Mirrors nConsumerBuffers() for the Produce port: when a producer acquires
  // K output slots but releases fewer than K per step, extra slots must be
  // allocated so the DMA engine can drain one slot while the core holds the
  // rest. Formula: max(depth, maxProduceAcquire + 1) maxProduceAcquire = 0 for
  // normal (non-sliding-window) producers.
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
    mlir::Value rotationBuf;             // shared tile buffer (consumer dir)
    int64_t rotationBufSlot = 0;         // slot index within that buffer
    mlir::Value producerRotationBuf;     // shared tile buffer (producer dir)
    int64_t producerRotationBufSlot = 0; // slot index within that buffer
    mlir::Operation *coreOp = nullptr;
  };

  // Resolve per-tile locks, buffers, and rotation counters for an op
  // inside a CoreOp.  Walks the parent chain to find the enclosing CoreOp,
  // then looks up per-tile overrides in consumerTileLocks/Buffers/RotationBufs.
  ResolvedTileResources resolveForTile(mlir::Operation *op);
};

// ---------------------------------------------------------------------------
// Metadata for an acquire_async op, recorded before erasure so that
// wait_window and wait_all can look up lock info after the op is gone.
// Populated when async acquires are recorded; read by the wait_window /
// wait_all lowering steps inside lowerPhase.
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
  AIE::DeviceOp deviceOp; // first device (primary; for legacy code)
  llvm::SmallVector<AIE::DeviceOp> deviceOps; // ALL devices in module order
  mlir::OpBuilder *builder;
  mlir::MLIRContext *ctx;

  // Target model queries.
  const AIE::AIETargetModel *targetModel = nullptr;
  AIE::AIEArch aieArch = AIE::AIEArch::AIE1;
  AIE::LockAction acqAction;

  // Convenience: true for AIE2 and AIE2p (all non-AIE1 architectures).
  bool isAIE2Plus() const { return aieArch != AIE::AIEArch::AIE1; }

  // Return the MemTile domain key for packet ID scoping.
  // If the tile is itself a MemTile, returns it directly.
  // Otherwise, looks up the MemTile in the same column (row=1).
  // Falls back to the tile itself if no MemTile is found (e.g., single-row
  // designs or architectures without MemTiles).
  mlir::Value getMemTileDomain(mlir::Value tileVal);

  // Multi-device support: true when the module contains >1 aie.device.
  bool isMultiDevice() const { return deviceOps.size() > 1; }

  // Return the index of a DeviceOp in deviceOps (0-based).
  int getDeviceIndex(AIE::DeviceOp dev) const {
    for (size_t i = 0; i < deviceOps.size(); ++i)
      if (deviceOps[i] == dev)
        return static_cast<int>(i);
    return 0;
  }

  // Build a device-qualified conduitMap key from a channel name and context op.
  // Single-device: returns the name as-is.
  // Multi-device: returns "name__dN" to prevent cross-device overwrites.
  // Uses "__d" separator which is valid in MLIR symbol names (unlike "#").
  std::string makeConduitKey(llvm::StringRef name,
                             mlir::Operation *contextOp) const;

  // Build a device-qualified fuse group key.
  // Single-device: returns the group label as-is.
  // Multi-device: returns "group__dN" to prevent cross-device fusion.
  std::string qualifyFuseGroup(llvm::StringRef group, int deviceIndex) const {
    if (group.empty() || !isMultiDevice() || deviceIndex < 0)
      return group.str();
    return group.str() + "__d" + std::to_string(deviceIndex);
  }

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
  // NOTE: In multi-device modules where devices share tile coordinates,
  // the global tileCache may be stale (last device wins).  Use the
  // per-device cache via lookupTileByCoord() with activeDevIdx set.
  llvm::DenseMap<std::pair<int64_t, int64_t>, AIE::TileOp> tileCache;

  // Per-device tile cache: one map per device in deviceOps.
  // Only populated when deviceOps.size() > 1.  Avoids tile-coordinate
  // collisions that occur in the global tileCache when multiple devices
  // share the same (col, row) coordinates.
  std::vector<llvm::DenseMap<std::pair<int64_t, int64_t>, AIE::TileOp>>
      perDevTileCache;

  // Active device index: used by lookupTileByCoord to select the correct
  // per-device tile cache.  Set by switchToDeviceIndex().  -1 = use global.
  int activeDevIdx = -1;

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

  // Per-tile shared rotation counter buffer pool.
  // Populated by allocPhase() pre-scan; each tile that needs N rotation
  // counters gets one memref<N xi32> aie.buffer (at device level) shared
  // across all conduits on that tile.
  llvm::DenseMap<mlir::Value, mlir::Value> tileRotationBuf;
  llvm::DenseMap<mlir::Value, int64_t> tileRotationBufNextSlot;

  // Shim conduit names for symbol rewriting in routePhase.
  llvm::StringSet<> shimConduitNames;

  // Packet flow ID allocator (replaces raw counter; initialized in Pass shell).
  // Use std::optional so the state struct can be default-constructed before
  // the module and architecture limit are known.
  std::optional<PacketIDAllocator> packetIDAllocator;

  // Packet channel state for the mode=any (circuit-exhausted) fallback.
  // Tracks which MM2S channels have been designated for packet use, and which
  // (flow_id, dst_tile) pairs are routed through each packet-mode channel.
  PacketChannelState pktChannelState;

  // Per-tile BD budget used (number of BD slots consumed so far).
  // Incremented by `depth` whenever a conduit allocates BD chains on a tile.
  // Used during the mode=any fallback to check whether the BD budget allows
  // a new flow.
  llvm::DenseMap<mlir::Value, int32_t> tileBDUsed;

  // Fuse group tracking for non-adjacent flow emission and BD-chain emission.
  llvm::StringMap<int32_t> fuseGroupMM2SChannel;
  llvm::StringMap<int32_t> fuseGroupS2MMChannel;
  llvm::StringMap<llvm::SmallVector<std::string, 4>> fuseGroupMembers;

  // Packet-mode S2MM channel reuse per consumer tile.
  // When multiple packet-mode channels target the same tile, they share
  // one physical S2MM port (differentiated by packet_id in BD headers).
  // Key: consumer tile SSA value.  Value: assigned S2MM channel index.
  // Populated during routePhase packet-mode flow emission.
  llvm::DenseMap<mlir::Value, int32_t> pktTileS2MMChannel;

  // Packet-mode S2MM lock sharing: when multiple packet-mode conduits share
  // an S2MM port on a consumer tile (via pktTileS2MMChannel), they also share
  // a single lock pair.  This prevents lock ID overflow on tiles with many
  // packet-muxed channels (e.g., flash attention Q+K+V → 1 lock pair instead
  // of 3).  Populated alongside pktTileS2MMChannel; consumed by linkPhase BD
  // chain generation (via info.consumerTileLocks overwrite).
  llvm::DenseMap<mlir::Value, std::pair<mlir::Value, mlir::Value>>
      pktTileS2MMLock;

  // Pre-computed used DMA channels per tile (populated before linkPhase BD
  // chain generation).
  llvm::DenseMap<mlir::Value, llvm::DenseSet<int32_t>> preUsedMM2SChannels;
  llvm::DenseMap<mlir::Value, llvm::DenseSet<int32_t>> preUsedS2MMChannels;

  // BD range tracking for fused channel groups (linkPhase post-pass).
  llvm::StringMap<std::pair<mlir::Block *, mlir::Block *>> conduitBDRange;

  // Per-conduit packet flow ID for MM2S BD packet headers (aie.dma_bd_packet).
  // For packet-mode channels, each MM2S BD needs a packet header matching the
  // packet flow ID so the switchbox can route data to the correct destination.
  // Key: conduit name; Value: packet ID (0-31).
  // Populated by routePhase. Read by linkPhase for BD chain emission.
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
    return lookupTileByCoord(col, row);
  }

  AIE::TileOp lookupTileByCoord(int64_t col, int64_t row);

  // Return the Location from the DeviceOp that owns the tile produced by
  // tileVal, falling back to deviceOp.getLoc() if the tile is not found.
  // Use this instead of deviceOp.getLoc() when emitting ops that belong to
  // a tile in a secondary device of a multi-device module.
  mlir::Location getLocForTile(mlir::Value tileVal);

  // Return the DeviceOp that owns the tile at (col, row), or null if not found.
  // Used by multi-device Pass C to select the correct DeviceOp body for
  // op insertion when emitting locks, buffers, and flows.
  AIE::DeviceOp getDeviceForTile(int64_t col, int64_t row) const;

  // Switch the active device context to the device at the given index
  // in deviceOps.  Updates activeDevIdx, deviceBody, and insertAfterTile.
  // Call this at the start of each conduitMap loop iteration to ensure
  // lookupTileByCoord returns tiles from the correct device.
  void switchToDeviceIndex(int devIdx);

  // Update state.deviceBody and state.insertAfterTile to point to the correct
  // DeviceOp for the tile at (col, row).  Call this before allocating
  // aie.buffer / aie.lock ops for a tile in a multi-device module.
  // No-op if the tile is not found or already in the active device.
  void switchDeviceForTile(int64_t col, int64_t row);

  // Emit a circuit or packet flow between two tiles.
  // For packet flows, the packet ID is allocated from packetIDAllocator.
  // If the ID budget is exhausted, passFailed is set and the flow is not
  // emitted (the error is reported by the allocator on the module op).
  void emitFlow(std::optional<RoutingMode> routingMode, mlir::Value srcTile,
                AIE::WireBundle srcBundle, int32_t srcChan, mlir::Value dstTile,
                AIE::WireBundle dstBundle, int32_t dstChan);

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
                                  int64_t depth, int64_t prodInit = -1);

  /// Look up a conduit by name in the conduitMap.
  /// Returns nullptr if not found.
  ConduitInfo *lookupConduit(mlir::StringRef name) {
    auto it = conduitMap.find(name.str());
    if (it == conduitMap.end())
      return nullptr;
    return &it->second;
  }

  /// Device-aware conduit lookup.  Builds a device-qualified key from
  /// the channel name and the context op's enclosing aie.device, then
  /// falls back to the unqualified name for single-device compatibility.
  ConduitInfo *lookupConduit(mlir::StringRef name, mlir::Operation *contextOp);

  /// Device-aware lookup for the per-conduit S2MM consumer channel.
  ///
  /// Mirrors lookupConduit's qualified-then-unqualified fallback for the
  /// `conduitConsS2MMChannel` map.  Phase 4a/4b in routePhase iterate
  /// `conduitMap` and write each channel under the device-qualified key
  /// (`name__d<idx>` from `makeConduitKey`); linkPhase reads with the raw
  /// (unqualified) IR symbol pulled from a Scatter/Gather attribute. In
  /// multi-device modules the unqualified read MISSES the qualified write,
  /// silently falling through to a stale `tileNextS2MMChannel++` counter
  /// and emitting a channel that does not match the Phase 4 flow.
  ///
  /// Returns the assigned channel index, or -1 if neither the qualified
  /// nor the unqualified key is present.  Callers that previously used
  /// `state.conduitConsS2MMChannel.find(...)` and on miss fell through to
  /// `tileNextS2MMChannel[tile]++` should preserve that fallback when the
  /// helper returns -1.
  ///
  /// PRECONDITION: contextOp must be inside the same aie.device as the
  /// conduit (true by MLIR symbol scoping for any op resolving `name` as a
  /// SymbolRefAttr in the same scope).  A caller from a different device
  /// silently mis-keys and re-introduces the multi-device bug class.
  int32_t lookupS2MMChannel(mlir::StringRef name, unsigned consIdx,
                            mlir::Operation *contextOp);

  /// Device-aware lookup for the per-conduit MM2S producer channel.
  /// See `lookupS2MMChannel` for the rationale; same qualified-then-
  /// unqualified fallback for the `conduitMM2SChannel` map.
  /// Returns -1 on miss.
  /// PRECONDITION: same as lookupS2MMChannel.
  int32_t lookupMM2SChannel(mlir::StringRef name, mlir::Operation *contextOp);

  /// Device-aware lookup for the per-conduit packet ID.
  /// See `lookupS2MMChannel` for the rationale; same qualified-then-
  /// unqualified fallback for the `conduitPacketID` map.
  /// Returns -1 on miss.
  /// PRECONDITION: same as lookupS2MMChannel.
  int32_t lookupPacketID(mlir::StringRef name, mlir::Operation *contextOp);

  /// Device-aware insert for the per-conduit S2MM consumer channel.
  /// Stores `ch` under the device-qualified key produced by
  /// `makeConduitKey(name, contextOp)` so that subsequent reads via
  /// `lookupS2MMChannel` find the entry across multi-device modules.
  /// Mirrors `lookupS2MMChannel`'s qualified-then-unqualified fallback by
  /// always writing the qualified key.
  /// PRECONDITION: same as lookupS2MMChannel.
  void insertS2MMChannel(mlir::StringRef name, unsigned consIdx, int32_t ch,
                         mlir::Operation *contextOp);

  /// Device-aware insert for the per-conduit MM2S producer channel.
  /// See `insertS2MMChannel`.
  /// PRECONDITION: same as lookupS2MMChannel.
  void insertMM2SChannel(mlir::StringRef name, int32_t ch,
                         mlir::Operation *contextOp);

  /// Device-aware insert for the per-conduit packet ID.
  /// See `insertS2MMChannel`.
  /// PRECONDITION: same as lookupS2MMChannel.
  void insertPacketID(mlir::StringRef name, uint8_t pktID,
                      mlir::Operation *contextOp);

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
                   AIE::BDDimLayoutArrayAttr dims = {}, int pktID = -1);

  // Allocate `count` buffers of type `bufTy` on the given tile.
  llvm::SmallVector<AIE::BufferOp> allocateBuffers(mlir::Value tileVal,
                                                   llvm::StringRef prefix,
                                                   mlir::Type bufTy,
                                                   int64_t count);
};

// ---------------------------------------------------------------------------
// Conduit-specific shared-memory feasibility predicate.
//
// Wraps `targetModel.isLegalMemAffinity()` with the two Conduit-lowering
// adjustments that all 7 query sites in Pass C / ConduitInferModes share:
//
//   1. Cross-column same-row "W-neighbor" adjacency is reported legal by
//      AIE2TargetModel::isLegalMemAffinity but has never been NPU-validated
//      through the Conduit shared-memory lowering path.  When NOT explicitly
//      requested via `routing_mode = "shared_memory"`, treat it as infeasible
//      so the DMA path is taken (which is exercised + known good).  Detected
//      via the `isMemWest` helper (in either direction) so future targets
//      (e.g. NPU3) can override neighbor relationships correctly rather than
//      a hand-written `prodRow == consRow && prodCol != consCol` guard.
//   2. Explicit `routing_mode = "shared_memory"` overrides the cross-column
//      exclusion (a future feasibility-error pass — #107 / #124 — will reject
//      geometrically infeasible explicit requests at validation time).
//
// Returns true iff the producer/consumer tile pair is feasible for the
// Conduit shared-memory lowering path, given the conduit's routing mode.
// ---------------------------------------------------------------------------
bool isConduitFeasibleSharedMemory(const AIE::AIETargetModel &targetModel,
                                   int64_t prodCol, int64_t prodRow,
                                   int64_t consCol, int64_t consRow,
                                   std::optional<RoutingMode> routingMode);

// ---------------------------------------------------------------------------
// Defensive HW-cap check for BD chain length, applied at every Pass C
// BD-emit site.  Returns true on cap-OK (caller proceeds), false on
// cap-exceeded (caller bails after setting state.passFailed=true).
// On overflow, emits a structured Pass C diagnostic naming the channel,
// tile, requested length, and per-tile cap (replacing the late
// AIEDialect verifier crash `'aie.mem' op has more than 16 blocks`).
// Returns true (no-op) when targetModel/tile is unavailable, so existing
// tests without a target model are unaffected.
// ---------------------------------------------------------------------------
bool checkBDChainCap(ConduitToDMAState &state, mlir::Value tile,
                     int64_t chainLen, llvm::StringRef channelName);

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
