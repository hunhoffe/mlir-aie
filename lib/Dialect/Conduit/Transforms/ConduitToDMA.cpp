//===- ConduitToDMA.cpp - Conduit IR → aie.dma_bd / aie.lock (Pass C)
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
// Pass C of the Conduit lowering pipeline: lower Conduit IR to raw AIE
// hardware programming ops (aie.dma_bd, aie.lock, aie.buffer, aie.flow).
//
// Architecture:
//
//   aie.objectfifo.*  ──┐
//                       ├──► Conduit IR ──► aie.dma_bd / aie.lock / aie.buffer
//   air.channel.*     ──┘
//
//   (ObjectFifoToConduit.cpp)   (this file)
//
// What this pass does
// -------------------
//
// The pass walks the module in source order and:
//
// 1. Collects all conduit.create ops to build a map from
//    conduit name → FifoInfo (producer tile coords, consumer tile coords,
//    element type string, depth, capacity).
//
// 2. For each conduit.create (non-shim producer):
//      Inserts into the aie.device body:
//        - depth × aie.buffer on the consumer tile
//        - aie.lock (prod_lock, init=depth) on the consumer tile
//        - aie.lock (cons_lock, init=0)     on the consumer tile
//      Tracks these in buffersPerConduit / locksPerConduit maps.
//
// 3. For conduit.create where producer_tile is a shim tile (row == 0):
//      Inserts aie.shim_dma_allocation and aie.flow.
//
// 4. For conduit.objectfifo_link (distribute or join):
//      Generates the MemTile DMA BD chain inside aie.memtile_dma.
//      The BD chain mirrors AIEObjectFifoStatefulTransform logic:
//        distribute: one S2MM channel ingests the full buffer; each MM2S
//                    channel sends the slice for one destination.
//        join:       each S2MM channel ingests the slice for one source;
//                    one MM2S channel outputs the joined buffer.
//
// 5. For conduit.acquire / conduit.release (inside func bodies):
//      Replaces them with aie.use_lock calls referencing the tracked locks.
//
// 6. Erases all processed Conduit ops.
//
// Implementation status
// ---------------------
// Handles depth-1, depth-2, and depth-3 single-consumer fifos; AIE1 and AIE2
// device variants; shared-memory (same-tile) optimization; broadcast to N
// consumer tiles; async acquire/release path; CSDF rate annotation; and
// packet routing (aie.packet_flow).  The memtile DMA BD chain covers the
// distribute and join link patterns.
//
// Known gaps (open items):
//   - aie.buffer / aie.lock insertion requires the aie.device TileOp SSA
//     values; these are looked up by (col,row) from a tile cache populated
//     during the walk.  If a tile is not declared before the conduit.create,
//     the pass emits a warning and skips that conduit.
//   - aie.shim_dma_allocation uses the conduit name + "_shim_alloc" as sym.
//   - conduit.objectfifo_link with join mode does not yet allocate buffers on
//     each source tile; only the memtile relay BD chain is emitted.
//   - conduit.wait_all_async is not yet lowered (the blocking conduit.wait is
//     lowered correctly).
//   - conduit.acquire_async on the produce port is not supported; only the
//     consume port async acquire is lowered.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

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
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITTODMA
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helper: parse "tile(col,row)" → (col, row).  Returns {-1,-1} on failure.
// ---------------------------------------------------------------------------

static std::pair<int64_t, int64_t> parseTileCoord(llvm::StringRef s) {
  // Format: "tile(col,row)"
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
// Per-conduit info gathered from conduit.create typed attributes
// ---------------------------------------------------------------------------

struct ConduitInfo {
  // Tile coordinates parsed from the typed Create attributes.
  std::pair<int64_t, int64_t> producerTileCoord = {-1, -1};
  llvm::SmallVector<std::pair<int64_t, int64_t>> consumerTileCoords;
  // Shim consumer tiles (row==0): DMA endpoints, no local memory.
  // These get aie.shim_dma_allocation + aie.flow, not aie.buffer + aie.lock.
  llvm::SmallVector<std::pair<int64_t, int64_t>> shimConsumerTileCoords;
  int64_t depth = 1;
  int64_t capacity = 0;
  mlir::Type elemType; // actual element memref type (may be null)
  // Cyclostatic (CSDF) access pattern from conduit.create access_pattern attr.
  // Empty means uniform SDF (all acquires use the same count = capacity/depth).
  // Non-empty means CSDF: access_pattern[i % period] is the acquire count for
  // iteration i, where period = access_pattern.size().
  // Pass C uses this in Phase 6 (Acquire lowering) to emit the correct
  // per-iteration lock acquire count, and in Phase 5.5 (aie.mem BD chain) to
  // emit a BD block per pattern element rather than a uniform ring.
  llvm::SmallVector<int64_t> accessPattern;
  // Shared memory flag: set by Phase 3 when producer and consumer tiles are
  // adjacent and can share memory without DMA.
  //
  // When sharedMemory is true:
  //   - Buffers and locks are allocated on the PRODUCER tile (not consumer).
  //   - No aie.flow, no aie.dma_bd, no aie.mem, no aie.shim_dma_allocation.
  //   - Phase 5.5 (BD chain generation) is skipped.
  //   - Phase 6 (acquire/release lowering) still emits aie.use_lock; locks are
  //     on the producer tile but accessible from both producer and consumer cores
  //     via the shared memory interface.
  //   - The consumerTileLocks map is keyed on the CONSUMER tile SSA value (so
  //     Phase 6 can look them up from the consumer core), but the LockOp itself
  //     is on the producer tile.
  bool sharedMemory = false;
  // Hardware SSA values populated during lowering:
  llvm::SmallVector<AIE::BufferOp> buffers; // depth-many on consumer_tile[0]
  // Per-consumer-tile lock pairs for multi-consumer (broadcast) correctness.
  // Keyed on the tile SSA Value (result of aie.tile op).
  // Phase 3 populates this for every consumer tile; Phase 6 looks up the pair
  // for the tile that contains the acquire/release being lowered.
  // NOTE: for sharedMemory conduits the LockOps are physically on the PRODUCER
  // tile, but the key is still the consumer tile value for Phase 6 lookup.
  llvm::DenseMap<mlir::Value, std::pair<AIE::LockOp, AIE::LockOp>>
      consumerTileLocks; // tile → (prodLock, consLock)
  // Per-consumer-tile buffer vectors for SubviewAccess resolution.
  // Each entry holds depth-many BufferOps for that tile.
  // NOTE: for sharedMemory conduits the BufferOps are physically on the PRODUCER
  // tile, but the key is still the consumer tile value for Phase 6 lookup.
  llvm::DenseMap<mlir::Value, llvm::SmallVector<AIE::BufferOp>>
      consumerTileBuffers; // tile → [buff_0, ..., buff_{depth-1}]
  // Convenience accessors for the single-consumer case and for the link phase
  // (which always uses consumer_tile[0]).  Populated from consumerTileLocks[0].
  AIE::LockOp prodLock; // prod lock (init=depth)
  AIE::LockOp consLock; // cons lock (init=0)
  // AIE1 per-slot locks: one lock per buffer slot (depth-many).
  // Only populated when !isAIE2.  BD block i uses aie1Locks[i % depth] for
  // both acquire and release (different value for each direction).
  // For AIE2, this is empty — use prodLock/consLock instead.
  llvm::SmallVector<AIE::LockOp> aie1Locks;
  // Per-consumer-tile AIE1 lock vectors (for multi-consumer broadcast).
  // Same structure as consumerTileBuffers but for AIE1 per-slot locks.
  llvm::DenseMap<mlir::Value, llvm::SmallVector<AIE::LockOp>>
      consumerTileAIE1Locks; // tile → [lock_0, ..., lock_{depth-1}]
  // For depth>1: a memref<1xi32> rotation counter on the consumer tile.
  // The core body loads this counter before each acquire to select the correct
  // ping-pong buffer (buff_{counter % depth}), then increments it after the
  // release.  Mirrors the stateful transform's buffer_0_N pattern.
  // rotationBuf is the counter for consumer_tile[0] (single-consumer fast path).
  // consumerTileRotationBufs maps each consumer tile to its own counter buffer
  // for multi-consumer (broadcast) correctness — each consumer tile needs its
  // own local counter since the counter lives in tile-local memory.
  AIE::BufferOp rotationBuf;
  llvm::DenseMap<mlir::Value, AIE::BufferOp>
      consumerTileRotationBufs; // tile → rotation counter buffer
  // Alloc tile override: when objectfifo.allocate specifies a delegate tile,
  // Pass A stores the [col, row] in conduit.create's alloc_tile attribute.
  // Pass C uses this to place buffers/locks on the delegate tile instead of
  // the default producer tile in the shared memory path.
  bool hasAllocTile = false;
  std::pair<int64_t, int64_t> allocTileCoord = {-1, -1};
  // Routing mode: "circuit" (default) or "packet".
  // When "packet", Phase 4 emits aie.packet_flow instead of aie.flow.
  std::string routingMode = "circuit";
  // Legacy string form for the ObjectFifoLink memtile lookup.
  // Populated from producer_tile for shim detection.
  std::string producerTileStr; // "tile(col,row)"
  llvm::SmallVector<std::string> consumerTileStrs; // for link memtile lookup
};

// ---------------------------------------------------------------------------
// Main pass
// ---------------------------------------------------------------------------

struct ConduitToDMAPass : impl::ConduitToDMABase<ConduitToDMAPass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    mlir::MLIRContext *ctx = module.getContext();

    // -----------------------------------------------------------------------
    // Phase 1: collect ConduitInfo from conduit.create typed attributes.
    //
    // The new conduit.create carries all metadata as typed attributes:
    //   producer_tile   : DenseI64Array [col, row]
    //   consumer_tiles  : DenseI64Array [col0, row0, col1, row1, ...]
    //   element_type    : TypeAttr
    //   depth           : I64Attr
    // -----------------------------------------------------------------------

    // MapVector preserves insertion order (= module walk order = source order),
    // giving deterministic range-for iteration in Phases 3, 4, and 5.5.
    // StringMap iteration is hash-order and non-deterministic.
    // lookup-only find() calls work identically on both containers.
    llvm::MapVector<llvm::StringRef, ConduitInfo> conduitMap;

    module.walk([&](Create op) {
      ConduitInfo info;
      info.capacity = op.getCapacity();

      // Extract depth. getDepth() returns std::optional<uint64_t>.
      if (auto depthOpt = op.getDepth())
        info.depth = static_cast<int64_t>(*depthOpt);

      // Extract element type. getElementType() returns std::optional<mlir::Type>.
      if (auto etOpt = op.getElementType())
        info.elemType = *etOpt;

      // Extract producer tile. getProducerTile() returns
      // std::optional<ArrayRef<int64_t>> with [col, row].
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

      // Extract consumer tiles. getConsumerTiles() returns
      // std::optional<ArrayRef<int64_t>> with [col0,row0,col1,row1,...].
      // These are non-shim (row > 0) compute tiles only (Pass A filters shim).
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

      // Extract shim consumer tiles. getShimConsumerTiles() returns
      // std::optional<ArrayRef<int64_t>> with [col0,row0,...] for row==0 tiles.
      // Pass A stores shim tiles here instead of consumer_tiles.
      if (auto sct = op.getShimConsumerTiles()) {
        for (size_t i = 0; i + 1 < sct->size(); i += 2) {
          int64_t col = (*sct)[i], row = (*sct)[i + 1];
          info.shimConsumerTileCoords.push_back(std::make_pair(col, row));
        }
      }

      // Extract cyclostatic access pattern (CSDF).
      // getAccessPattern() returns std::optional<ArrayRef<int64_t>>.
      // When present, the consumer acquires access_pattern[i % period] elements
      // on iteration i instead of a uniform count.
      if (auto ap = op.getAccessPattern()) {
        for (int64_t v : *ap)
          info.accessPattern.push_back(v);
      }

      // Extract routing_mode ("circuit" or "packet").
      // getRoutingMode() returns std::optional<StringRef>.
      if (auto rm = op.getRoutingMode())
        info.routingMode = rm->str();

      // Extract alloc_tile delegate coordinates.
      // getAllocTile() returns std::optional<ArrayRef<int64_t>> with [col, row].
      // Set by Pass A Phase 4.5b from objectfifo.allocate's delegate tile.
      if (auto at = op.getAllocTile()) {
        if (at->size() >= 2) {
          info.hasAllocTile = true;
          info.allocTileCoord = std::make_pair((*at)[0], (*at)[1]);
        }
      }

      conduitMap[op.getName()] = std::move(info);
    });

    // -----------------------------------------------------------------------
    // Phase 2: find the aie.device op and build a tile SSA value cache
    // (col,row) → TileOp.  We need this to create buffers/locks on the
    // correct tiles.
    // -----------------------------------------------------------------------

    // Find the first aie.device op (or skip if not present).
    AIE::DeviceOp deviceOp;
    module.walk([&](AIE::DeviceOp op) {
      if (!deviceOp)
        deviceOp = op;
    });

    if (!deviceOp) {
      // No device op — nothing to lower; emit a warning and return.
      module.emitWarning(
          "conduit-to-dma: no aie.device found; skipping lowering");
      return;
    }

    // Obtain the target model for adjacency queries (isLegalMemAffinity).
    const AIE::AIETargetModel &targetModel = AIE::getTargetModel(deviceOp);

    // AIE1 (xcvc1902) uses value-based Acquire semantics; AIE2 uses semaphore-
    // based AcquireGreaterEqual.  Mirror the check in AIEObjectFifoStatefulTransform.
    const bool isAIE2 =
        (targetModel.getTargetArch() != AIE::AIEArch::AIE1);
    const AIE::LockAction acqAction = isAIE2
                                          ? AIE::LockAction::AcquireGreaterEqual
                                          : AIE::LockAction::Acquire;

    // AIE1 BD block lock value constants.
    //
    // AIE1 uses a SINGLE lock per buffer slot with value-based semantics:
    //   0 = buffer slot empty (producer can fill / DMA can receive into it)
    //   1 = buffer slot full  (consumer can drain / DMA can send from it)
    //
    // For AIE2, acquire and release always use count=1 (semaphore delta).
    //
    // S2MM (receiving into buffer):
    //   acquire(0)  → wait until slot is empty
    //   release(1)  → signal slot is now full
    // MM2S (sending from buffer):
    //   acquire(1)  → wait until slot is full
    //   release(0)  → signal slot is now empty
    //
    // These are only consulted when !isAIE2.
    const int32_t aie1S2MMAcqVal = 0; // acquire empty slot before receive
    const int32_t aie1S2MMRelVal = 1; // release as full after receive
    const int32_t aie1MM2SAcqVal = 1; // acquire full slot before send
    const int32_t aie1MM2SRelVal = 0; // release as empty after send

    // Build tile cache.
    llvm::DenseMap<std::pair<int64_t, int64_t>, AIE::TileOp> tileCache;
    deviceOp.walk([&](AIE::TileOp tile) {
      tileCache[{tile.getCol(), tile.getRow()}] = tile;
    });

    // Helper: look up a TileOp from a "tile(col,row)" string.
    auto lookupTile = [&](llvm::StringRef coord) -> AIE::TileOp {
      auto [col, row] = parseTileCoord(coord);
      if (col < 0)
        return {};
      auto it = tileCache.find({col, row});
      if (it == tileCache.end())
        return {};
      return it->second;
    };

    // -----------------------------------------------------------------------
    // Pre-Phase 3: collect link source names before Phase 3 runs.
    //
    // Phase 5 (conduit.objectfifo_link lowering) allocates its OWN per-slice
    // lock pairs on the MemTile for link source conduits.  If Phase 3 also
    // allocates prodLock/consLock for those conduits (as it would for any
    // conduit with a consumer tile), the result is 2 orphan locks that waste
    // lock IDs and shift all subsequent lock indices.
    //
    // Fix: collect the set of conduit names that are link sources BEFORE
    // Phase 3 runs, then skip lock (and buffer) allocation for those conduits
    // inside Phase 3.  Phase 5 handles their lock allocation.
    //
    // Note: buffers on the MemTile consumer tile are still needed by Phase 5
    // (srcInfo.buffers is used for dma_bd references in the BD chain).  Only
    // the LOCKS are skipped; the buffers are still allocated by Phase 3.
    // -----------------------------------------------------------------------

    // Collect link source names for DISTRIBUTE links only.
    //
    // For a distribute link (1 src → N dsts), Phase 5 allocates its own
    // per-slice lock pairs on the MemTile.  Phase 3 would also allocate a
    // prodLock/consLock pair for the same conduit (as a consumer of the
    // MemTile), resulting in 2 orphan locks with wasted IDs.
    //
    // For a join link (N srcs → 1 dst), Phase 5 uses the EXISTING per-source
    // lock pairs that Phase 3 allocates.  Skipping Phase 3 locks for join
    // sources would leave Phase 5 with null lock references.
    //
    // Therefore, only skip Phase 3 lock allocation for distribute sources.
    llvm::StringSet<> linkSrcNamesEarly;
    // Join source names: conduits that appear as srcs in a join link.
    // Tracked separately so Phase 3 can allocate their buffers/locks on the
    // producer compute tile (not the memtile consumer) for correct aie.mem
    // MM2S generation in Phase 5.5.
    llvm::StringSet<> linkJoinSrcNames;
    module.walk([&](ObjectFifoLink linkOp) {
      if (linkOp.getMode() == "distribute") {
        for (auto s : linkOp.getSrcs())
          linkSrcNamesEarly.insert(mlir::cast<mlir::StringAttr>(s).getValue());
      } else {
        // join: record source names for Phase 3 producer-tile reallocation.
        for (auto s : linkOp.getSrcs())
          linkJoinSrcNames.insert(mlir::cast<mlir::StringAttr>(s).getValue());
      }
    });

    // Pre-scan: collect conduit names that have at least one Consume-port
    // acquire op. Conduits without consumer acquires (e.g., link destinations
    // with no aie.core body) don't need rotation counter buffers.
    llvm::StringSet<> conduitNamesWithConsumerAcquire;
    module.walk([&](Acquire acqOp) {
      if (acqOp.getPort() == "Consume")
        conduitNamesWithConsumerAcquire.insert(acqOp.getName());
    });
    module.walk([&](AcquireAsync acqOp) {
      // AcquireAsync is always consumer-side (no port attribute).
      conduitNamesWithConsumerAcquire.insert(acqOp.getName());
    });

    // -----------------------------------------------------------------------
    // Phase 3: for each conduit.create, allocate aie.buffer + aie.lock pairs
    //          in the aie.device body.
    // -----------------------------------------------------------------------

    // Device body reference for insertions.
    mlir::Block &deviceBody = deviceOp.getBodyRegion().front();

    // Find the last tile op in the device body — we'll insert new hardware ops
    // (buffers, locks) immediately after the last tile declaration.  This
    // ensures the new SSA values dominate any aie.core bodies that follow.
    // If there are no tile ops, insert at the start of the block.
    mlir::Operation *insertAfterTile = nullptr;
    for (mlir::Operation &op : deviceBody) {
      if (mlir::isa<AIE::TileOp>(op))
        insertAfterTile = &op;
    }

    // Per-tile lock ID counter to avoid collisions.
    llvm::DenseMap<mlir::Value, int> lockIdCounter;

    // C2 — LockAnalysis-style pre-population:
    //
    // Walk all existing aie.lock ops already present in the device body and
    // advance lockIdCounter[tile] to max(current, lockID + 1).  This ensures
    // that locks we emit during Phase 3 start AFTER any locks that were
    // declared before this pass ran (e.g., by another pass, a user-written
    // aie.lock, or a prior invocation of --conduit-to-dma on a partial module).
    //
    // Only locks that carry an explicit lockID attribute can collide; locks
    // without an assigned ID are handled later by AIEAssignLockIDs and are
    // safe to ignore here.
    //
    // When no pre-existing locks are present this walk is a no-op and
    // lockIdCounter remains empty — identical to the old behaviour.
    deviceOp.walk([&](AIE::LockOp existingLock) {
      if (!existingLock.getLockID().has_value())
        return; // no explicit ID — AIEAssignLockIDs will handle it
      mlir::Value tileVal = existingLock.getTile();
      int existingId = static_cast<int>(existingLock.getLockID().value());
      int &counter = lockIdCounter[tileVal];
      if (existingId + 1 > counter)
        counter = existingId + 1;
    });

    // Helper: look up a TileOp from (col, row) coordinates.
    auto lookupTileByCoord = [&](int64_t col, int64_t row) -> AIE::TileOp {
      auto it = tileCache.find({col, row});
      if (it == tileCache.end())
        return {};
      return it->second;
    };

    for (auto &[name, info] : conduitMap) {
      if (info.consumerTileCoords.empty() &&
          info.shimConsumerTileCoords.empty()) {
        // Producer-only conduit (shim DMA source) — handled below in Phase 4.
        continue;
      }

      // Phase 3b: conduit has shim consumer(s) but no compute consumer.
      // In this case the buffer and locks live on the PRODUCER tile (compute
      // tile), not on the shim tile.  The DMA runs MM2S from the producer tile
      // to the shim receiver (Phase 4b generates shim_dma_allocation + flow).
      if (info.consumerTileCoords.empty() &&
          !info.shimConsumerTileCoords.empty()) {
        auto [prodCol, prodRow] = info.producerTileCoord;
        if (prodCol < 0 || prodRow == 0)
          continue; // skip if producer is also shim (degenerate)

        AIE::TileOp prodTile = lookupTileByCoord(prodCol, prodRow);
        if (!prodTile)
          continue;

        int64_t depth = info.depth > 0 ? info.depth : 1;
        mlir::Type bufTy = info.elemType;
        if (!bufTy) {
          int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
          bufTy = mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
        }

        if (insertAfterTile)
          builder.setInsertionPointAfter(insertAfterTile);
        else
          builder.setInsertionPointToStart(&deviceBody);

        mlir::Value prodTileVal = prodTile.getResult();

        // Allocate depth-many buffers on the producer tile.
        for (int64_t i = 0; i < depth; ++i) {
          std::string symName = name.str() + "_buff_" + std::to_string(i);
          auto buf = builder.create<AIE::BufferOp>(
              deviceOp.getLoc(), bufTy, prodTileVal,
              mlir::StringAttr::get(ctx, symName),
              /*address=*/mlir::IntegerAttr{},
              /*initial_value=*/mlir::ElementsAttr{},
              /*mem_bank=*/mlir::IntegerAttr{});
          info.buffers.push_back(buf);
        }

        if (isAIE2) {
          // AIE2: two semaphore locks — prod_lock (init=depth) and cons_lock (init=0).
          {
            int lockIdx = lockIdCounter[prodTileVal]++;
            std::string symName = name.str() + "_prod_lock_0";
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                deviceOp.getLoc(), prodTileVal, lockIdx,
                static_cast<int>(depth));
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            info.prodLock = lk;
          }
          {
            int lockIdx = lockIdCounter[prodTileVal]++;
            std::string symName = name.str() + "_cons_lock_0";
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                deviceOp.getLoc(), prodTileVal, lockIdx,
                static_cast<int>(0));
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            info.consLock = lk;
          }
        } else {
          // AIE1: one lock per buffer slot (init=0 = empty), depth-many total.
          // Each BD block uses ONE lock (its own slot's lock), so no
          // multiple-lock-per-BD-block violation.
          for (int64_t i = 0; i < depth; ++i) {
            int lockIdx = lockIdCounter[prodTileVal]++;
            std::string symName = name.str() + "_lock_" + std::to_string(i);
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                deviceOp.getLoc(), prodTileVal, lockIdx,
                static_cast<int>(0));
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            info.aie1Locks.push_back(lk);
          }
          // Set convenience accessors to locks[0] for Phase 6 fallback.
          if (!info.aie1Locks.empty()) {
            info.prodLock = info.aie1Locks[0];
            info.consLock = info.aie1Locks[0];
          }
        }
        continue;
      }

      // -----------------------------------------------------------------------
      // Phase 3c: Shared memory detection.
      //
      // If the producer and the single consumer are adjacent tiles (i.e.,
      // targetModel.isLegalMemAffinity() returns true in either direction),
      // the producer and consumer cores can access a common memory bank without
      // any DMA transfer.  In this case we allocate buffers and locks on the
      // PRODUCER tile (so the producer owns the buffer) and skip all DMA
      // setup — no aie.flow, no aie.dma_bd, no aie.mem.
      //
      // Conditions for shared memory path:
      //   1. Exactly one compute consumer (no multi-consumer broadcast, no
      //      shim consumers — those always require DMA).
      //   2. isLegalMemAffinity(prodCol, prodRow, consCol, consRow) OR
      //      isLegalMemAffinity(consCol, consRow, prodCol, prodRow).
      //   3. Not a link source (link ops always require DMA on the MemTile).
      //
      // The stateful transform uses the same check (requiresDMAs → isSharedMemory)
      // and allocates on the producer tile when share_direction != 1.
      // -----------------------------------------------------------------------
      if (info.consumerTileCoords.size() == 1 &&
          info.shimConsumerTileCoords.empty() &&
          !linkSrcNamesEarly.count(name) &&
          !linkJoinSrcNames.count(name)) {
        // Note: linkJoinSrcNames check above prevents shared-memory detection
        // for join source conduits — join sources always require DMA.
        auto [prodCol, prodRow] = info.producerTileCoord;
        auto [consCol, consRow] = info.consumerTileCoords[0];
        // Neither tile may be a shim tile (row==0) or MemTile (row==1);
        // shared memory only applies between compute tiles.
        bool prodIsShim = (prodRow == 0);
        bool consIsShim = (consRow == 0);
        bool prodIsMemtile = targetModel.isMemTile(prodCol, prodRow);
        bool consIsMemtile = targetModel.isMemTile(consCol, consRow);
        if (!prodIsShim && !consIsShim && !prodIsMemtile && !consIsMemtile) {
          bool rightShared = targetModel.isLegalMemAffinity(
              prodCol, prodRow, consCol, consRow);
          bool leftShared = targetModel.isLegalMemAffinity(
              consCol, consRow, prodCol, prodRow);
          if (rightShared || leftShared) {
            // --- Shared memory path ---
            info.sharedMemory = true;

            // Determine the allocation tile: use alloc_tile if specified,
            // otherwise default to the producer tile.
            int64_t allocCol = prodCol, allocRow = prodRow;
            if (info.hasAllocTile) {
              allocCol = info.allocTileCoord.first;
              allocRow = info.allocTileCoord.second;
            }

            AIE::TileOp allocTile = lookupTileByCoord(allocCol, allocRow);
            AIE::TileOp consTile = lookupTileByCoord(consCol, consRow);
            AIE::TileOp prodTile = lookupTileByCoord(prodCol, prodRow);
            if (!allocTile || !consTile || !prodTile) {
              module.emitWarning(
                  "conduit-to-dma: shared memory conduit '" + name.str() +
                  "' has missing tile op; falling back to DMA path");
              info.sharedMemory = false;
              // Fall through to normal consumer loop below.
            } else {
              int64_t depth = info.depth > 0 ? info.depth : 1;
              mlir::Type bufTy = info.elemType;
              if (!bufTy) {
                int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
                bufTy = mlir::MemRefType::get({bufSize},
                                              mlir::IntegerType::get(ctx, 32));
              }

              if (insertAfterTile)
                builder.setInsertionPointAfter(insertAfterTile);
              else
                builder.setInsertionPointToStart(&deviceBody);

              // Use the allocation tile for buffer/lock placement.
              // When alloc_tile is set (from objectfifo.allocate), buffers and
              // locks go on the delegate tile instead of the producer tile.
              mlir::Value allocTileVal = allocTile.getResult();
              mlir::Value prodTileVal = prodTile.getResult();
              mlir::Value consTileVal = consTile.getResult();

              // Allocate depth-many buffers on the ALLOCATION tile.
              // Both producer and consumer cores can access these via the
              // shared memory interface (no DMA needed).
              llvm::SmallVector<AIE::BufferOp> sharedBuffers;
              for (int64_t i = 0; i < depth; ++i) {
                std::string symName =
                    name.str() + "_buff_" + std::to_string(i);
                auto buf = builder.create<AIE::BufferOp>(
                    deviceOp.getLoc(), bufTy, allocTileVal,
                    mlir::StringAttr::get(ctx, symName),
                    /*address=*/mlir::IntegerAttr{},
                    /*initial_value=*/mlir::ElementsAttr{},
                    /*mem_bank=*/mlir::IntegerAttr{});
                sharedBuffers.push_back(buf);
                info.buffers.push_back(buf);
              }

              // Allocate lock(s) on the ALLOCATION tile.
              AIE::LockOp sharedProdLock;
              AIE::LockOp sharedConsLock;
              if (isAIE2) {
                // AIE2: two semaphore locks.
                {
                  int lockIdx = lockIdCounter[allocTileVal]++;
                  std::string symName = name.str() + "_prod_lock_0";
                  AIE::LockOp lk = builder.create<AIE::LockOp>(
                      deviceOp.getLoc(), allocTileVal, lockIdx,
                      static_cast<int>(depth));
                  lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                  sharedProdLock = lk;
                  info.prodLock = lk;
                }
                {
                  int lockIdx = lockIdCounter[allocTileVal]++;
                  std::string symName = name.str() + "_cons_lock_0";
                  AIE::LockOp lk = builder.create<AIE::LockOp>(
                      deviceOp.getLoc(), allocTileVal, lockIdx,
                      static_cast<int>(0));
                  lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                  sharedConsLock = lk;
                  info.consLock = lk;
                }
              } else {
                // AIE1: one lock per buffer slot (init=0 = empty), depth-many.
                // Each BD block uses ONE lock (its own slot's lock).
                for (int64_t i = 0; i < depth; ++i) {
                  int lockIdx = lockIdCounter[allocTileVal]++;
                  std::string symName = name.str() + "_lock_" + std::to_string(i);
                  AIE::LockOp lk = builder.create<AIE::LockOp>(
                      deviceOp.getLoc(), allocTileVal, lockIdx,
                      static_cast<int>(0));
                  lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                  info.aie1Locks.push_back(lk);
                }
                // Use locks[0] as convenience accessor fallbacks.
                if (!info.aie1Locks.empty()) {
                  sharedProdLock = info.aie1Locks[0];
                  sharedConsLock = info.aie1Locks[0];
                  info.prodLock = info.aie1Locks[0];
                  info.consLock = info.aie1Locks[0];
                }
              }

              // Register locks keyed on the CONSUMER tile so Phase 6
              // (acquire/release lowering inside the consumer aie.core) can
              // look them up by the tile SSA value of the core it walks into.
              info.consumerTileLocks[consTileVal] = {sharedProdLock,
                                                     sharedConsLock};
              // Register buffers keyed on consumer tile for SubviewAccess.
              info.consumerTileBuffers[consTileVal] = sharedBuffers;

              // Also register on producer tile for producer-side acquires.
              info.consumerTileLocks[prodTileVal] = {sharedProdLock,
                                                     sharedConsLock};
              info.consumerTileBuffers[prodTileVal] = sharedBuffers;

              // For AIE1: register per-tile per-slot locks.
              if (!isAIE2) {
                info.consumerTileAIE1Locks[consTileVal] = info.aie1Locks;
                info.consumerTileAIE1Locks[prodTileVal] = info.aie1Locks;
              }

              // Rotation counter for depth>1 (on consumer tile — the consumer
              // core uses it, so it lives in the consumer's local memory).
              if (depth > 1 && conduitNamesWithConsumerAcquire.count(name)) {
                auto counterTy = mlir::MemRefType::get(
                    {1}, mlir::IntegerType::get(ctx, 32));
                info.rotationBuf = builder.create<AIE::BufferOp>(
                    deviceOp.getLoc(), counterTy, consTileVal,
                    /*sym_name=*/mlir::StringAttr{},
                    /*address=*/mlir::IntegerAttr{},
                    /*initial_value=*/mlir::ElementsAttr{},
                    /*mem_bank=*/mlir::IntegerAttr{});
                info.consumerTileRotationBufs[consTileVal] = info.rotationBuf;
              }

              continue; // skip the normal DMA consumer loop for this conduit
            }
          }
        }
      }

      // Phase 3j: Join source conduits — allocate buffers/locks on PRODUCER tile.
      if (linkJoinSrcNames.count(name)) {
        auto [prodCol, prodRow] = info.producerTileCoord;
        if (prodCol < 0 || prodRow < 2)
          continue;

        AIE::TileOp prodTile = lookupTileByCoord(prodCol, prodRow);
        if (!prodTile)
          continue;

        int64_t depth = info.depth > 0 ? info.depth : 1;
        mlir::Type bufTy = info.elemType;
        if (!bufTy) {
          int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
          bufTy = mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
        }

        if (insertAfterTile)
          builder.setInsertionPointAfter(insertAfterTile);
        else
          builder.setInsertionPointToStart(&deviceBody);

        mlir::Value prodTileVal = prodTile.getResult();

        for (int64_t i = 0; i < depth; ++i) {
          std::string symName = name.str() + "_buff_" + std::to_string(i);
          auto buf = builder.create<AIE::BufferOp>(
              deviceOp.getLoc(), bufTy, prodTileVal,
              mlir::StringAttr::get(ctx, symName),
              mlir::IntegerAttr{}, mlir::ElementsAttr{}, mlir::IntegerAttr{});
          info.buffers.push_back(buf);
        }

        if (isAIE2) {
          { int lockIdx = lockIdCounter[prodTileVal]++;
            std::string symName = name.str() + "_prod_lock_0";
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                deviceOp.getLoc(), prodTileVal, lockIdx, static_cast<int>(depth));
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            info.prodLock = lk; }
          { int lockIdx = lockIdCounter[prodTileVal]++;
            std::string symName = name.str() + "_cons_lock_0";
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                deviceOp.getLoc(), prodTileVal, lockIdx, 0);
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            info.consLock = lk; }
        } else {
          for (int64_t i = 0; i < depth; ++i) {
            int lockIdx = lockIdCounter[prodTileVal]++;
            std::string symName = name.str() + "_lock_" + std::to_string(i);
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                deviceOp.getLoc(), prodTileVal, lockIdx, 0);
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            info.aie1Locks.push_back(lk);
          }
          if (!info.aie1Locks.empty()) {
            info.prodLock = info.aie1Locks[0];
            info.consLock = info.aie1Locks[0];
          }
        }
        continue; // skip normal consumer loop for join sources
      }

      // Handle all consumer tiles for multi-consumer broadcast.
      int64_t depth = info.depth > 0 ? info.depth : 1;

      // Buffer type: use the actual element type from the conduit.create
      // element_type attribute if available; fall back to per-slot i32.
      mlir::Type bufTy = info.elemType;
      if (!bufTy) {
        int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
        bufTy = mlir::MemRefType::get({bufSize}, mlir::IntegerType::get(ctx, 32));
      }

      for (unsigned consIdx = 0; consIdx < info.consumerTileCoords.size();
           ++consIdx) {
        auto [consCol, consRow] = info.consumerTileCoords[consIdx];
        AIE::TileOp consTile = lookupTileByCoord(consCol, consRow);
        if (!consTile) {
          // Tile not declared in the device — skip this consumer with warning.
          module.emitWarning(
              "conduit-to-dma: consumer tile (" + std::to_string(consCol) +
              "," + std::to_string(consRow) +
              ") for conduit '" + name.str() + "' not found in device");
          continue;
        }

        // Insert after the last tile op so SSA values dominate aie.core bodies.
        if (insertAfterTile)
          builder.setInsertionPointAfter(insertAfterTile);
        else
          builder.setInsertionPointToStart(&deviceBody);

        // For the first consumer, track buffers and locks in info.buffers /
        // info.prodLock / info.consLock (used by acquire/release lowering and
        // the link phase).  Additional consumers get their own buffers/locks
        // but are not yet wired into the acquire/release path.
        mlir::Value consTileVal = consTile.getResult();

        // Allocate depth-many buffers on this consumer tile.
        // The "_cons_N" infix distinguishes consumer-tile resources from
        // shim-tile resources (Phase 4) which use the base name.
        // With multiple consumers, append the consumer index: _cons_0, _cons_1...
        // With a single consumer, use the plain _cons_ infix to match the
        // naming convention of --aie-objectFifo-stateful-transform.
        std::string bufSuffix =
            info.consumerTileCoords.size() > 1
                ? "_cons_" + std::to_string(consIdx)
                : "_cons";

        llvm::SmallVector<AIE::BufferOp> consBuffers;
        for (int64_t i = 0; i < depth; ++i) {
          std::string symName =
              name.str() + bufSuffix + "_buff_" + std::to_string(i);
          auto buf = builder.create<AIE::BufferOp>(
              deviceOp.getLoc(), bufTy, consTileVal,
              mlir::StringAttr::get(ctx, symName),
              /*address=*/mlir::IntegerAttr{},
              /*initial_value=*/mlir::ElementsAttr{},
              /*mem_bank=*/mlir::IntegerAttr{});
          consBuffers.push_back(buf);
          if (consIdx == 0)
            info.buffers.push_back(buf);
        }

        // Link source conduits: register MemTile-side buffers but skip
        // MemTile-side lock allocation.  Phase 5 allocates its own per-slice
        // lock pairs on the MemTile for distribute mode; for join mode,
        // Phase 3 already allocated per-source locks above.
        //
        // The MemTile buffers are still needed: Phase 5 references
        // srcInfo.buffers for the dma_bd descriptors in the BD chain.
        //
        // For distribute sources with a compute producer (row >= 2):
        // also allocate producer-side buffers+locks on the compute tile.
        // The compute tile's aie.mem MM2S (Phase 5.5) pushes data to
        // the MemTile's S2MM flow and needs its own local resources.
        if (linkSrcNamesEarly.count(name)) {
          // Register MemTile-side buffers (no locks — Phase 5 handles those).
          info.consumerTileBuffers[consTileVal] = consBuffers;

          // Distribute sources with a compute producer: allocate producer-
          // side buffers+locks on the compute tile for the aie.mem MM2S.
          if (linkSrcNamesEarly.count(name)) {
            auto [pCol, pRow] = info.producerTileCoord;
            if (pCol >= 0 && pRow >= 2) {
              AIE::TileOp pTile = lookupTileByCoord(pCol, pRow);
              if (pTile) {
                mlir::Value pTileVal = pTile.getResult();
                if (!info.consumerTileBuffers.count(pTileVal)) {
                  // Allocate depth-many buffers on the producer compute tile.
                  llvm::SmallVector<AIE::BufferOp> pBufs;
                  for (int64_t i = 0; i < depth; ++i) {
                    std::string symName =
                        name.str() + "_buff_" + std::to_string(i);
                    auto buf = builder.create<AIE::BufferOp>(
                        deviceOp.getLoc(), bufTy, pTileVal,
                        mlir::StringAttr::get(ctx, symName),
                        /*address=*/mlir::IntegerAttr{},
                        /*initial_value=*/mlir::ElementsAttr{},
                        /*mem_bank=*/mlir::IntegerAttr{});
                    pBufs.push_back(buf);
                  }

                  // Allocate producer-side locks.
                  AIE::LockOp pProdLock, pConsLock;
                  if (isAIE2) {
                    {
                      int lockIdx = lockIdCounter[pTileVal]++;
                      std::string symName = name.str() + "_prod_lock_0";
                      AIE::LockOp lk = builder.create<AIE::LockOp>(
                          deviceOp.getLoc(), pTileVal, lockIdx,
                          static_cast<int>(depth));
                      lk.setSymNameAttr(
                          mlir::StringAttr::get(ctx, symName));
                      pProdLock = lk;
                    }
                    {
                      int lockIdx = lockIdCounter[pTileVal]++;
                      std::string symName = name.str() + "_cons_lock_0";
                      AIE::LockOp lk = builder.create<AIE::LockOp>(
                          deviceOp.getLoc(), pTileVal, lockIdx,
                          static_cast<int>(0));
                      lk.setSymNameAttr(
                          mlir::StringAttr::get(ctx, symName));
                      pConsLock = lk;
                    }
                  } else {
                    // AIE1: one lock per buffer slot.
                    llvm::SmallVector<AIE::LockOp> pA1Locks;
                    for (int64_t i = 0; i < depth; ++i) {
                      int lockIdx = lockIdCounter[pTileVal]++;
                      std::string symName =
                          name.str() + "_lock_" + std::to_string(i);
                      AIE::LockOp lk = builder.create<AIE::LockOp>(
                          deviceOp.getLoc(), pTileVal, lockIdx,
                          static_cast<int>(0));
                      lk.setSymNameAttr(
                          mlir::StringAttr::get(ctx, symName));
                      pA1Locks.push_back(lk);
                    }
                    if (!pA1Locks.empty()) {
                      pProdLock = pA1Locks[0];
                      pConsLock = pA1Locks[0];
                    }
                    info.consumerTileAIE1Locks[pTileVal] = pA1Locks;
                  }

                  info.consumerTileBuffers[pTileVal] = pBufs;
                  info.consumerTileLocks[pTileVal] = {pProdLock, pConsLock};

                  // Rotation counter for depth > 1.
                  if (depth > 1) {
                    auto counterTy = mlir::MemRefType::get(
                        {1}, mlir::IntegerType::get(ctx, 32));
                    AIE::BufferOp rotBuf = builder.create<AIE::BufferOp>(
                        deviceOp.getLoc(), counterTy, pTileVal,
                        /*sym_name=*/mlir::StringAttr{},
                        /*address=*/mlir::IntegerAttr{},
                        /*initial_value=*/mlir::ElementsAttr{},
                        /*mem_bank=*/mlir::IntegerAttr{});
                    info.consumerTileRotationBufs[pTileVal] = rotBuf;
                  }
                }
              }
            }
          }

          continue; // advance to next consIdx (or exit the for loop)
        }

        // Allocate lock(s) on the consumer tile.
        AIE::LockOp thisProdLock;
        AIE::LockOp thisConsLock;
        if (isAIE2) {
          // AIE2: prod_lock (init=depth → free slots) + cons_lock (init=0).
          {
            int lockIdx = lockIdCounter[consTileVal]++;
            std::string symName =
                name.str() + bufSuffix + "_prod_lock_0";
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                deviceOp.getLoc(), consTileVal, lockIdx,
                static_cast<int>(depth));
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            thisProdLock = lk;
            if (consIdx == 0)
              info.prodLock = lk;
          }
          {
            int lockIdx = lockIdCounter[consTileVal]++;
            std::string symName =
                name.str() + bufSuffix + "_cons_lock_0";
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                deviceOp.getLoc(), consTileVal, lockIdx, 0);
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            thisConsLock = lk;
            if (consIdx == 0)
              info.consLock = lk;
          }
        } else {
          // AIE1: one lock per buffer slot (init=0 = empty), depth-many.
          // Each BD block i uses aie1Locks[i] for both acquire and release.
          llvm::SmallVector<AIE::LockOp> theseAIE1Locks;
          for (int64_t i = 0; i < depth; ++i) {
            int lockIdx = lockIdCounter[consTileVal]++;
            std::string symName =
                name.str() + bufSuffix + "_lock_" + std::to_string(i);
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                deviceOp.getLoc(), consTileVal, lockIdx, 0);
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            theseAIE1Locks.push_back(lk);
          }
          // Use locks[0] as convenience fallback.
          if (!theseAIE1Locks.empty()) {
            thisProdLock = theseAIE1Locks[0];
            thisConsLock = theseAIE1Locks[0];
            if (consIdx == 0) {
              info.prodLock = theseAIE1Locks[0];
              info.consLock = theseAIE1Locks[0];
              info.aie1Locks = theseAIE1Locks;
            }
          }
          // Register per-consumer-tile AIE1 lock vector for Phase 5.5.
          info.consumerTileAIE1Locks[consTileVal] = theseAIE1Locks;
        }

        // Register the lock pair for this consumer tile so Phase 6 can look
        // up the correct locks for cores on any consumer tile (not just [0]).
        info.consumerTileLocks[consTileVal] = {thisProdLock, thisConsLock};
        // Register the buffer vector for this consumer tile so Phase 6
        // SubviewAccess lowering can use the correct per-tile buffers.
        info.consumerTileBuffers[consTileVal] = consBuffers;

        // For depth>1, allocate a rotation counter buffer on each consumer tile.
        // Each consumer tile needs its own local counter — the counter lives in
        // tile-local memory so it is not accessible from other tiles.
        // Mirrors the stateful transform's `buffer_col_row : memref<1xi32>`.
        if (depth > 1 && conduitNamesWithConsumerAcquire.count(name)) {
          auto counterTy =
              mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));
          AIE::BufferOp rotBuf = builder.create<AIE::BufferOp>(
              deviceOp.getLoc(), counterTy, consTileVal,
              /*sym_name=*/mlir::StringAttr{},
              /*address=*/mlir::IntegerAttr{},
              /*initial_value=*/mlir::ElementsAttr{},
              /*mem_bank=*/mlir::IntegerAttr{});
          info.consumerTileRotationBufs[consTileVal] = rotBuf;
          if (consIdx == 0)
            info.rotationBuf = rotBuf; // fast-path for single-consumer
        }
      }
    }

    // -----------------------------------------------------------------------
    // Phase 3d: allocate producer-side buffers and locks for non-adjacent
    //           compute→compute conduits.
    // -----------------------------------------------------------------------

    for (auto &[name, info] : conduitMap) {
      if (info.sharedMemory)
        continue;
      if (linkSrcNamesEarly.count(name) || linkJoinSrcNames.count(name))
        continue;
      auto [prodCol, prodRow] = info.producerTileCoord;
      if (prodCol < 0 || prodRow < 2)
        continue;
      if (info.consumerTileCoords.empty())
        continue;

      AIE::TileOp prodTile = lookupTileByCoord(prodCol, prodRow);
      if (!prodTile)
        continue;
      mlir::Value prodTileVal = prodTile.getResult();

      if (info.consumerTileBuffers.count(prodTileVal))
        continue;

      bool needsProdSide = false;
      if (info.consumerTileCoords.size() > 1) {
        needsProdSide = true;
      } else {
        auto [consCol, consRow] = info.consumerTileCoords[0];
        if (consRow >= 2) {
          bool rightAdj = targetModel.isLegalMemAffinity(
              prodCol, prodRow, consCol, consRow);
          bool leftAdj = targetModel.isLegalMemAffinity(
              consCol, consRow, prodCol, prodRow);
          if (!rightAdj && !leftAdj)
            needsProdSide = true;
        }
      }
      if (!needsProdSide)
        continue;

      int64_t depth = info.depth > 0 ? info.depth : 1;
      mlir::Type bufTy = info.elemType;
      if (!bufTy) {
        int64_t bufSize = info.capacity > 0 ? info.capacity / depth : 1;
        bufTy = mlir::MemRefType::get({bufSize},
                                       mlir::IntegerType::get(ctx, 32));
      }

      if (insertAfterTile)
        builder.setInsertionPointAfter(insertAfterTile);
      else
        builder.setInsertionPointToStart(&deviceBody);

      llvm::SmallVector<AIE::BufferOp> prodBuffers;
      for (int64_t i = 0; i < depth; ++i) {
        std::string symName = name.str() + "_buff_" + std::to_string(i);
        auto buf = builder.create<AIE::BufferOp>(
            deviceOp.getLoc(), bufTy, prodTileVal,
            mlir::StringAttr::get(ctx, symName),
            /*address=*/mlir::IntegerAttr{},
            /*initial_value=*/mlir::ElementsAttr{},
            /*mem_bank=*/mlir::IntegerAttr{});
        prodBuffers.push_back(buf);
      }

      AIE::LockOp pProdLock, pConsLock;
      llvm::SmallVector<AIE::LockOp> pAIE1Locks;
      if (isAIE2) {
        {
          int lockIdx = lockIdCounter[prodTileVal]++;
          std::string symName = name.str() + "_prod_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), prodTileVal, lockIdx,
              static_cast<int>(depth));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          pProdLock = lk;
        }
        {
          int lockIdx = lockIdCounter[prodTileVal]++;
          std::string symName = name.str() + "_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), prodTileVal, lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          pConsLock = lk;
        }
      } else {
        for (int64_t i = 0; i < depth; ++i) {
          int lockIdx = lockIdCounter[prodTileVal]++;
          std::string symName = name.str() + "_lock_" + std::to_string(i);
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), prodTileVal, lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
          pAIE1Locks.push_back(lk);
        }
        if (!pAIE1Locks.empty()) {
          pProdLock = pAIE1Locks[0];
          pConsLock = pAIE1Locks[0];
        }
      }

      info.consumerTileLocks[prodTileVal] = {pProdLock, pConsLock};
      info.consumerTileBuffers[prodTileVal] = prodBuffers;
      if (!isAIE2)
        info.consumerTileAIE1Locks[prodTileVal] = pAIE1Locks;

      if (depth > 1 && conduitNamesWithConsumerAcquire.count(name)) {
        auto counterTy =
            mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));
        AIE::BufferOp rotBuf = builder.create<AIE::BufferOp>(
            deviceOp.getLoc(), counterTy, prodTileVal,
            /*sym_name=*/mlir::StringAttr{},
            /*address=*/mlir::IntegerAttr{},
            /*initial_value=*/mlir::ElementsAttr{},
            /*mem_bank=*/mlir::IntegerAttr{});
        info.consumerTileRotationBufs[prodTileVal] = rotBuf;
      }
    }

    // -----------------------------------------------------------------------
    // Phase 4: handle shim-tile endpoints → aie.shim_dma_allocation + aie.flow
    //
    // Two sub-cases:
    //   4a. Producer is shim (row==0): shim sends (MM2S) to compute consumer.
    //       Emits: shim_dma_allocation(MM2S), flow(shim→consumer),
    //              and shim-side prod/cons locks (needed by host runtime).
    //   4b. Consumer is shim (row==0): compute producer sends (MM2S) to shim
    //       receiver (S2MM). Emits: shim_dma_allocation(S2MM),
    //              flow(producer→shim).
    //
    // When a conduit's routing_mode is "packet", aie.packet_flow is emitted
    // instead of aie.flow.  Packet IDs are assigned sequentially starting at 0.
    // -----------------------------------------------------------------------

    // Packet flow ID counter — incremented for each packet-switched flow.
    int packetFlowID = 0;

    // Helper: emit a circuit or packet flow between two tiles.
    // When routingMode == "packet", emits aie.packet_flow with
    // aie.packet_source and aie.packet_dest inside its region.
    // Otherwise emits a plain aie.flow.
    auto emitFlow = [&](llvm::StringRef routingMode, mlir::Value srcTile,
                        AIE::WireBundle srcBundle, int32_t srcChan,
                        mlir::Value dstTile, AIE::WireBundle dstBundle,
                        int32_t dstChan) {
      if (routingMode == "packet") {
        // aie.packet_flow(ID) { aie.packet_source<src, bundle:chan>
        //                       aie.packet_dest<dst, bundle:chan> }
        auto pktFlow = builder.create<AIE::PacketFlowOp>(
            deviceOp.getLoc(),
            static_cast<int8_t>(packetFlowID++ & 0x7F),
            /*keep_pkt_header=*/mlir::BoolAttr{},
            /*priority_route=*/mlir::BoolAttr{});
        mlir::Region &region = pktFlow.getPorts();
        mlir::Block *block = builder.createBlock(&region);
        builder.setInsertionPointToStart(block);
        builder.create<AIE::PacketSourceOp>(deviceOp.getLoc(), srcTile,
                                            srcBundle,
                                            static_cast<int32_t>(srcChan));
        builder.create<AIE::PacketDestOp>(deviceOp.getLoc(), dstTile,
                                          dstBundle,
                                          static_cast<int32_t>(dstChan));
        builder.create<AIE::EndOp>(deviceOp.getLoc());
        // Restore insertion point to before the packet_flow's region.
        builder.setInsertionPointAfter(pktFlow);
      } else {
        builder.create<AIE::FlowOp>(deviceOp.getLoc(), srcTile, srcBundle,
                                    srcChan, dstTile, dstBundle, dstChan);
      }
    };

    // Track which conduit names get shim_dma_allocation (for Phase 4.5).
    llvm::DenseSet<llvm::StringRef> shimConduitNames;

    for (auto &[name, info] : conduitMap) {
      auto [prodCol, prodRow] = info.producerTileCoord;
      if (prodCol < 0)
        continue;

      // --- Sub-case 4a: producer is a shim tile (row==0) ---
      //
      // Fix NF6: broadcast correctness — emit one aie.flow per consumer tile.
      //
      // For a 1-producer → N-consumer broadcast, the shim DMA has N MM2S
      // channels (one per consumer).  Each consumer tile's DMA runs S2MM on
      // channel 0 and has its own independent buffer and lock pair (allocated
      // by Phase 3).  The stateful transform emits N flows; we must too.
      //
      // Single-consumer (N==1) case: channel index is 0 (unchanged from
      // the original implementation).  Multi-consumer (N>1): channel i sends
      // to consumer tile i via DMA channel i → DMA 0 on the consumer tile.
      if (prodRow == 0) {
        if (info.consumerTileCoords.empty())
          continue;

        AIE::TileOp shimTile = lookupTileByCoord(prodCol, prodRow);
        if (!shimTile)
          continue;

        builder.setInsertionPoint(deviceBody.getTerminator());

        // Emit shim-side prod/cons locks (matching stateful transform output).
        // These have init=0 on the shim tile.  The host runtime uses them.
        // Only emit for AIE2: AIE1 uses per-slot value locks which are not
        // needed on the shim side (the shim has no local buffer to synchronize).
        if (isAIE2) {
          int lockIdx = lockIdCounter[shimTile.getResult()]++;
          std::string symName = name.str() + "_prod_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), shimTile.getResult(), lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        }
        if (isAIE2) {
          int lockIdx = lockIdCounter[shimTile.getResult()]++;
          std::string symName = name.str() + "_cons_lock_0";
          AIE::LockOp lk = builder.create<AIE::LockOp>(
              deviceOp.getLoc(), shimTile.getResult(), lockIdx,
              static_cast<int>(0));
          lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
        }

        // aie.shim_dma_allocation @<name>_shim_alloc(%shimTile, MM2S, 0)
        // Note: shim_dma_allocation always uses channel 0 — the allocation
        // registers the shim tile as a DMA endpoint; the per-consumer channel
        // routing is expressed via aie.flow ops below.
        std::string allocSym = name.str() + "_shim_alloc";
        shimConduitNames.insert(name);
        if (!mlir::SymbolTable::lookupSymbolIn(
                deviceOp, mlir::StringAttr::get(ctx, allocSym)))
          builder.create<AIE::ShimDMAAllocationOp>(
              deviceOp.getLoc(), allocSym, shimTile.getResult(),
              AIE::DMAChannelDir::MM2S,
              /*channel_index=*/static_cast<int64_t>(0),
              /*plio=*/false,
              /*packet=*/nullptr);

        // Emit one aie.flow (or aie.packet_flow) per consumer tile (broadcast fix NF6).
        // Channel index i on the shim MM2S side → channel 0 on consumer i.
        for (unsigned consIdx = 0;
             consIdx < info.consumerTileCoords.size(); ++consIdx) {
          auto [consCol, consRow] = info.consumerTileCoords[consIdx];
          AIE::TileOp consTile = lookupTileByCoord(consCol, consRow);
          if (!consTile)
            continue;
          emitFlow(info.routingMode, shimTile.getResult(),
                   AIE::WireBundle::DMA, static_cast<int32_t>(consIdx),
                   consTile.getResult(), AIE::WireBundle::DMA,
                   static_cast<int32_t>(0));
        }
      }

      // --- Sub-case 4b: consumer is a shim tile (row==0) ---
      // Compute producer sends MM2S to shim; shim receives S2MM.
      for (auto [shimCol, shimRow] : info.shimConsumerTileCoords) {
        if (shimRow != 0)
          continue; // safety check

        AIE::TileOp prodTile = lookupTileByCoord(prodCol, prodRow);
        if (!prodTile)
          continue;

        AIE::TileOp shimTile = lookupTileByCoord(shimCol, shimRow);
        if (!shimTile)
          continue;

        builder.setInsertionPoint(deviceBody.getTerminator());

        // aie.shim_dma_allocation @<name>_shim_alloc(%shimTile, S2MM, 0)
        // The shim is the receiver when the compute tile is the producer.
        std::string allocSym = name.str() + "_shim_alloc";
        shimConduitNames.insert(name);
        if (!mlir::SymbolTable::lookupSymbolIn(
                deviceOp, mlir::StringAttr::get(ctx, allocSym)))
          builder.create<AIE::ShimDMAAllocationOp>(
              deviceOp.getLoc(), allocSym, shimTile.getResult(),
              AIE::DMAChannelDir::S2MM,
              /*channel_index=*/static_cast<int64_t>(0),
            /*plio=*/false,
            /*packet=*/nullptr);

        // aie.flow or aie.packet_flow(%prodTile, DMA:0 → %shimTile, DMA:0)
        emitFlow(info.routingMode, prodTile.getResult(),
                 AIE::WireBundle::DMA, static_cast<int32_t>(0),
                 shimTile.getResult(), AIE::WireBundle::DMA,
                 static_cast<int32_t>(0));
      }
    }

    // -----------------------------------------------------------------------
    // Phase 4.5a: emit aie.flow for non-adjacent compute→compute conduits.
    // -----------------------------------------------------------------------

    for (auto &[name, info] : conduitMap) {
      if (info.sharedMemory)
        continue;
      if (linkSrcNamesEarly.count(name) || linkJoinSrcNames.count(name))
        continue;
      auto [prodCol, prodRow] = info.producerTileCoord;
      if (prodCol < 0 || prodRow < 2)
        continue;
      if (info.consumerTileCoords.empty())
        continue;

      AIE::TileOp prodTile = lookupTileByCoord(prodCol, prodRow);
      if (!prodTile)
        continue;
      mlir::Value prodTileVal = prodTile.getResult();

      if (!info.consumerTileBuffers.count(prodTileVal))
        continue;

      builder.setInsertionPoint(deviceBody.getTerminator());

      int32_t mm2sChannel = 0;
      {
        llvm::DenseSet<int32_t> usedCh;
        deviceOp.walk([&](AIE::DMAStartOp dmaStart) {
          if (auto memOp =
                  mlir::dyn_cast<AIE::MemOp>(dmaStart->getParentOp())) {
            if (memOp.getTile() == prodTileVal &&
                dmaStart.getChannelDir() == AIE::DMAChannelDir::MM2S)
              usedCh.insert(static_cast<int32_t>(dmaStart.getChannelIndex()));
          }
        });
        while (usedCh.count(mm2sChannel))
          ++mm2sChannel;
      }

      for (unsigned consIdx = 0; consIdx < info.consumerTileCoords.size();
           ++consIdx) {
        auto [consCol, consRow] = info.consumerTileCoords[consIdx];
        if (consRow < 2)
          continue;

        bool rightAdj = targetModel.isLegalMemAffinity(
            prodCol, prodRow, consCol, consRow);
        bool leftAdj = targetModel.isLegalMemAffinity(
            consCol, consRow, prodCol, prodRow);
        if (rightAdj || leftAdj)
          continue;

        AIE::TileOp consTile = lookupTileByCoord(consCol, consRow);
        if (!consTile)
          continue;

        emitFlow(info.routingMode, prodTileVal,
                 AIE::WireBundle::DMA, mm2sChannel,
                 consTile.getResult(), AIE::WireBundle::DMA,
                 static_cast<int32_t>(0));
      }
    }

    // -----------------------------------------------------------------------
    // Phase 4.5: rewrite aiex.npu.dma_wait / aiex.npu.dma_memcpy_nd symbol
    // references from @<conduit_name> to @<conduit_name>_shim_alloc.
    //
    // Pass C emits aie.shim_dma_allocation with symbol @<name>_shim_alloc.
    // The input IR retains the original objectfifo name in the
    // aie.runtime_sequence block.  Symbol resolution fails at verify time
    // unless we update these references to match the renamed allocation.
    //
    // We use generic op attribute rewriting so we do not need to include
    // the AIEX dialect headers here.
    // -----------------------------------------------------------------------
    if (!shimConduitNames.empty()) {
      module.walk([&](mlir::Operation *op) {
        llvm::StringRef opName = op->getName().getStringRef();

        // aiex.npu.dma_wait { symbol = @<name> }
        if (opName == "aiex.npu.dma_wait") {
          if (auto symAttr =
                  op->getAttrOfType<mlir::FlatSymbolRefAttr>("symbol")) {
            llvm::StringRef ref = symAttr.getValue();
            if (shimConduitNames.count(ref)) {
              op->setAttr("symbol",
                          mlir::FlatSymbolRefAttr::get(
                              ctx, (ref + "_shim_alloc").str()));
            }
          }
          return;
        }

        // aiex.npu.dma_memcpy_nd { metadata = @<name> }
        if (opName == "aiex.npu.dma_memcpy_nd") {
          if (auto symAttr =
                  op->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata")) {
            llvm::StringRef ref = symAttr.getValue();
            if (shimConduitNames.count(ref)) {
              op->setAttr("metadata",
                          mlir::FlatSymbolRefAttr::get(
                              ctx, (ref + "_shim_alloc").str()));
            }
          } else if (auto symAttr =
                         op->getAttrOfType<mlir::SymbolRefAttr>("metadata")) {
            llvm::StringRef ref = symAttr.getRootReference().getValue();
            if (!shimConduitNames.count(ref))
              return;
            op->setAttr("metadata",
                        mlir::SymbolRefAttr::get(
                            mlir::StringAttr::get(
                                ctx, (ref + "_shim_alloc").str()),
                            symAttr.getNestedReferences()));
          }
          return;
        }
      });
    }

    // -----------------------------------------------------------------------
    // Phase 5: lower conduit.objectfifo_link → MemTile DMA BD chain
    //
    // For the distribute case (1 src → N dsts):
    //   One S2MM channel ingests the full buffer.
    //   N MM2S channels each send their own slice to one dst.
    //
    //   FIXED (Defects 1, 2, 3): N independent lock pairs on the MemTile —
    //   one pair per destination slice.  The S2MM sequences through all N
    //   slices per buffer slot (depth*N total ingest BD blocks).  Each MM2S
    //   channel i uses its own independent cons_lock_i/prod_lock_i so the
    //   channels can run concurrently without racing.  aie.flow ops connect
    //   each MM2S channel i to the corresponding compute tile's DMA S2MM 0.
    //
    // For the join case (N srcs → 1 dst):
    //   N S2MM channels each ingest their own slice.
    //   One MM2S channel outputs the joined buffer.
    //   Lock protocol uses the src conduit's existing lock pair (simple ring).
    //
    // The BD chain pattern (distribute, depth=D, N destinations):
    //   aie.memtile_dma(%memtile) {
    //     %s = aie.dma_start(S2MM, 0, ^ingest_0_0, ^mm2s_chain_0)
    //   ^ingest_0_0:   // buff_0, slice 0
    //     use_lock(prod_lock_0, AcquireGreaterEqual, 1)
    //     dma_bd(buff_0, off_0, len_0)
    //     use_lock(cons_lock_0, Release, 1)
    //     next_bd ^ingest_0_1
    //   ^ingest_0_1:   // buff_0, slice 1
    //     use_lock(prod_lock_1, AcquireGreaterEqual, 1)
    //     dma_bd(buff_0, off_1, len_1)
    //     use_lock(cons_lock_1, Release, 1)
    //     next_bd ^ingest_0_{N-1}
    //   ...
    //   ^ingest_{D-1}_{N-1}:   // last buff, last slice
    //     ...
    //     next_bd ^ingest_0_0  // ring back to buff_0, slice_0
    //   ^mm2s_chain_0:
    //     %0 = aie.dma_start(MM2S, 0, ^send_0_0, ^mm2s_chain_1)
    //   ^send_0_0:   // MM2S ch 0, buff_0
    //     use_lock(cons_lock_0, AcquireGreaterEqual, 1)  // independent lock
    //     dma_bd(buff_0, off_0, len_0)
    //     use_lock(prod_lock_0, Release, 1)
    //     next_bd ^send_0_1
    //   ...
    //   ^mm2s_chain_1:
    //     %1 = aie.dma_start(MM2S, 1, ^send_1_0, ^mm2s_chain_2)
    //   ...
    //   ^end: aie.end
    //   }
    //
    //   aie.flow(memtile, DMA:0, compute_tile_0, DMA:0)
    //   aie.flow(memtile, DMA:1, compute_tile_1, DMA:0)
    //   ...
    // -----------------------------------------------------------------------

    // linkSrcNames: for Phase 5.5 (skip aie.mem for link sources), we need ALL
    // link source names (both distribute and join).  linkSrcNamesEarly only
    // contains distribute sources (to avoid skipping Phase 3 lock allocation
    // for join sources, which Phase 5 join mode needs).  Build a full set here.
    llvm::StringSet<> linkSrcNames;
    module.walk([&](ObjectFifoLink linkOp) {
      for (auto s : linkOp.getSrcs())
        linkSrcNames.insert(mlir::cast<mlir::StringAttr>(s).getValue());
    });

    module.walk([&](ObjectFifoLink linkOp) {
      builder.setInsertionPoint(deviceBody.getTerminator());
      mlir::Location loc = linkOp.getLoc();

      auto srcs = linkOp.getSrcs();
      auto dsts = linkOp.getDsts();
      llvm::StringRef memtileStr = linkOp.getMemtile();
      llvm::StringRef mode = linkOp.getMode();
      auto offsets = linkOp.getOffsets();

      AIE::TileOp memtile = lookupTile(memtileStr);
      if (!memtile) {
        // Cannot lower without the memtile — emit a warning and skip.
        module.emitWarning("conduit-to-dma: memtile not found for link op: " +
                           memtileStr.str());
        linkOp.erase();
        return;
      }

      // Look up the src conduit (first one) for buffer/lock references.
      std::string srcName =
          mlir::cast<mlir::StringAttr>(srcs[0]).getValue().str();
      auto srcIt = conduitMap.find(srcName);
      if (srcIt == conduitMap.end() || srcIt->second.buffers.empty()) {
        // Src conduit not yet lowered or has no buffers — emit a warning and skip.
        module.emitWarning("conduit-to-dma: src conduit '" + srcName +
                           "' buffers not allocated for link op");
        linkOp.erase();
        return;
      }

      ConduitInfo &srcInfo = srcIt->second;

      // Determine depth from the src conduit (clamped to at least 1).
      int64_t linkDepth = srcInfo.depth > 0 ? srcInfo.depth : 1;
      // Per-buffer element count (capacity / depth).
      int64_t perBufLen =
          srcInfo.capacity > 0 ? srcInfo.capacity / linkDepth : 1;

      // -----------------------------------------------------------------------
      // FIX — Defects 1 & 3: Per-destination independent lock pairs on the
      // MemTile for distribute mode.
      //
      // The stateful transform allocates 2*N locks on the MemTile for an
      // 1→N distribute link: lock pair [i*2, i*2+1] is dedicated to slice i.
      // This allows all N MM2S channels to operate independently without
      // racing on a shared lock — a shared lock causes deadlock when N > 1.
      //
      // We allocate the per-slice locks here, BEFORE creating the memtile_dma,
      // so the lock SSA values are available in both the S2MM and MM2S BD
      // bodies.
      //
      // For join mode (N→1), we use the existing srcInfo locks (single pair
      // is correct: all S2MM channels compete for the same buffer slot, so
      // one cons_lock and one prod_lock is the right protocol).
      // -----------------------------------------------------------------------

      bool isDistribute = (mode == "distribute");
      unsigned numDsts = static_cast<unsigned>(dsts.size());

      // Per-slice lock pairs for distribute.  Index i → (prod_lock_i, cons_lock_i).
      llvm::SmallVector<AIE::LockOp> sliceProdLocks; // init = depth (free slots)
      llvm::SmallVector<AIE::LockOp> sliceConsLocks; // init = 0 (nothing filled)

      mlir::Value memtileVal = memtile.getResult();

      if (isDistribute && numDsts > 0) {
        // Insert lock allocs before the memtile_dma so they dominate the DMA body.
        builder.setInsertionPoint(deviceBody.getTerminator());
        for (unsigned sliceIdx = 0; sliceIdx < numDsts; ++sliceIdx) {
          if (isAIE2) {
            // AIE2: independent prod_lock_i + cons_lock_i per slice.
            {
              int lockIdx = lockIdCounter[memtileVal]++;
              std::string symName = srcName + "_link_prod_lock_" +
                                    std::to_string(sliceIdx);
              AIE::LockOp lk = builder.create<AIE::LockOp>(
                  deviceOp.getLoc(), memtileVal, lockIdx,
                  static_cast<int>(linkDepth));
              lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
              sliceProdLocks.push_back(lk);
            }
            {
              int lockIdx = lockIdCounter[memtileVal]++;
              std::string symName = srcName + "_link_cons_lock_" +
                                    std::to_string(sliceIdx);
              AIE::LockOp lk = builder.create<AIE::LockOp>(
                  deviceOp.getLoc(), memtileVal, lockIdx, 0);
              lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
              sliceConsLocks.push_back(lk);
            }
          } else {
            // AIE1: single value-based lock per slice (init=0 = empty).
            int lockIdx = lockIdCounter[memtileVal]++;
            std::string symName = srcName + "_link_lock_" +
                                  std::to_string(sliceIdx);
            AIE::LockOp lk = builder.create<AIE::LockOp>(
                deviceOp.getLoc(), memtileVal, lockIdx, 0);
            lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
            sliceProdLocks.push_back(lk);
            sliceConsLocks.push_back(lk); // same lock for AIE1
          }
        }
      }

      llvm::SmallVector<AIE::BufferOp> joinIntermediateBuffers;
      llvm::SmallVector<AIE::LockOp> joinSrcProdLocks;
      llvm::SmallVector<AIE::LockOp> joinSrcConsLocks;
      int64_t joinDstPerBufForLen = 1;

      if (!isDistribute && !dsts.empty()) {
        std::string jDstName = mlir::cast<mlir::StringAttr>(dsts[0]).getValue().str();
        auto jDstIt = conduitMap.find(jDstName);
        if (jDstIt != conduitMap.end()) {
          ConduitInfo &jDstInfo = jDstIt->second;
          int64_t jDstDepth = jDstInfo.depth > 0 ? jDstInfo.depth : 1;
          joinDstPerBufForLen = jDstInfo.capacity > 0 ? jDstInfo.capacity / jDstDepth : 1;

          mlir::Type intBufTy = jDstInfo.elemType;
          if (!intBufTy)
            intBufTy = mlir::MemRefType::get({joinDstPerBufForLen}, mlir::IntegerType::get(ctx, 32));

          builder.setInsertionPoint(deviceBody.getTerminator());
          unsigned numJoinSrcs = static_cast<unsigned>(srcs.size());

          for (int64_t i = 0; i < jDstDepth; ++i) {
            std::string symName = jDstName + "_join_buff_" + std::to_string(i);
            auto buf = builder.create<AIE::BufferOp>(deviceOp.getLoc(), intBufTy, memtileVal,
                mlir::StringAttr::get(ctx, symName), mlir::IntegerAttr{},
                mlir::ElementsAttr{}, mlir::IntegerAttr{});
            joinIntermediateBuffers.push_back(buf);
          }

          for (unsigned srcIdx = 0; srcIdx < numJoinSrcs; ++srcIdx) {
            if (isAIE2) {
              { int lockIdx = lockIdCounter[memtileVal]++;
                std::string symName = jDstName + "_join_prod_lock_" + std::to_string(srcIdx);
                AIE::LockOp lk = builder.create<AIE::LockOp>(
                    deviceOp.getLoc(), memtileVal, lockIdx, static_cast<int>(jDstDepth));
                lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                joinSrcProdLocks.push_back(lk); }
              { int lockIdx = lockIdCounter[memtileVal]++;
                std::string symName = jDstName + "_join_cons_lock_" + std::to_string(srcIdx);
                AIE::LockOp lk = builder.create<AIE::LockOp>(
                    deviceOp.getLoc(), memtileVal, lockIdx, 0);
                lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
                joinSrcConsLocks.push_back(lk); }
            } else {
              int lockIdx = lockIdCounter[memtileVal]++;
              std::string symName = jDstName + "_join_lock_" + std::to_string(srcIdx);
              AIE::LockOp lk = builder.create<AIE::LockOp>(
                  deviceOp.getLoc(), memtileVal, lockIdx, 0);
              lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
              joinSrcProdLocks.push_back(lk);
              joinSrcConsLocks.push_back(lk);
            }
          }
        }
      }

      // -----------------------------------------------------------------------
      // FIX — Defect 2: Emit aie.flow from MemTile MM2S channel i to each
      // destination compute tile's DMA S2MM channel 0.
      //
      // The stateful transform emits one aie.flow per (producer, consumer) pair
      // in splitFifos.  For a distribute link, each dst conduit is a separate
      // consumer fifo whose producer is the MemTile; the flow connects
      // MemTile MM2S channel i → compute_tile_i DMA S2MM 0.
      //
      // Without these flows, the MM2S channels have no NoC routing and no data
      // reaches the compute tiles.
      // -----------------------------------------------------------------------

      if (isDistribute) {
        // Distribute: emit aie.flow from MemTile MM2S channel i to each
        // destination compute tile's DMA S2MM channel 0.
        builder.setInsertionPoint(deviceBody.getTerminator());
        for (unsigned dstIdx = 0; dstIdx < numDsts; ++dstIdx) {
          std::string dstName =
              mlir::cast<mlir::StringAttr>(dsts[dstIdx]).getValue().str();
          auto dstIt = conduitMap.find(dstName);
          if (dstIt == conduitMap.end() ||
              dstIt->second.consumerTileCoords.empty())
            continue;

          auto [dstConsCol, dstConsRow] =
              dstIt->second.consumerTileCoords[0];
          AIE::TileOp dstConsTile =
              lookupTileByCoord(dstConsCol, dstConsRow);
          if (!dstConsTile)
            continue;

          // aie.flow(%memtile, DMA : dstIdx, %compute_tile, DMA : 0)
          builder.create<AIE::FlowOp>(
              deviceOp.getLoc(), memtileVal, AIE::WireBundle::DMA,
              static_cast<int32_t>(dstIdx), dstConsTile.getResult(),
              AIE::WireBundle::DMA, static_cast<int32_t>(0));
        }

        // Source→MemTile flow: if the distribute source has a compute-tile
        // producer (row >= 2), emit aie.flow from the producer's DMA MM2S
        // to the MemTile's DMA S2MM channel 0.  Without this flow, the
        // compute tile's MM2S has no NoC routing to the MemTile.
        {
          auto [srcProdCol, srcProdRow] = srcInfo.producerTileCoord;
          if (srcProdCol >= 0 && srcProdRow >= 2) {
            AIE::TileOp srcProdTile =
                lookupTileByCoord(srcProdCol, srcProdRow);
            if (srcProdTile) {
              builder.create<AIE::FlowOp>(
                  deviceOp.getLoc(), srcProdTile.getResult(),
                  AIE::WireBundle::DMA, static_cast<int32_t>(0),
                  memtileVal, AIE::WireBundle::DMA,
                  static_cast<int32_t>(0));
            }
          }
        }
      } else {
        // Join mode: emit aie.flow from each source compute tile MM2S 0 to
        // the MemTile S2MM channel srcIdx.
        // Also emit aie.flow from MemTile MM2S 0 to the destination (shim).
        builder.setInsertionPoint(deviceBody.getTerminator());

        // Per-source flows: compute_tile_i DMA:0 → memtile DMA:srcIdx
        for (unsigned srcIdx = 0; srcIdx < srcs.size(); ++srcIdx) {
          std::string sName =
              mlir::cast<mlir::StringAttr>(srcs[srcIdx]).getValue().str();
          auto sIt = conduitMap.find(sName);
          if (sIt == conduitMap.end())
            continue;
          // The source conduit's PRODUCER tile is the compute tile.
          auto [srcProdCol, srcProdRow] = sIt->second.producerTileCoord;
          if (srcProdCol < 0 || srcProdRow == 0)
            continue;
          AIE::TileOp srcProdTile = lookupTileByCoord(srcProdCol, srcProdRow);
          if (!srcProdTile)
            continue;
          // aie.flow(%compute_tile, DMA : 0, %memtile, DMA : srcIdx)
          builder.create<AIE::FlowOp>(
              deviceOp.getLoc(), srcProdTile.getResult(),
              AIE::WireBundle::DMA, static_cast<int32_t>(0),
              memtileVal, AIE::WireBundle::DMA,
              static_cast<int32_t>(srcIdx));
        }

        // Destination flow: memtile MM2S 0 → dst consumer (compute or shim).
        if (!dsts.empty()) {
          std::string dstName = mlir::cast<mlir::StringAttr>(dsts[0]).getValue().str();
          auto dstIt = conduitMap.find(dstName);
          if (dstIt != conduitMap.end()) {
            ConduitInfo &dstFlowInfo = dstIt->second;
            for (unsigned ci = 0; ci < dstFlowInfo.consumerTileCoords.size(); ++ci) {
              auto [consCol, consRow] = dstFlowInfo.consumerTileCoords[ci];
              AIE::TileOp consTile = lookupTileByCoord(consCol, consRow);
              if (consTile)
                builder.create<AIE::FlowOp>(deviceOp.getLoc(), memtileVal,
                    AIE::WireBundle::DMA, 0, consTile.getResult(),
                    AIE::WireBundle::DMA, 0);
            }
            for (auto [shimCol, shimRow] : dstFlowInfo.shimConsumerTileCoords) {
              AIE::TileOp shimTile = lookupTileByCoord(shimCol, shimRow);
              if (shimTile)
                builder.create<AIE::FlowOp>(deviceOp.getLoc(), memtileVal,
                    AIE::WireBundle::DMA, 0, shimTile.getResult(),
                    AIE::WireBundle::DMA, 0);
            }
          }
        }
      }

      // Create DMA block for the relay tile.
      // MemTiles (row==1 in AIE2) use aie.memtile_dma.
      // Compute tiles used as relay (shared-memory link between adjacent
      // compute tiles) use aie.mem — same BD chain structure, different op.
      builder.setInsertionPoint(deviceBody.getTerminator());
      bool relayIsMemTile =
          targetModel.isMemTile(memtile.getCol(), memtile.getRow());
      mlir::Region *dmaRegionPtr;
      if (relayIsMemTile) {
        auto memtileDMA =
            builder.create<AIE::MemTileDMAOp>(loc, memtileVal);
        dmaRegionPtr = &memtileDMA.getBody();
      } else {
        auto memOp = builder.create<AIE::MemOp>(loc, memtileVal);
        dmaRegionPtr = &memOp.getBody();
      }
      mlir::Region &dmaRegion = *dmaRegionPtr;

      // Helper: create a new basic block in the DMA region.
      auto addBlock = [&]() -> mlir::Block * {
        return builder.createBlock(&dmaRegion);
      };

      // -----------------------------------------------------------------------
      // Build the S2MM (ingest) path.
      //
      // Distribute mode (1 src → N dsts):
      //   One S2MM channel (channel 0) ingests the full source buffer.
      //   The S2MM BD ring has depth*N blocks — each buffer slot is broken
      //   into N slice blocks (one per destination).
      //
      // Join mode (N srcs → 1 dst):
      //   N S2MM channels (channels 0..N-1), one per source.
      //   Each channel i has an independent depth-many BD ring referencing
      //   source_i's buffers and lock pair.  The S2MM channel starts are
      //   chained: S2MM_entry_0 → S2MM_entry_1 → ... → S2MM_entry_{N-1}
      //   → MM2S_chain.
      // -----------------------------------------------------------------------

      // The final "next chain" block — connects the last S2MM chain to MM2S.
      mlir::Block *mm2sChainStartBlock = nullptr;

      if (isDistribute) {
        // ===== Distribute: single S2MM entry, depth*numDsts BD ring =====

        mlir::Block *entryBlock = addBlock();

        // Create depth * numDsts ingest blocks.
        llvm::SmallVector<mlir::Block *> ingestBlocks;
        for (int64_t bufIdx = 0; bufIdx < linkDepth; ++bufIdx)
          for (unsigned sliceIdx = 0; sliceIdx < numDsts; ++sliceIdx)
            ingestBlocks.push_back(addBlock());

        mm2sChainStartBlock = addBlock();

        // ^entry: dma_start(S2MM, 0, ^ingest_first, ^mm2s_chain_start)
        builder.setInsertionPointToEnd(entryBlock);
        builder.create<AIE::DMAStartOp>(
            loc, AIE::DMAChannelDir::S2MM,
            /*channel_index=*/static_cast<int32_t>(0),
            /*repeat_count=*/static_cast<int32_t>(0),
            ingestBlocks[0], mm2sChainStartBlock);

        // Fill each ingest BD block.
        unsigned totalIngest = static_cast<unsigned>(linkDepth) * numDsts;
        for (unsigned blkIdx = 0; blkIdx < totalIngest; ++blkIdx) {
          unsigned bufIdx   = blkIdx / numDsts;
          unsigned sliceIdx = blkIdx % numDsts;

          int64_t dstOffset = 0;
          int64_t dstLen    = perBufLen;
          if (offsets.has_value() && !offsets->empty()) {
            if (sliceIdx < static_cast<unsigned>(offsets->size()))
              dstOffset = (*offsets)[sliceIdx];
            if (sliceIdx + 1 < static_cast<unsigned>(offsets->size()))
              dstLen = (*offsets)[sliceIdx + 1] - dstOffset;
            else
              dstLen = perBufLen - dstOffset;
          }

          builder.setInsertionPointToEnd(ingestBlocks[blkIdx]);
          // S2MM: receive into buffer slot.
          // AIE2: acquire prod_lock (free slot), release cons_lock (data ready).
          // AIE1: acquire lock at 0 (empty), release at 1 (full) — same lock.
          builder.create<AIE::UseLockOp>(
              loc, sliceProdLocks[sliceIdx].getResult(),
              acqAction,
              static_cast<int32_t>(isAIE2 ? 1 : aie1S2MMAcqVal));
          builder.create<AIE::DMABDOp>(
              loc, srcInfo.buffers[bufIdx].getResult(),
              static_cast<int>(dstOffset),
              static_cast<int>(dstLen));
          builder.create<AIE::UseLockOp>(
              loc, sliceConsLocks[sliceIdx].getResult(),
              AIE::LockAction::Release,
              static_cast<int32_t>(isAIE2 ? 1 : aie1S2MMRelVal));
          mlir::Block *nextIngest = ingestBlocks[(blkIdx + 1) % totalIngest];
          builder.create<AIE::NextBDOp>(loc, nextIngest);
        }

      } else {
        // Join: N independent S2MM channels using intermediate buffers on memtile
        unsigned numSrcs = static_cast<unsigned>(srcs.size());
        int64_t jDepth = static_cast<int64_t>(joinIntermediateBuffers.size());
        if (jDepth == 0) jDepth = 1;

        // Per-source offsets and lengths — use DESTINATION conduit per-buf length (Bug H fix)
        llvm::SmallVector<int64_t> srcOffsets(numSrcs, 0);
        llvm::SmallVector<int64_t> srcLens(numSrcs, 1);
        for (unsigned srcIdx = 0; srcIdx < numSrcs; ++srcIdx) {
          if (offsets.has_value() && !offsets->empty()) {
            if (srcIdx < static_cast<unsigned>(offsets->size()))
              srcOffsets[srcIdx] = (*offsets)[srcIdx];
            if (srcIdx + 1 < static_cast<unsigned>(offsets->size()))
              srcLens[srcIdx] = (*offsets)[srcIdx + 1] - srcOffsets[srcIdx];
            else
              srcLens[srcIdx] = joinDstPerBufForLen - srcOffsets[srcIdx];
          } else {
            srcLens[srcIdx] = joinDstPerBufForLen;
          }
        }

        llvm::SmallVector<mlir::Block *> s2mmEntries(numSrcs);
        llvm::SmallVector<llvm::SmallVector<mlir::Block *>> srcIngestBlocks(numSrcs);
        for (unsigned srcIdx = 0; srcIdx < numSrcs; ++srcIdx) {
          s2mmEntries[srcIdx] = addBlock();
          for (int64_t i = 0; i < jDepth; ++i)
            srcIngestBlocks[srcIdx].push_back(addBlock());
        }
        mm2sChainStartBlock = addBlock();

        for (unsigned srcIdx = 0; srcIdx < numSrcs; ++srcIdx) {
          int64_t srcOffset = srcOffsets[srcIdx];
          int64_t srcLen = srcLens[srcIdx];
          mlir::Block *nextBlock = (srcIdx + 1 < numSrcs) ?
              s2mmEntries[srcIdx + 1] : mm2sChainStartBlock;

          builder.setInsertionPointToEnd(s2mmEntries[srcIdx]);
          builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::S2MM,
              static_cast<int32_t>(srcIdx), 0,
              srcIngestBlocks[srcIdx].empty() ? nextBlock : srcIngestBlocks[srcIdx][0],
              nextBlock);

          for (int64_t i = 0; i < jDepth; ++i) {
            builder.setInsertionPointToEnd(srcIngestBlocks[srcIdx][i]);
            if (srcIdx < joinSrcProdLocks.size())
              builder.create<AIE::UseLockOp>(loc, joinSrcProdLocks[srcIdx].getResult(),
                  acqAction, static_cast<int32_t>(isAIE2 ? 1 : aie1S2MMAcqVal));
            if (!joinIntermediateBuffers.empty())
              builder.create<AIE::DMABDOp>(loc,
                  joinIntermediateBuffers[i % joinIntermediateBuffers.size()].getResult(),
                  static_cast<int>(srcOffset), static_cast<int>(srcLen));
            if (srcIdx < joinSrcConsLocks.size())
              builder.create<AIE::UseLockOp>(loc, joinSrcConsLocks[srcIdx].getResult(),
                  AIE::LockAction::Release, static_cast<int32_t>(isAIE2 ? 1 : aie1S2MMRelVal));
            builder.create<AIE::NextBDOp>(loc, srcIngestBlocks[srcIdx][(i + 1) % jDepth]);
          }
        }
      }

      // -----------------------------------------------------------------------
      // Build MM2S send chains.
      //
      // Join: single MM2S channel, depth×N interleaved BDs using intermediate
      //   buffers on the MemTile (joinIntermediateBuffers from Change 4).
      // Distribute: per-destination MM2S channels with independent lock pairs.
      // -----------------------------------------------------------------------

      mlir::Block *prevChainBlock = mm2sChainStartBlock ? mm2sChainStartBlock : addBlock();
      mlir::Block *endBlock = nullptr;

      if (!isDistribute && !joinIntermediateBuffers.empty()) {
        // Join MM2S: single channel, depth×N interleaved BDs
        unsigned numJoinSrcs = static_cast<unsigned>(srcs.size());
        int64_t jDepth = static_cast<int64_t>(joinIntermediateBuffers.size());
        unsigned totalBDs = static_cast<unsigned>(jDepth) * numJoinSrcs;

        llvm::SmallVector<int64_t> mm2sOffsets(numJoinSrcs, 0);
        llvm::SmallVector<int64_t> mm2sLens(numJoinSrcs, 1);
        for (unsigned s = 0; s < numJoinSrcs; ++s) {
          if (offsets.has_value() && !offsets->empty()) {
            if (s < static_cast<unsigned>(offsets->size()))
              mm2sOffsets[s] = (*offsets)[s];
            if (s + 1 < static_cast<unsigned>(offsets->size()))
              mm2sLens[s] = (*offsets)[s + 1] - mm2sOffsets[s];
            else
              mm2sLens[s] = joinDstPerBufForLen - mm2sOffsets[s];
          } else {
            mm2sLens[s] = joinDstPerBufForLen;
          }
        }

        llvm::SmallVector<mlir::Block *> sendBDBlocks;
        for (unsigned i = 0; i < totalBDs; ++i)
          sendBDBlocks.push_back(addBlock());
        endBlock = addBlock();

        builder.setInsertionPointToEnd(prevChainBlock);
        builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::MM2S,
            0, 0, sendBDBlocks[0], endBlock);

        for (unsigned bdIdx = 0; bdIdx < totalBDs; ++bdIdx) {
          unsigned bufIdx = bdIdx / numJoinSrcs;
          unsigned srcIdx = bdIdx % numJoinSrcs;
          builder.setInsertionPointToEnd(sendBDBlocks[bdIdx]);
          if (srcIdx < joinSrcConsLocks.size())
            builder.create<AIE::UseLockOp>(loc, joinSrcConsLocks[srcIdx].getResult(),
                acqAction, static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SAcqVal));
          builder.create<AIE::DMABDOp>(loc,
              joinIntermediateBuffers[bufIdx % joinIntermediateBuffers.size()].getResult(),
              static_cast<int>(mm2sOffsets[srcIdx]), static_cast<int>(mm2sLens[srcIdx]));
          if (srcIdx < joinSrcProdLocks.size())
            builder.create<AIE::UseLockOp>(loc, joinSrcProdLocks[srcIdx].getResult(),
                AIE::LockAction::Release, static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SRelVal));
          builder.create<AIE::NextBDOp>(loc, sendBDBlocks[(bdIdx + 1) % totalBDs]);
        }
      } else {
        // Distribute MM2S: per-destination chains (preserve existing behavior)
        for (unsigned dstIdx = 0; dstIdx < dsts.size(); ++dstIdx) {
          int64_t thisDstDepth = linkDepth;
          int64_t dstOffset = 0, dstLen = perBufLen;
          if (offsets.has_value() && !offsets->empty()) {
            if (dstIdx < static_cast<unsigned>(offsets->size()))
              dstOffset = (*offsets)[dstIdx];
            if (dstIdx + 1 < static_cast<unsigned>(offsets->size()))
              dstLen = (*offsets)[dstIdx + 1] - dstOffset;
            else
              dstLen = perBufLen - dstOffset;
          }

          AIE::LockOp mm2sAcqLock, mm2sRelLock;
          if (dstIdx < sliceConsLocks.size()) {
            mm2sAcqLock = sliceConsLocks[dstIdx];
            mm2sRelLock = sliceProdLocks[dstIdx];
          }

          llvm::SmallVector<mlir::Block *> sendBDBlocks;
          for (int64_t i = 0; i < thisDstDepth; ++i)
            sendBDBlocks.push_back(addBlock());

          mlir::Block *nextChainBlock;
          if (dstIdx + 1 == dsts.size()) {
            endBlock = addBlock();
            nextChainBlock = endBlock;
          } else {
            nextChainBlock = addBlock();
          }

          builder.setInsertionPointToEnd(prevChainBlock);
          builder.create<AIE::DMAStartOp>(loc, AIE::DMAChannelDir::MM2S,
              static_cast<int32_t>(dstIdx), 0, sendBDBlocks[0], nextChainBlock);

          for (int64_t i = 0; i < thisDstDepth; ++i) {
            builder.setInsertionPointToEnd(sendBDBlocks[i]);
            if (mm2sAcqLock)
              builder.create<AIE::UseLockOp>(loc, mm2sAcqLock.getResult(), acqAction,
                  static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SAcqVal));
            if (!srcInfo.buffers.empty())
              builder.create<AIE::DMABDOp>(loc,
                  srcInfo.buffers[i % srcInfo.buffers.size()].getResult(),
                  static_cast<int>(dstOffset), static_cast<int>(dstLen));
            if (mm2sRelLock)
              builder.create<AIE::UseLockOp>(loc, mm2sRelLock.getResult(),
                  AIE::LockAction::Release,
                  static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SRelVal));
            builder.create<AIE::NextBDOp>(loc, sendBDBlocks[(i + 1) % thisDstDepth]);
          }
          prevChainBlock = nextChainBlock;
        }
      }

      // If dsts is empty (degenerate case), create the end block now.
      if (!endBlock)
        endBlock = addBlock();

      // ^end: aie.end
      builder.setInsertionPointToEnd(endBlock);
      builder.create<AIE::EndOp>(loc);

      linkOp.erase();
    });

    // -----------------------------------------------------------------------
    // Phase 5.5: for each simple (non-link) conduit that has a non-shim
    // producer tile and at least one buffer, generate an aie.mem BD chain
    // on the consumer tile so the DMA engine can transfer data.
    //
    // The pattern mirrors createAIETileDMA in AIEObjectFifoStatefulTransform:
    //
    //   aie.mem(%consTile) {
    //     %0 = aie.dma_start(S2MM, 0, ^bd_0, ^end)
    //   ^bd_0:
    //     aie.use_lock(%prod_lock, AcquireGreaterEqual, 1)
    //     aie.dma_bd(%buff_0 : memref<...>, 0, len)
    //     aie.use_lock(%cons_lock, Release, 1)
    //     aie.next_bd ^bd_1           // depth-1: loops back to ^bd_0
    //   ^bd_1:                        // depth > 1 only
    //     ...
    //     aie.next_bd ^bd_0
    //   ^end: aie.end
    //   }
    //
    // We only emit aie.mem for compute tiles (row > 1).  MemTile (row==1) and
    // shim (row==0) use memtile_dma and shim_dma_allocation respectively, which
    // are handled in Phases 4 and 5.
    //
    // Limitation: If a conduit participates in a link (its buffers/locks were
    // consumed by Phase 5), the aie.mem will still reference the same lock SSA
    // values.  For now we only generate aie.mem for conduits whose consumer
    // tile is a compute tile and that did not appear as the src of any link op
    // processed above.  Conduits that are link sources are skipped here because
    // the link phase already handles the DMA for those buffers.
    // -----------------------------------------------------------------------

    // linkSrcNames was populated before Phase 5 (above) so it correctly
    // contains the source conduit names even though Phase 5 has erased all
    // ObjectFifoLink ops by the time Phase 5.5 runs.

    for (auto &[name, info] : conduitMap) {
      if (info.buffers.empty() || !info.prodLock || !info.consLock)
        continue;

      // Handle link source conduits that have a compute-tile producer and a
      // memtile consumer (join pattern).  The memtile_dma S2MM channels have
      // already been generated by Phase 5; what is still missing is the
      // aie.mem MM2S block on each source compute tile that pushes data into
      // the memtile's S2MM flow.
      //
      // For join sources:
      //   producerTileCoord  = compute tile (row >= 2)
      //   consumerTileCoords = [memtile]   (row == 1)
      //
      // We emit: aie.mem(%prodTile) { dma_start(MM2S, 0, ^bd_0, ^end) ... }
      // using the existing consumerTile-side lock pair (info.prodLock /
      // info.consLock), which physically reside on the memtile.  The compute
      // tile's aie.mem body may reference memtile locks when the tiles are
      // memory-adjacent, which is true for xcve2302/npu row-1 memtiles.
      //
      // Distribute sources (memtile → compute tiles) do NOT need this — the
      // distribute memtile_dma already handles MM2S to compute tiles.  Only
      // skip those (isDistribute link sources).
      if (linkSrcNames.count(name)) {
        // Link source conduit: emit aie.mem MM2S on the producer compute tile
        // to push data into the MemTile's S2MM flow.
        //
        // Distribute sources (linkSrcNamesEarly): use producer-side buffers+locks
        // allocated by Phase 3 (Change 1) on the compute tile.
        //
        // Join sources (linkJoinSrcNames): use memtile-side buffers/locks
        // (allocated by Phase 3).  The compute tile's aie.mem body references
        // memtile locks — valid only when tiles are memory-adjacent (xcve2302:
        // same column, compute row directly above memtile row).
        if (linkSrcNamesEarly.count(name)) {
          // Distribute source with compute producer: emit aie.mem MM2S using
          // producer-side resources (consumerTileBuffers/Locks[prodTileVal]).
          auto [prodCol, prodRow] = info.producerTileCoord;
          if (prodRow < 2)
            continue; // degenerate: distribute producer is shim or memtile

          AIE::TileOp prodTile = lookupTileByCoord(prodCol, prodRow);
          if (!prodTile)
            continue;

          mlir::Value prodTileVal = prodTile.getResult();

          // Look up producer-side buffers+locks allocated by Phase 3 Change 1.
          auto bufIt = info.consumerTileBuffers.find(prodTileVal);
          auto lockIt = info.consumerTileLocks.find(prodTileVal);
          if (bufIt == info.consumerTileBuffers.end() ||
              bufIt->second.empty() ||
              lockIt == info.consumerTileLocks.end())
            continue;

          llvm::SmallVector<AIE::BufferOp> &prodBuffers = bufIt->second;
          AIE::LockOp pProdLock = lockIt->second.first;
          AIE::LockOp pConsLock = lockIt->second.second;

          int64_t depth = info.depth > 0 ? info.depth : 1;
          int64_t perBufLen = info.capacity > 0 ? info.capacity / depth : 1;

          builder.setInsertionPoint(deviceBody.getTerminator());
          auto memOp =
              builder.create<AIE::MemOp>(deviceOp.getLoc(), prodTileVal);
          mlir::Region &memRegion = memOp.getBody();
          auto addMemBlock = [&]() -> mlir::Block * {
            return builder.createBlock(&memRegion);
          };
          mlir::Block *dmaStartBlock = addMemBlock();
          llvm::SmallVector<mlir::Block *> bdBlocks;
          for (int64_t i = 0; i < depth; ++i)
            bdBlocks.push_back(addMemBlock());
          mlir::Block *endMemBlock = addMemBlock();

          // dma_start(MM2S, 0, ^bd_0, ^end)
          builder.setInsertionPointToEnd(dmaStartBlock);
          builder.create<AIE::DMAStartOp>(deviceOp.getLoc(),
                                          AIE::DMAChannelDir::MM2S,
                                          static_cast<int32_t>(0),
                                          static_cast<int32_t>(0),
                                          bdBlocks[0], endMemBlock);

          // BD ring: MM2S direction (sending from compute tile to MemTile).
          // AIE2: acquire cons_lock (data ready), release prod_lock (free slot).
          // AIE1: per-slot lock — acquire(1) [full], release(0) [empty].
          llvm::SmallVector<AIE::LockOp> *pA1Locks = nullptr;
          {
            auto a1It = info.consumerTileAIE1Locks.find(prodTileVal);
            if (a1It != info.consumerTileAIE1Locks.end() &&
                !a1It->second.empty())
              pA1Locks = &a1It->second;
          }
          for (int64_t i = 0; i < depth; ++i) {
            builder.setInsertionPointToEnd(bdBlocks[i]);
            mlir::Value blockAcq =
                isAIE2 ? pConsLock.getResult()
                       : (pA1Locks && !pA1Locks->empty()
                              ? (*pA1Locks)[i % pA1Locks->size()].getResult()
                              : pConsLock.getResult());
            mlir::Value blockRel =
                isAIE2 ? pProdLock.getResult() : blockAcq;
            builder.create<AIE::UseLockOp>(
                deviceOp.getLoc(), blockAcq, acqAction,
                static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SAcqVal));
            builder.create<AIE::DMABDOp>(
                deviceOp.getLoc(),
                prodBuffers[i % prodBuffers.size()].getResult(),
                0, static_cast<int>(perBufLen));
            builder.create<AIE::UseLockOp>(
                deviceOp.getLoc(), blockRel,
                AIE::LockAction::Release,
                static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SRelVal));
            builder.create<AIE::NextBDOp>(deviceOp.getLoc(),
                                          bdBlocks[(i + 1) % depth]);
          }
          builder.setInsertionPointToEnd(endMemBlock);
          builder.create<AIE::EndOp>(deviceOp.getLoc());
          continue; // distribute source handled
        }

        // Join sources: compute producer (row >= 2) sending to memtile (row == 1).
        // NOTE: The join source's buffers/locks are on the memtile consumer side
        // (allocated by Phase 3).  The compute tile's aie.mem body references
        // memtile locks — valid only when tiles are memory-adjacent (xcve2302:
        // same column, compute row directly above memtile row).
        if (!linkJoinSrcNames.count(name))
          continue; // unknown link source — skip

        auto [prodCol, prodRow] = info.producerTileCoord;
        if (prodRow < 2)
          continue; // degenerate: join producer is shim or memtile

        // Emit aie.mem MM2S on the producer compute tile.
        AIE::TileOp prodTile = lookupTileByCoord(prodCol, prodRow);
        if (!prodTile)
          continue;

        int64_t depth = info.depth > 0 ? info.depth : 1;
        int64_t perBufLen = info.capacity > 0 ? info.capacity / depth : 1;

        builder.setInsertionPoint(deviceBody.getTerminator());
        auto memOp =
            builder.create<AIE::MemOp>(deviceOp.getLoc(), prodTile.getResult());
        mlir::Region &memRegion = memOp.getBody();
        auto addMemBlock = [&]() -> mlir::Block * {
          return builder.createBlock(&memRegion);
        };
        mlir::Block *dmaStartBlock = addMemBlock();
        llvm::SmallVector<mlir::Block *> bdBlocks;
        for (int64_t i = 0; i < depth; ++i)
          bdBlocks.push_back(addMemBlock());
        mlir::Block *endMemBlock = addMemBlock();

        // dma_start(MM2S, 0, ^bd_0, ^end)
        builder.setInsertionPointToEnd(dmaStartBlock);
        builder.create<AIE::DMAStartOp>(deviceOp.getLoc(),
                                        AIE::DMAChannelDir::MM2S,
                                        static_cast<int32_t>(0),
                                        static_cast<int32_t>(0),
                                        bdBlocks[0], endMemBlock);

        // BD ring: MM2S direction (sending from compute tile to memtile S2MM).
        // AIE2: acquire cons_lock (data ready), release prod_lock (free slot).
        // AIE1: each BD block i uses aie1Locks[i] — one lock per slot.
        //        acquire(1) [full], release(0) [empty], same lock per block.
        for (int64_t i = 0; i < depth; ++i) {
          builder.setInsertionPointToEnd(bdBlocks[i]);
          mlir::Value blockLock = isAIE2 ? info.consLock.getResult()
              : (info.aie1Locks.empty()
                     ? info.consLock.getResult()
                     : info.aie1Locks[i % info.aie1Locks.size()].getResult());
          mlir::Value blockRelLock = isAIE2 ? info.prodLock.getResult()
              : blockLock; // same lock for AIE1
          builder.create<AIE::UseLockOp>(
              deviceOp.getLoc(), blockLock,
              acqAction,
              static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SAcqVal));
          builder.create<AIE::DMABDOp>(
              deviceOp.getLoc(),
              info.buffers[i % info.buffers.size()].getResult(),
              0, static_cast<int>(perBufLen));
          builder.create<AIE::UseLockOp>(
              deviceOp.getLoc(), blockRelLock,
              AIE::LockAction::Release,
              static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SRelVal));
          builder.create<AIE::NextBDOp>(deviceOp.getLoc(),
                                        bdBlocks[(i + 1) % depth]);
        }
        builder.setInsertionPointToEnd(endMemBlock);
        builder.create<AIE::EndOp>(deviceOp.getLoc());
        continue; // join source handled — skip the rest of the loop body
      }

      // Skip shared memory conduits: they require no DMA BD chain.
      // Buffers and locks on the producer tile are directly accessible from
      // both producer and consumer cores via shared memory; no aie.mem needed.
      if (info.sharedMemory)
        continue;

      // Case C: non-adjacent compute→compute MM2S on producer tile.
      // Does NOT continue — falls through to Case A for consumer S2MM.
      {
        auto [prodCol, prodRow] = info.producerTileCoord;
        if (prodCol >= 0 && prodRow >= 2 &&
            !linkSrcNames.count(name) &&
            !info.consumerTileCoords.empty()) {
          AIE::TileOp prodTile = lookupTileByCoord(prodCol, prodRow);
          if (prodTile) {
            mlir::Value prodTileVal = prodTile.getResult();
            auto bufIt = info.consumerTileBuffers.find(prodTileVal);
            if (bufIt != info.consumerTileBuffers.end() &&
                !bufIt->second.empty()) {
              llvm::SmallVector<AIE::BufferOp> &prodBuffers = bufIt->second;
              int64_t depth = info.depth > 0 ? info.depth : 1;
              int64_t perBufLen =
                  info.capacity > 0 ? info.capacity / depth : 1;

              AIE::LockOp mm2sAcqLock, mm2sRelLock;
              llvm::SmallVector<AIE::LockOp> *prodAIE1Locks = nullptr;
              {
                auto lockIt = info.consumerTileLocks.find(prodTileVal);
                if (lockIt != info.consumerTileLocks.end()) {
                  mm2sAcqLock = lockIt->second.second; // consLock
                  mm2sRelLock = lockIt->second.first;  // prodLock
                }
                auto aie1It = info.consumerTileAIE1Locks.find(prodTileVal);
                if (aie1It != info.consumerTileAIE1Locks.end() &&
                    !aie1It->second.empty())
                  prodAIE1Locks = &aie1It->second;
              }

              if (mm2sAcqLock && mm2sRelLock) {
                int32_t mm2sChannel = 0;
                {
                  llvm::DenseSet<int32_t> usedCh;
                  deviceOp.walk([&](AIE::DMAStartOp dmaStart) {
                    if (auto memOp = mlir::dyn_cast<AIE::MemOp>(
                            dmaStart->getParentOp())) {
                      if (memOp.getTile() == prodTileVal &&
                          dmaStart.getChannelDir() ==
                              AIE::DMAChannelDir::MM2S)
                        usedCh.insert(static_cast<int32_t>(
                            dmaStart.getChannelIndex()));
                    }
                  });
                  while (usedCh.count(mm2sChannel))
                    ++mm2sChannel;
                }

                AIE::MemOp existingMemOp;
                deviceOp.walk([&](AIE::MemOp memOp) {
                  if (memOp.getTile() == prodTileVal)
                    existingMemOp = memOp;
                });

                if (existingMemOp) {
                  mlir::Region &memRegion = existingMemOp.getBody();
                  mlir::Block *endBlock = nullptr;
                  for (mlir::Block &block : memRegion) {
                    for (mlir::Operation &opInBlock : block) {
                      if (mlir::isa<AIE::EndOp>(opInBlock)) {
                        endBlock = &block;
                        break;
                      }
                    }
                    if (endBlock)
                      break;
                  }

                  if (endBlock) {
                    mlir::Operation *oldEnd = endBlock->getTerminator();
                    auto addBlock = [&]() -> mlir::Block * {
                      return builder.createBlock(&memRegion);
                    };
                    llvm::SmallVector<mlir::Block *> bdBlocks;
                    for (int64_t i = 0; i < depth; ++i)
                      bdBlocks.push_back(addBlock());
                    mlir::Block *newEndBlock = addBlock();

                    builder.setInsertionPointToEnd(endBlock);
                    oldEnd->erase();
                    builder.create<AIE::DMAStartOp>(
                        deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
                        mm2sChannel, static_cast<int32_t>(0),
                        bdBlocks[0], newEndBlock);

                    for (int64_t i = 0; i < depth; ++i) {
                      builder.setInsertionPointToEnd(bdBlocks[i]);
                      mlir::Value blockAcq =
                          isAIE2 ? mm2sAcqLock.getResult()
                                 : (prodAIE1Locks && !prodAIE1Locks->empty()
                                        ? (*prodAIE1Locks)
                                              [i % prodAIE1Locks->size()]
                                                  .getResult()
                                        : mm2sAcqLock.getResult());
                      mlir::Value blockRel =
                          isAIE2 ? mm2sRelLock.getResult() : blockAcq;
                      builder.create<AIE::UseLockOp>(
                          deviceOp.getLoc(), blockAcq, acqAction,
                          static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SAcqVal));
                      builder.create<AIE::DMABDOp>(
                          deviceOp.getLoc(),
                          prodBuffers[i % prodBuffers.size()].getResult(),
                          0, static_cast<int>(perBufLen));
                      builder.create<AIE::UseLockOp>(
                          deviceOp.getLoc(), blockRel,
                          AIE::LockAction::Release,
                          static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SRelVal));
                      builder.create<AIE::NextBDOp>(
                          deviceOp.getLoc(), bdBlocks[(i + 1) % depth]);
                    }
                    builder.setInsertionPointToEnd(newEndBlock);
                    builder.create<AIE::EndOp>(deviceOp.getLoc());
                  }
                } else {
                  builder.setInsertionPoint(deviceBody.getTerminator());
                  auto memOp = builder.create<AIE::MemOp>(
                      deviceOp.getLoc(), prodTileVal);
                  mlir::Region &memRegion = memOp.getBody();
                  auto addBlock = [&]() -> mlir::Block * {
                    return builder.createBlock(&memRegion);
                  };
                  mlir::Block *dmaStartBlock = addBlock();
                  llvm::SmallVector<mlir::Block *> bdBlocks;
                  for (int64_t i = 0; i < depth; ++i)
                    bdBlocks.push_back(addBlock());
                  mlir::Block *endBlock = addBlock();

                  builder.setInsertionPointToEnd(dmaStartBlock);
                  builder.create<AIE::DMAStartOp>(
                      deviceOp.getLoc(), AIE::DMAChannelDir::MM2S,
                      mm2sChannel, static_cast<int32_t>(0),
                      bdBlocks[0], endBlock);

                  for (int64_t i = 0; i < depth; ++i) {
                    builder.setInsertionPointToEnd(bdBlocks[i]);
                    mlir::Value blockAcq =
                        isAIE2 ? mm2sAcqLock.getResult()
                               : (prodAIE1Locks && !prodAIE1Locks->empty()
                                      ? (*prodAIE1Locks)
                                            [i % prodAIE1Locks->size()]
                                                .getResult()
                                      : mm2sAcqLock.getResult());
                    mlir::Value blockRel =
                        isAIE2 ? mm2sRelLock.getResult() : blockAcq;
                    builder.create<AIE::UseLockOp>(
                        deviceOp.getLoc(), blockAcq, acqAction,
                        static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SAcqVal));
                    builder.create<AIE::DMABDOp>(
                        deviceOp.getLoc(),
                        prodBuffers[i % prodBuffers.size()].getResult(),
                        0, static_cast<int>(perBufLen));
                    builder.create<AIE::UseLockOp>(
                        deviceOp.getLoc(), blockRel,
                        AIE::LockAction::Release,
                        static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SRelVal));
                    builder.create<AIE::NextBDOp>(
                        deviceOp.getLoc(), bdBlocks[(i + 1) % depth]);
                  }
                  builder.setInsertionPointToEnd(endBlock);
                  builder.create<AIE::EndOp>(deviceOp.getLoc());
                }
              }
            }
          }
        }
      }
      // Case C falls through — consumer S2MM still needed from Case A.

      // Determine which tile gets the aie.mem and what DMA direction to use.
      //
      // Case A: shim producer → compute consumer.
      //   aie.mem on the CONSUMER tile, S2MM (receives data from shim).
      //   This is the common case handled originally.
      //
      // Case B: compute producer → shim consumer.
      //   aie.mem on the PRODUCER tile, MM2S (sends data to shim).
      //   In this case consumerTileCoords is empty; shimConsumerTileCoords
      //   has the shim tile.  The buffer/locks were allocated on the producer
      //   tile by Phase 3b.
      //   BD lock semantics for MM2S:
      //     acquire consLock (data filled, ready to send)
      //     dma_bd(buff, offset, len)
      //     release prodLock (slot freed for next fill)

      AIE::TileOp dmaHostTile;
      AIE::DMAChannelDir dmaDir;
      AIE::LockOp acqLock;  // lock to acquire before DMA transfer
      AIE::LockOp relLock;  // lock to release after DMA transfer

      bool isProducerToShim = info.consumerTileCoords.empty() &&
                              !info.shimConsumerTileCoords.empty();

      if (isProducerToShim) {
        // Case B: compute tile sends MM2S to shim.
        auto [prodCol, prodRow] = info.producerTileCoord;
        if (prodCol < 0 || prodRow == 0)
          continue;
        dmaHostTile = lookupTileByCoord(prodCol, prodRow);
        if (!dmaHostTile)
          continue;
        // Only compute tiles (row >= 2) use aie.mem.
        if (prodRow < 2)
          continue;
        dmaDir = AIE::DMAChannelDir::MM2S;
        // MM2S BD: wait for data filled (consLock), send, signal slot free (prodLock).
        acqLock = info.consLock;
        relLock = info.prodLock;

        int64_t depth = info.depth > 0 ? info.depth : 1;
        int64_t perBufLen = info.capacity > 0 ? info.capacity / depth : 1;

        // Create aie.mem on the producer tile (single tile for Case B).
        builder.setInsertionPoint(deviceBody.getTerminator());
        auto memOp = builder.create<AIE::MemOp>(deviceOp.getLoc(),
                                                dmaHostTile.getResult());
        mlir::Region &memRegion = memOp.getBody();
        auto addMemBlock = [&]() -> mlir::Block * {
          return builder.createBlock(&memRegion);
        };
        mlir::Block *dmaStartBlock = addMemBlock();
        llvm::SmallVector<mlir::Block *> bdBlocks;
        for (int64_t i = 0; i < depth; ++i)
          bdBlocks.push_back(addMemBlock());
        mlir::Block *endMemBlock = addMemBlock();
        builder.setInsertionPointToEnd(dmaStartBlock);
        builder.create<AIE::DMAStartOp>(deviceOp.getLoc(), dmaDir,
                                        static_cast<int32_t>(0),
                                        static_cast<int32_t>(0),
                                        bdBlocks[0], endMemBlock);
        // MM2S BD: wait for data filled, send, signal slot freed.
        // AIE2: acquire cons_lock(1), release prod_lock(1).
        // AIE1: each BD block i uses aie1Locks[i] — one lock per slot.
        //        acquire(1) [full], release(0) [empty], same lock per block.
        for (int64_t i = 0; i < depth; ++i) {
          builder.setInsertionPointToEnd(bdBlocks[i]);
          // For AIE1, use per-slot lock; for AIE2, use the shared acqLock.
          mlir::Value blockAcqVal = isAIE2 ? acqLock.getResult()
              : (info.aie1Locks.empty()
                     ? acqLock.getResult()
                     : info.aie1Locks[i % info.aie1Locks.size()].getResult());
          mlir::Value blockRelVal = isAIE2 ? relLock.getResult()
              : (info.aie1Locks.empty()
                     ? relLock.getResult()
                     : info.aie1Locks[i % info.aie1Locks.size()].getResult());
          builder.create<AIE::UseLockOp>(deviceOp.getLoc(), blockAcqVal,
                                         acqAction,
                                         static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SAcqVal));
          builder.create<AIE::DMABDOp>(deviceOp.getLoc(),
                                       info.buffers[i].getResult(), 0,
                                       static_cast<int>(perBufLen));
          builder.create<AIE::UseLockOp>(deviceOp.getLoc(), blockRelVal,
                                         AIE::LockAction::Release,
                                         static_cast<int32_t>(isAIE2 ? 1 : aie1MM2SRelVal));
          builder.create<AIE::NextBDOp>(deviceOp.getLoc(),
                                        bdBlocks[(i + 1) % depth]);
        }
        builder.setInsertionPointToEnd(endMemBlock);
        builder.create<AIE::EndOp>(deviceOp.getLoc());

      } else {
        // Case A: shim sends S2MM to compute consumer tile(s).
        //
        // Fix NF6 (Phase 5.5): for broadcast (N consumer tiles), emit one
        // aie.mem per consumer tile, each using that tile's own buffer vector
        // and lock pair from consumerTileBuffers / consumerTileLocks.
        //
        // Single-consumer case: consumerTileCoords.size() == 1 → exactly one
        // aie.mem is emitted (same behavior as before this fix).
        if (info.consumerTileCoords.empty())
          continue;

        int64_t depth = info.depth > 0 ? info.depth : 1;
        int64_t perBufLen = info.capacity > 0 ? info.capacity / depth : 1;
        dmaDir = AIE::DMAChannelDir::S2MM;

        for (unsigned consIdx = 0; consIdx < info.consumerTileCoords.size();
             ++consIdx) {
          auto [consCol, consRow] = info.consumerTileCoords[consIdx];
          dmaHostTile = lookupTileByCoord(consCol, consRow);
          if (!dmaHostTile)
            continue;
          // Only compute tiles (row >= 2) use aie.mem.
          if (consRow < 2)
            continue;

          // Resolve the per-consumer-tile buffer vector, lock pair, and AIE1 locks.
          // Falls back to info.buffers / prodLock / consLock for single-
          // consumer conduits where consumerTileBuffers may be empty.
          llvm::SmallVector<AIE::BufferOp> *tileBuffers = &info.buffers;
          AIE::LockOp tileProdLock = info.prodLock;
          AIE::LockOp tileConsLock = info.consLock;
          llvm::SmallVector<AIE::LockOp> *tileAIE1Locks = &info.aie1Locks;
          {
            mlir::Value consTileVal = dmaHostTile.getResult();
            auto bufIt = info.consumerTileBuffers.find(consTileVal);
            if (bufIt != info.consumerTileBuffers.end() &&
                !bufIt->second.empty())
              tileBuffers = &bufIt->second;
            auto lockIt = info.consumerTileLocks.find(consTileVal);
            if (lockIt != info.consumerTileLocks.end()) {
              tileProdLock = lockIt->second.first;
              tileConsLock = lockIt->second.second;
            }
            auto aie1It = info.consumerTileAIE1Locks.find(consTileVal);
            if (aie1It != info.consumerTileAIE1Locks.end() &&
                !aie1It->second.empty())
              tileAIE1Locks = &aie1It->second;
          }

          if (!tileProdLock || !tileConsLock || tileBuffers->empty())
            continue;

          // S2MM BD: wait for free slot (prodLock), receive, signal filled (consLock).
          acqLock = tileProdLock;
          relLock = tileConsLock;

          // Create aie.mem on this consumer tile.
          builder.setInsertionPoint(deviceBody.getTerminator());
          auto memOp = builder.create<AIE::MemOp>(deviceOp.getLoc(),
                                                  dmaHostTile.getResult());
          mlir::Region &memRegion = memOp.getBody();
          auto addMemBlock = [&]() -> mlir::Block * {
            return builder.createBlock(&memRegion);
          };
          mlir::Block *dmaStartBlock = addMemBlock();
          llvm::SmallVector<mlir::Block *> bdBlocks;
          for (int64_t i = 0; i < depth; ++i)
            bdBlocks.push_back(addMemBlock());
          mlir::Block *endMemBlock = addMemBlock();
          builder.setInsertionPointToEnd(dmaStartBlock);
          builder.create<AIE::DMAStartOp>(deviceOp.getLoc(), dmaDir,
                                          static_cast<int32_t>(0),
                                          static_cast<int32_t>(0),
                                          bdBlocks[0], endMemBlock);
          // S2MM BD: wait for free slot, receive, signal filled.
          // AIE2: acquire prod_lock(1) [free slot], release cons_lock(1) [data ready].
          // AIE1: each BD block i uses tileAIE1Locks[i] — one lock per slot.
          //        acquire(0) [empty], release(1) [full], same lock per block.
          for (int64_t i = 0; i < depth; ++i) {
            builder.setInsertionPointToEnd(bdBlocks[i]);
            // For AIE1: use per-slot lock; for AIE2: use the shared acqLock/relLock.
            mlir::Value blockLockAcq = isAIE2 ? acqLock.getResult()
                : (tileAIE1Locks->empty()
                       ? acqLock.getResult()
                       : (*tileAIE1Locks)[i % tileAIE1Locks->size()].getResult());
            mlir::Value blockLockRel = isAIE2 ? relLock.getResult()
                : (tileAIE1Locks->empty()
                       ? relLock.getResult()
                       : (*tileAIE1Locks)[i % tileAIE1Locks->size()].getResult());
            builder.create<AIE::UseLockOp>(
                deviceOp.getLoc(), blockLockAcq,
                acqAction, static_cast<int32_t>(isAIE2 ? 1 : aie1S2MMAcqVal));
            builder.create<AIE::DMABDOp>(
                deviceOp.getLoc(), (*tileBuffers)[i % tileBuffers->size()].getResult(),
                0, static_cast<int>(perBufLen));
            builder.create<AIE::UseLockOp>(
                deviceOp.getLoc(), blockLockRel,
                AIE::LockAction::Release, static_cast<int32_t>(isAIE2 ? 1 : aie1S2MMRelVal));
            builder.create<AIE::NextBDOp>(deviceOp.getLoc(),
                                          bdBlocks[(i + 1) % depth]);
          }
          builder.setInsertionPointToEnd(endMemBlock);
          builder.create<AIE::EndOp>(deviceOp.getLoc());
        }
      }
    }

    // -----------------------------------------------------------------------
    // Phase 6: lower conduit.acquire / conduit.release inside func bodies
    //          → aie.use_lock
    //
    // Port semantics:
    //   port="Consume" acquire → AcquireGreaterEqual on consLock
    //   port="Produce" acquire → AcquireGreaterEqual on prodLock
    //   port="Consume" release → Release on prodLock (free up the prod slot)
    //   port="Produce" release → Release on consLock (signal data ready)
    // -----------------------------------------------------------------------

    // Lowering order (use-before-def safety):
    //   1. SubviewAccess  — uses Acquire's window result; erase immediately.
    //   2. Release        — uses Acquire's window result to find conduit name;
    //                       emit use_lock now, but defer erase until after
    //                       Acquire is erased (Release is a user of Acquire).
    //   3. Acquire        — emit use_lock; erase (no remaining users now).
    //   4. Erase Release ops collected in step 2.

    // Step 1: SubviewAccess — replace memref result with aie.buffer, then erase.
    //
    // For depth == 1: direct replacement with buffers[idx] (static selection).
    //
    // For depth > 1: emit a dynamic ping-pong selection using the rotation
    // counter buffer allocated in Phase 3.  The pattern mirrors the stateful
    // transform's counter + scf.index_switch approach:
    //
    //   %ctr_val = memref.load %rotation_buf[%c0]
    //   %ctr_idx = arith.index_cast %ctr_val : i32 to index
    //   // offset by the subview index (for sliding windows with count>1)
    //   %abs_idx = arith.addi %ctr_idx, %idx_const  // (% depth handled by switch)
    //   %selected = scf.index_switch %abs_idx -> memref<T>
    //     case 0 { scf.yield %buff_0 }
    //     case 1 { scf.yield %buff_1 }
    //     ...
    //     default { scf.yield %buff_0 }
    //
    // The counter increment (counter = (counter+1) % depth) is emitted in
    // Step 4 after the Release use_lock, once per acquire-release pair.
    //
    // Fix NF3: the window operand of SubviewAccess may come from either:
    //   a) conduit.acquire  (blocking path)  — look up name via acqOp.getName()
    //   b) conduit.wait_window (async path)  — look up name via waitOp.getName()
    // If neither matches, the pass emits a hard error (no silent fallback).
    module.walk([&](SubviewAccess op) {
      llvm::StringRef conduitName;
      llvm::StringRef acquirePort; // port attribute of the defining Acquire op
      if (auto acqOp = mlir::dyn_cast_or_null<Acquire>(
              op.getWindow().getDefiningOp())) {
        conduitName = acqOp.getName();
        acquirePort = acqOp.getPort();
      } else if (auto waitOp = mlir::dyn_cast_or_null<WaitWindow>(
                   op.getWindow().getDefiningOp()))
        conduitName = waitOp.getName();

      bool replaced = false;
      if (!conduitName.empty()) {
        auto it = conduitMap.find(conduitName);
        if (it != conduitMap.end() && !it->second.buffers.empty()) {
          int64_t idx = op.getIndex();
          ConduitInfo &cinfo = it->second;
          int64_t depth = cinfo.depth > 0 ? cinfo.depth : 1;

          // Broadcast buffer correctness: resolve to the buffer vector for the
          // specific consumer tile that contains this SubviewAccess.
          // Falls back to cinfo.buffers (consumer_tile[0]) when not found.
          llvm::SmallVector<AIE::BufferOp> *tileBuffers = &cinfo.buffers;
          // Also resolve the per-tile rotation counter buffer.  Each consumer
          // tile has its own local rotation counter (tile-local memory cannot
          // be accessed cross-tile).  Falls back to cinfo.rotationBuf.
          AIE::BufferOp tileRotationBuf = cinfo.rotationBuf;
          {
            mlir::Operation *coreOp = op->getParentOp();
            while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
              coreOp = coreOp->getParentOp();
            if (coreOp) {
              mlir::Value coreTile =
                  mlir::cast<AIE::CoreOp>(coreOp).getTile();
              auto bufIt = cinfo.consumerTileBuffers.find(coreTile);
              if (bufIt != cinfo.consumerTileBuffers.end())
                tileBuffers = &bufIt->second;
              auto rotIt = cinfo.consumerTileRotationBufs.find(coreTile);
              if (rotIt != cinfo.consumerTileRotationBufs.end())
                tileRotationBuf = rotIt->second;
            }
          }

          {
            // Clamp idx to valid buffer range via modulo.
            // For sliding-window acquires (count > depth), idx may exceed
            // depth-1: e.g., acquire(count=3) on a depth=1 fifo produces
            // subview_access at indices 0, 1, 2 but only buffers[0] exists.
            // Wrapping with % depth ensures all indices resolve to a valid
            // buffer without an out-of-range crash.
            int64_t bufIdx = static_cast<int64_t>(tileBuffers->size()) > 1
                                 ? idx % static_cast<int64_t>(tileBuffers->size())
                                 : 0;
            // Port-gated rotation counter: only Consume-port acquires use the
            // dynamic rotation counter path.  Producer-port SubviewAccess ops
            // reference buffers allocated on the consumer tile; the rotation
            // buffer (rotationBuf) is also on the consumer tile and is
            // inaccessible from the producer core.  Use static selection for
            // Produce-port acquires regardless of depth.
            bool useStaticSelection =
                (depth == 1 || !tileRotationBuf || acquirePort == "Produce");
            if (useStaticSelection) {
              // Depth-1 case, no rotation buffer, or produce-port: static selection.
              mlir::Value bufVal = (*tileBuffers)[bufIdx].getResult();
              if (bufVal.getType() == op.getResult().getType()) {
                op.getResult().replaceAllUsesWith(bufVal);
                replaced = true;
              }
            } else {
              // Depth>1 case, Consume port: dynamic selection via counter + index_switch.
              builder.setInsertionPoint(op);
              mlir::Location loc = op.getLoc();

              // Load the rotation counter (per-tile buffer for broadcast correctness).
              mlir::Value c0Idx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
              mlir::Value ctrI32 = builder.create<mlir::memref::LoadOp>(
                  loc, tileRotationBuf.getResult(),
                  mlir::ValueRange{c0Idx});

              // Cast to index for index_switch.
              mlir::Value ctrIdx = builder.create<mlir::arith::IndexCastOp>(
                  loc, builder.getIndexType(), ctrI32);

              // Add the subview index offset (for sliding windows idx>0).
              mlir::Value absIdx = ctrIdx;
              if (idx > 0) {
                mlir::Value idxConst =
                    builder.create<mlir::arith::ConstantIndexOp>(loc, idx);
                // Compute (counter + idx) % depth for correct wrap-around.
                mlir::Value sum = builder.create<mlir::arith::AddIOp>(
                    loc, ctrIdx, idxConst);
                mlir::Value depthConst =
                    builder.create<mlir::arith::ConstantIndexOp>(loc, depth);
                absIdx = builder.create<mlir::arith::RemUIOp>(loc, sum,
                                                               depthConst);
              }

              // Emit scf.index_switch over [0..depth-1].
              mlir::Type bufTy = op.getResult().getType();
              llvm::SmallVector<int64_t> caseVals;
              for (int64_t i = 0; i < depth; ++i)
                caseVals.push_back(i);

              auto switchOp = builder.create<mlir::scf::IndexSwitchOp>(
                  loc, mlir::TypeRange{bufTy}, absIdx, caseVals,
                  /*numRegions=*/static_cast<int>(depth));

              // Fill each case region: yield tileBuffers[i].
              for (int64_t i = 0; i < depth; ++i) {
                mlir::Block *caseBlock =
                    &switchOp.getCaseRegions()[i].emplaceBlock();
                mlir::OpBuilder caseBuilder(ctx);
                caseBuilder.setInsertionPointToEnd(caseBlock);
                caseBuilder.create<mlir::scf::YieldOp>(
                    loc, (*tileBuffers)[i].getResult());
              }
              // Default region: yield tileBuffers[0].
              {
                mlir::Block *defBlock =
                    &switchOp.getDefaultRegion().emplaceBlock();
                mlir::OpBuilder defBuilder(ctx);
                defBuilder.setInsertionPointToEnd(defBlock);
                defBuilder.create<mlir::scf::YieldOp>(
                    loc, (*tileBuffers)[0].getResult());
              }

              op.getResult().replaceAllUsesWith(switchOp.getResult(0));
              replaced = true;
            }
          }
        }
      }
      // If the buffer could not be resolved, emit a hard error and fail the
      // pass.  A silent memref.alloc fallback would re-introduce the garbage-
      // memory correctness bug that Fix 1 was designed to eliminate.
      if (!replaced) {
        op.emitError("conduit-to-dma: SubviewAccess could not be resolved to "
                     "an allocated aie.buffer — type mismatch, index out of "
                     "range, or conduit not in map");
        return signalPassFailure();
      }
      op.erase();
    });

    // Step 2: Release — emit use_lock; collect for deferred erase.
    // Must happen BEFORE Acquire is erased because Release holds a reference
    // to Acquire's window result (needed to look up the conduit name).
    //
    // Fix NF3: the window operand of Release may come from either:
    //   a) conduit.acquire  (blocking path)  — look up name via acqOp.getName()
    //   b) conduit.wait_window (async path)  — look up name via waitOp.getName()
    llvm::SmallVector<Release> releasesToErase;
    module.walk([&](Release op) {
      // conduit.release takes a !conduit.window<T> operand; derive the conduit
      // name by walking up to the defining conduit.acquire (or wait_window) op.
      llvm::StringRef conduitName;
      if (auto acqOp = mlir::dyn_cast_or_null<Acquire>(
              op.getWindow().getDefiningOp()))
        conduitName = acqOp.getName();
      else if (auto waitOp = mlir::dyn_cast_or_null<WaitWindow>(
                   op.getWindow().getDefiningOp()))
        conduitName = waitOp.getName();

      if (conduitName.empty()) {
        releasesToErase.push_back(op);
        return;
      }
      auto it = conduitMap.find(conduitName);
      if (it == conduitMap.end()) {
        releasesToErase.push_back(op);
        return;
      }
      ConduitInfo &cinfo = it->second;
      builder.setInsertionPoint(op);
      int64_t count = static_cast<int64_t>(op.getCount());
      llvm::StringRef port = op.getPort();

      // Broadcast lock correctness: find the enclosing aie.core tile and look
      // up that tile's specific lock pair from consumerTileLocks.  Falls back
      // to cinfo.prodLock/consLock (consumer_tile[0]) if the tile is not found
      // (e.g., producer-side release outside a core, or no multi-consumer map).
      // Also resolve the per-tile rotation counter buffer for broadcast.
      AIE::LockOp resolvedProdLock = cinfo.prodLock;
      AIE::LockOp resolvedConsLock = cinfo.consLock;
      AIE::BufferOp resolvedRotationBuf = cinfo.rotationBuf;
      {
        mlir::Operation *coreOp = op->getParentOp();
        while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
          coreOp = coreOp->getParentOp();
        if (coreOp) {
          mlir::Value coreTile = mlir::cast<AIE::CoreOp>(coreOp).getTile();
          auto lockIt = cinfo.consumerTileLocks.find(coreTile);
          if (lockIt != cinfo.consumerTileLocks.end()) {
            resolvedProdLock = lockIt->second.first;
            resolvedConsLock = lockIt->second.second;
          }
          auto rotIt = cinfo.consumerTileRotationBufs.find(coreTile);
          if (rotIt != cinfo.consumerTileRotationBufs.end())
            resolvedRotationBuf = rotIt->second;
        }
      }

      // Consumer releases prod-lock (freeing up producer slots);
      // Producer releases cons-lock (signalling data ready for consumer).
      AIE::LockOp lock =
          (port == "Consume") ? resolvedProdLock : resolvedConsLock;
      if (lock) {
        // AIE1: value-based release — Consumer releases to 0 (empty),
        //       Producer releases to 1 (full).  AIE2: release by 'count'.
        int32_t relVal = isAIE2 ? static_cast<int32_t>(count)
                                : (port == "Consume" ? 0 : 1);
        builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                       AIE::LockAction::Release,
                                       relVal);
      }
      // For depth>1 with Consume port, emit counter increment after the
      // release use_lock.  Counter advances by 'count' (the number of
      // elements released) mod depth, matching the stateful transform pattern.
      // Uses the per-tile rotation buffer for broadcast correctness.
      if (resolvedRotationBuf && port == "Consume" && cinfo.depth > 1) {
        mlir::Location loc = op.getLoc();
        mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
        mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        // Load current counter.
        mlir::Value curI32 = builder.create<mlir::memref::LoadOp>(
            loc, resolvedRotationBuf.getResult(), mlir::ValueRange{c0});
        // Increment by count.
        mlir::Value incI32 = mlir::arith::ConstantIntOp::create(
            builder, loc, i32Ty, count);
        mlir::Value newVal =
            builder.create<mlir::arith::AddIOp>(loc, curI32, incI32);
        // Wrap modulo depth: if newVal >= depth, subtract depth.
        mlir::Value depthI32 = mlir::arith::ConstantIntOp::create(
            builder, loc, i32Ty, cinfo.depth);
        mlir::Value cmp = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sge, newVal, depthI32);
        mlir::Value wrapped =
            builder.create<mlir::arith::SubIOp>(loc, newVal, depthI32);
        mlir::Value result =
            builder.create<mlir::arith::SelectOp>(loc, cmp, wrapped, newVal);
        builder.create<mlir::memref::StoreOp>(
            loc, result, resolvedRotationBuf.getResult(), mlir::ValueRange{c0});
      }
      releasesToErase.push_back(op);
    });

    // Step 3: Erase Release ops BEFORE erasing Acquire.
    // Release holds a use of the Acquire's window SSA result.  Erasing
    // Acquire first while Release is still live causes "destroyed with uses".
    for (auto op : releasesToErase)
      op.erase();

    // Step 4: Acquire — emit use_lock (and counter init on first use); erase.
    // SubviewAccess (step 1) and Release (step 3) users are gone; safe to erase.
    //
    // Counter initialization: the rotation counter must be set to 0 before the
    // first acquire for each depth>1 conduit in each core.  Track which
    // (conduit, parent-op) pairs have been initialized to emit the store once.
    llvm::DenseSet<std::pair<llvm::StringRef, mlir::Operation *>>
        counterInitialized;

    module.walk([&](Acquire op) {
      auto it = conduitMap.find(op.getName());
      if (it == conduitMap.end()) {
        op.erase();
        return;
      }
      ConduitInfo &cinfo = it->second;
      builder.setInsertionPoint(op);
      int64_t count = static_cast<int64_t>(op.getCount());
      llvm::StringRef port = op.getPort();

      // Broadcast lock correctness: resolve the lock pair for the specific
      // consumer tile that contains this acquire, using consumerTileLocks.
      // Also resolve the per-tile rotation counter buffer for broadcast.
      AIE::LockOp resolvedProdLock = cinfo.prodLock;
      AIE::LockOp resolvedConsLock = cinfo.consLock;
      AIE::BufferOp resolvedRotationBuf = cinfo.rotationBuf;
      mlir::Operation *acquireCoreOp = op->getParentOp();
      while (acquireCoreOp && !mlir::isa<AIE::CoreOp>(acquireCoreOp))
        acquireCoreOp = acquireCoreOp->getParentOp();
      {
        if (acquireCoreOp) {
          mlir::Value coreTile = mlir::cast<AIE::CoreOp>(acquireCoreOp).getTile();
          auto lockIt = cinfo.consumerTileLocks.find(coreTile);
          if (lockIt != cinfo.consumerTileLocks.end()) {
            resolvedProdLock = lockIt->second.first;
            resolvedConsLock = lockIt->second.second;
          }
          auto rotIt = cinfo.consumerTileRotationBufs.find(coreTile);
          if (rotIt != cinfo.consumerTileRotationBufs.end())
            resolvedRotationBuf = rotIt->second;
        }
      }

      // Consumer acquires consume-lock; producer acquires produce-lock.
      AIE::LockOp lock =
          (port == "Produce") ? resolvedProdLock : resolvedConsLock;

      // For depth>1 Consume acquires: initialize the rotation counter to 0
      // at the start of the enclosing aie.core body (once per core+conduit).
      // Uses the per-tile rotation buffer for broadcast correctness.
      if (resolvedRotationBuf && port == "Consume" && cinfo.depth > 1) {
        auto key = std::make_pair(op.getName(), acquireCoreOp);
        if (acquireCoreOp && !counterInitialized.count(key)) {
          counterInitialized.insert(key);
          // Insert at the very start of the core body's entry block, before
          // any acquire-related ops.
          mlir::Block *entryBlock = &acquireCoreOp->getRegion(0).front();
          mlir::OpBuilder initBuilder(entryBlock, entryBlock->begin());
          mlir::Location loc = op.getLoc();
          mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
          mlir::Value zero = mlir::arith::ConstantIntOp::create(
              initBuilder, loc, i32Ty, 0);
          mlir::Value c0 =
              initBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
          initBuilder.create<mlir::memref::StoreOp>(
              loc, zero, resolvedRotationBuf.getResult(),
              mlir::ValueRange{c0});
        }
      }

      if (lock) {
        // AIE1: value-based acquire — Consumer acquires at 1 (full),
        //       Producer acquires at 0 (empty).  AIE2: acquire by 'count'.
        int32_t acqVal = isAIE2 ? static_cast<int32_t>(count)
                                : (port == "Produce" ? 0 : 1);
        builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                       acqAction,
                                       acqVal);
      }
      op.erase();
    });

    // -----------------------------------------------------------------------
    // Phase 7: erase remaining Conduit ops (create, wait).
    // SubviewAccess and Acquire/Release were already erased in Phase 6.
    // conduit.annotate was removed from the dialect; no Annotate erase.
    //
    // Fix NF5: conduit.wait ops (used as DMA token synchronization fences)
    // are not lowered to any hardware op — the program ordering they encode
    // is already captured by the surrounding lock acquire/release sequence.
    // They must be erased here so they do not survive into the output IR as
    // dangling Conduit ops.  Erase conduit.wait BEFORE conduit.create so
    // that any wait ops inside device body are caught first.
    // -----------------------------------------------------------------------

    module.walk([&](Wait op) { op.erase(); });
    module.walk([&](Create op) { op.erase(); });
    // conduit.wait_all_async merges tokens but has no hardware lowering — the
    // ordering it encodes is already captured by the surrounding lock sequences.
    // Its token result is consumed only by conduit.wait (already erased above)
    // or other wait_all_async ops.
    //
    // Collect before erasing in reverse order: MLIR walk() visits ops in
    // pre-order within a block, so a def (waa_A) is collected before its use
    // (waa_B that chains waa_A's result).  Erasing in forward pre-order would
    // destroy waa_A while waa_B still holds a use of its result.  Reversing
    // the erase order ensures uses (waa_B) are destroyed before their defs
    // (waa_A), satisfying MLIR's "no erase with live uses" invariant.
    {
      llvm::SmallVector<WaitAllAsync> waitAllAsyncsToErase;
      module.walk([&](WaitAllAsync op) {
        waitAllAsyncsToErase.push_back(op);
      });
      for (auto op : llvm::reverse(waitAllAsyncsToErase))
        op.erase();
    }

    // -----------------------------------------------------------------------
    // Phase 8: lower conduit.acquire_async / conduit.wait_window / conduit.wait_all
    //
    // The async acquire path enables DMA/compute overlap:
    //
    //   %tok = conduit.acquire_async {name="x", count=1, port="Consume"}
    //                                 : !conduit.async.token
    //   // ... DMA or other async work here ...
    //   %win = conduit.wait_window %tok for "x"
    //                               : !conduit.async.token -> !conduit.window<T>
    //   // use %win via conduit.subview_access ...
    //
    // Lowering:
    //   acquire_async → emit NOTHING (deferred to wait_window)
    //   wait_window   → emit aie.use_lock(consLock, AcquireGreaterEqual, count)
    //                    and replace the window result with the buffer SSA value
    //   wait_all      → for each acquire_async token: emit use_lock
    //
    // Moving the use_lock later in the instruction stream gives the DMA more
    // time to complete before the core reaches the lock acquire — the key
    // performance claim for the async path.
    //
    // Step 8a: Collect acquire_async metadata BEFORE erasing the ops.
    // We record (token SSA value → {conduit name, port, count}) so that
    // wait_window and wait_all can look up the lock info after the acquire_async
    // is gone.
    // -----------------------------------------------------------------------

    struct AsyncAcquireInfo {
      std::string conduitName;
      std::string port;
      int64_t count;
    };
    llvm::DenseMap<mlir::Value, AsyncAcquireInfo> asyncAcquireMap;

    // Step 8a: Record acquire_async metadata (do NOT erase yet).
    // The AcquireAsync op must stay alive until after WaitWindow is processed,
    // because WaitWindow's getToken() SSA operand points to the AcquireAsync
    // result.  Erasing AcquireAsync before WaitWindow causes use-after-erase.
    // We only erase AcquireAsync ops in Step 8a-erase, after WaitWindow is done.
    llvm::SmallVector<AcquireAsync> asyncAcquiresToErase;
    module.walk([&](AcquireAsync op) {
      AsyncAcquireInfo info;
      info.conduitName = op.getName().str();
      // acquire_async has no port attribute — it always represents a consumer
      // acquire (waiting for data to become available for reading).
      info.port = "Consume";
      info.count = static_cast<int64_t>(op.getCount());
      asyncAcquireMap[op.getToken()] = info;
      asyncAcquiresToErase.push_back(op);
    });

    // Step 8b: Lower conduit.wait_window.
    // For each wait_window, the token's defining acquire_async recorded the
    // conduit name.  Emit use_lock at this point (the actual hardware wait),
    // then replace the window result with the allocated aie.buffer SSA value.
    llvm::SmallVector<WaitWindow> waitWindowsToErase;
    module.walk([&](WaitWindow op) {
      // The conduit name is an attribute on the wait_window op itself.
      llvm::StringRef conduitName = op.getName();
      auto it = conduitMap.find(conduitName);
      if (it == conduitMap.end()) {
        op.emitError("conduit-to-dma: wait_window references unknown conduit '")
            << conduitName << "'";
        signalPassFailure();
        return;
      }
      ConduitInfo &cinfo = it->second;

      // Determine port and count from the async acquire map (keyed by token).
      // After reading, erase the entry so Step 8c (wait_all) does NOT also
      // emit use_lock for tokens that wait_window has already handled.
      // This fixes the double-emit bug: without the erase, both wait_window
      // and wait_all emit AcquireGreaterEqual for the same lock.
      llvm::StringRef port = "Consume";
      int64_t count = 1;
      {
        auto ait = asyncAcquireMap.find(op.getToken());
        if (ait != asyncAcquireMap.end()) {
          port = ait->second.port;
          count = ait->second.count;
          asyncAcquireMap.erase(ait); // consumed — skip in Step 8c
        }
      }

      // Resolve the lock pair for the core that contains this wait_window.
      AIE::LockOp resolvedProdLock = cinfo.prodLock;
      AIE::LockOp resolvedConsLock = cinfo.consLock;
      {
        mlir::Operation *coreOp = op->getParentOp();
        while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
          coreOp = coreOp->getParentOp();
        if (coreOp) {
          mlir::Value coreTile = mlir::cast<AIE::CoreOp>(coreOp).getTile();
          auto lockIt = cinfo.consumerTileLocks.find(coreTile);
          if (lockIt != cinfo.consumerTileLocks.end()) {
            resolvedProdLock = lockIt->second.first;
            resolvedConsLock = lockIt->second.second;
          }
        }
      }

      // Emit the deferred use_lock at this point in the instruction stream.
      // AIE1: Consumer acquires at 1 (full), Producer acquires at 0 (empty).
      builder.setInsertionPoint(op);
      AIE::LockOp lock = (port == "Produce") ? resolvedProdLock : resolvedConsLock;
      if (lock) {
        int32_t acqVal = isAIE2 ? static_cast<int32_t>(count)
                                : (port == "Produce" ? 0 : 1);
        builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                       acqAction,
                                       acqVal);
      }

      // Replace the window result with the appropriate aie.buffer SSA value.
      // Same logic as Phase 6 Step 1 (SubviewAccess): use per-tile buffer map.
      llvm::SmallVector<AIE::BufferOp> *tileBuffers = &cinfo.buffers;
      {
        mlir::Operation *coreOp = op->getParentOp();
        while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
          coreOp = coreOp->getParentOp();
        if (coreOp) {
          mlir::Value coreTile = mlir::cast<AIE::CoreOp>(coreOp).getTile();
          auto bufIt = cinfo.consumerTileBuffers.find(coreTile);
          if (bufIt != cinfo.consumerTileBuffers.end())
            tileBuffers = &bufIt->second;
        }
      }

      // The window result type wraps the element memref — replace all uses of
      // the window with the buffer.  Users will be conduit.subview_access ops
      // which Phase 6 Step 1 will have already replaced with aie.buffer refs;
      // any remaining users get the first buffer as a fallback.
      if (!tileBuffers->empty() && op.getResult().use_empty() == false) {
        // If the window SSA value has surviving users (e.g., conduit.subview_access
        // that Phase 6 missed), replace with buffer[0].
        mlir::Value bufVal = (*tileBuffers)[0].getResult();
        if (bufVal.getType() == op.getResult().getType()) {
          op.getResult().replaceAllUsesWith(bufVal);
        }
      }

      waitWindowsToErase.push_back(op);
    });
    for (auto op : waitWindowsToErase)
      op.erase();

    // Step 8c: Lower conduit.wait_all containing acquire_async tokens.
    // Must happen BEFORE erasing AcquireAsync (Step 8a-erase) because
    // WaitAll holds SSA uses of the AcquireAsync token results.
    llvm::SmallVector<WaitAll> waitAllToErase;
    module.walk([&](WaitAll op) {
      builder.setInsertionPoint(op);
      for (mlir::Value tok : op.getTokens()) {
        auto ait = asyncAcquireMap.find(tok);
        if (ait == asyncAcquireMap.end())
          continue; // not an acquire_async token — skip (DMA token, etc.)

        const AsyncAcquireInfo &ainfo = ait->second;
        auto it = conduitMap.find(ainfo.conduitName);
        if (it == conduitMap.end())
          continue;
        ConduitInfo &cinfo = it->second;

        // Resolve per-tile locks.
        AIE::LockOp resolvedProdLock = cinfo.prodLock;
        AIE::LockOp resolvedConsLock = cinfo.consLock;
        {
          mlir::Operation *coreOp = op->getParentOp();
          while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
            coreOp = coreOp->getParentOp();
          if (coreOp) {
            mlir::Value coreTile = mlir::cast<AIE::CoreOp>(coreOp).getTile();
            auto lockIt = cinfo.consumerTileLocks.find(coreTile);
            if (lockIt != cinfo.consumerTileLocks.end()) {
              resolvedProdLock = lockIt->second.first;
              resolvedConsLock = lockIt->second.second;
            }
          }
        }

        AIE::LockOp lock =
            (ainfo.port == "Produce") ? resolvedProdLock : resolvedConsLock;
        if (lock) {
          // AIE1: Consumer acquires at 1 (full), Producer acquires at 0 (empty).
          int32_t acqVal = isAIE2 ? static_cast<int32_t>(ainfo.count)
                                  : (ainfo.port == "Produce" ? 0 : 1);
          builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                         acqAction,
                                         acqVal);
        }
      }
      waitAllToErase.push_back(op);
    });
    for (auto op : waitAllToErase)
      op.erase();

    // Step 8a-erase: NOW safe to erase AcquireAsync ops.
    // WaitWindow (Step 8b) and WaitAll (Step 8c) are both gone, so no op
    // holds a use of the AcquireAsync token SSA value anymore.
    for (auto op : asyncAcquiresToErase)
      op.erase();

    // Step 8d: Lower conduit.release_async.
    //
    // release_async {name="x", count=N} : !conduit.window.token
    //
    // Lowering:
    //   aie.use_lock(prodLock, Release, N)
    //
    // The spec says "Hardware lowering: aie.use_lock(prodLock, Release, count)".
    // Like the blocking conduit.release with port="Consume", the async variant
    // signals the producer that N buffer slots are free (releases prodLock).
    // The !conduit.window.token result has no hardware equivalent; its only
    // consumers are conduit.wait_all_async ops, which were erased in Phase 7.
    llvm::SmallVector<ReleaseAsync> releaseAsyncsToErase;
    module.walk([&](ReleaseAsync op) {
      llvm::StringRef conduitName = op.getName();
      auto it = conduitMap.find(conduitName);
      if (it == conduitMap.end()) {
        // Unknown conduit — skip silently (no hardware allocation was made).
        releaseAsyncsToErase.push_back(op);
        return;
      }
      ConduitInfo &cinfo = it->second;

      // Resolve per-tile lock pair and rotation buffer: find the enclosing
      // aie.core tile and look up that tile's locks and rotation counter.
      // Falls back to cinfo.prodLock/consLock / cinfo.rotationBuf if not
      // inside a core or no per-tile entry exists (same pattern as blocking
      // Release).
      llvm::StringRef port = op.getPort();
      AIE::LockOp resolvedProdLock = cinfo.prodLock;
      AIE::LockOp resolvedConsLock = cinfo.consLock;
      AIE::BufferOp resolvedRotationBuf = cinfo.rotationBuf;
      {
        mlir::Operation *coreOp = op->getParentOp();
        while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
          coreOp = coreOp->getParentOp();
        if (coreOp) {
          mlir::Value coreTile = mlir::cast<AIE::CoreOp>(coreOp).getTile();
          auto lockIt = cinfo.consumerTileLocks.find(coreTile);
          if (lockIt != cinfo.consumerTileLocks.end()) {
            resolvedProdLock = lockIt->second.first;
            resolvedConsLock = lockIt->second.second;
          }
          auto rotIt = cinfo.consumerTileRotationBufs.find(coreTile);
          if (rotIt != cinfo.consumerTileRotationBufs.end())
            resolvedRotationBuf = rotIt->second;
        }
      }

      builder.setInsertionPoint(op);
      int64_t count = static_cast<int64_t>(op.getCount());
      // Consumer releases prod-lock (freeing up producer slots);
      // Producer releases cons-lock (signalling data ready for consumer).
      AIE::LockOp lock =
          (port == "Consume") ? resolvedProdLock : resolvedConsLock;
      if (lock) {
        // AIE1: value-based release — Consumer releases to 0 (empty),
        //       Producer releases to 1 (full).  AIE2: release by 'count'.
        int32_t relVal = isAIE2 ? static_cast<int32_t>(count)
                                : (port == "Consume" ? 0 : 1);
        builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                       AIE::LockAction::Release,
                                       relVal);
      }
      // For depth>1 with Consume port: increment the rotation counter
      // after the release, matching the blocking Release path (Step 3).
      if (resolvedRotationBuf && port == "Consume" && cinfo.depth > 1) {
        mlir::Location loc = op.getLoc();
        mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
        mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        mlir::Value curI32 = builder.create<mlir::memref::LoadOp>(
            loc, resolvedRotationBuf.getResult(), mlir::ValueRange{c0});
        mlir::Value incI32 = mlir::arith::ConstantIntOp::create(
            builder, loc, i32Ty, count);
        mlir::Value newVal =
            builder.create<mlir::arith::AddIOp>(loc, curI32, incI32);
        mlir::Value depthI32 = mlir::arith::ConstantIntOp::create(
            builder, loc, i32Ty, cinfo.depth);
        mlir::Value cmp = builder.create<mlir::arith::CmpIOp>(
            loc, mlir::arith::CmpIPredicate::sge, newVal, depthI32);
        mlir::Value wrapped =
            builder.create<mlir::arith::SubIOp>(loc, newVal, depthI32);
        mlir::Value result =
            builder.create<mlir::arith::SelectOp>(loc, cmp, wrapped, newVal);
        builder.create<mlir::memref::StoreOp>(
            loc, result, resolvedRotationBuf.getResult(), mlir::ValueRange{c0});
      }
      releaseAsyncsToErase.push_back(op);
    });
    for (auto op : releaseAsyncsToErase)
      op.erase();

    // -----------------------------------------------------------------------
    // Steps 8e–8h: erase conduit put/get memref ops.
    //
    // These ops have no hardware lowering in Pass C (DMA descriptor emission
    // for the producer side is a separate future gap).  They must be erased
    // AFTER Step 8c (WaitAll) because conduit.wait_all can hold uses of
    // PutMemrefAsync / GetMemrefAsync token results as variadic operands.
    // Erasing the token-producing op before WaitAll is gone would destroy an
    // op that still has live SSA uses — use-after-erase in debug builds.
    //
    // Correct ordering:
    //   8e: PutMemrefAsync (result !conduit.dma.token; consumers gone by 8c)
    //   8f: GetMemrefAsync (same)
    //   8g: PutMemref  (no result; no ordering constraint)
    //   8h: GetMemref  (no result; no ordering constraint)
    //
    // Use llvm::reverse() for consistency with the WaitAllAsync erasure
    // pattern in Phase 7.
    // -----------------------------------------------------------------------

    // Step 8e: erase conduit.put_memref_async.
    {
      llvm::SmallVector<PutMemrefAsync> toErase;
      module.walk([&](PutMemrefAsync op) { toErase.push_back(op); });
      for (auto op : llvm::reverse(toErase))
        op.erase();
    }

    // Step 8f: erase conduit.get_memref_async.
    {
      llvm::SmallVector<GetMemrefAsync> toErase;
      module.walk([&](GetMemrefAsync op) { toErase.push_back(op); });
      for (auto op : llvm::reverse(toErase))
        op.erase();
    }

    // Step 8g: erase conduit.put_memref (blocking, no result).
    module.walk([&](PutMemref op) { op.erase(); });

    // Step 8h: erase conduit.get_memref (blocking, no result).
    module.walk([&](GetMemref op) { op.erase(); });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConduitToDMAPass() {
  return std::make_unique<ConduitToDMAPass>();
}

} // namespace xilinx::conduit
