//===- ConduitFuseOperators.cpp - conduit-fuse-operators pass ----*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --conduit-fuse-operators: Spatial IRON operator fusion via Conduit IR.
//
// Given two aie.device ops in one module (e.g., GEMV at col=0 and ReLU at
// col=1), this pass:
//
//   1. Identifies the "output channel" of device A: a conduit.create whose
//      only consumers are shim tiles (row==0), meaning it exits to LPDDR5.
//   2. Identifies the "input channel" of device B: a conduit.create whose
//      producer tile is a shim tile (row==0), meaning it reads from LPDDR5.
//   3. Matches channels by element_type (unambiguous for single-output/
//      single-input operators like GEMV→ReLU).
//   4. Offsets all tile coordinates in device B by +C_max+1, where C_max is
//      the maximum column index in device A. This ensures no coordinate
//      conflicts when Pass C walks all devices together.
//   5. Emits a module-level conduit.create @fused_intermediate_N with:
//        routing_mode = "any"  (resolved to shared-mem by
//        --conduit-infer-modes) depth = 0             (sentinel; resolved by
//        --conduit-depth-promote) producer_tile = tile from A's output (after
//        offset: still col A_max) consumer_tiles = tile from B's input (after
//        offset: col A_max + 1) element_type = matched element_type
//   6. Deletes the matched conduit.create ops and their aie.shim_dma_allocation
//      ops in both devices.
//   7. Deletes the aiex.dma_configure_task_for / aiex.dma_start_task /
//      aiex.dma_await_task / aiex.dma_free_task ops driving the intermediate
//      shim DMA in both runtime sequences.  The shared-memory path needs no
//      runtime sequence ops.
//
// Prerequisites:
//   --objectfifo-to-conduit [--objectfifo-to-conduit-infer-rates=true]
//   (Must run before this pass; produces conduit.create ops with element_type.)
//
// Run before:
//   --conduit-infer-modes (resolves routing_mode="any" → shared-mem)
//   --conduit-depth-promote (resolves depth=0)
//   --conduit-to-dma
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "ConduitTileInference.h"
#include "DeviceMergeUtils.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITFUSEOPERATORS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helper: resolve the routing_mode of a fused intermediate channel from its
// two endpoints (outCh = producer-side, inCh = consumer-side).
//
// Conflict policy:
//   * Both absent           → empty attr (downstream resolves)
//   * Both equal             → that value
//   * Exactly one set        → the one that's set
//   * Cascade vs anything    → ERROR
//   * SharedMemory vs Circuit→ ERROR
//   * Otherwise              → producer (outCh) wins
//
// On error: emits a diagnostic on outCh and sets `errored = true`; the empty
// attr returned MUST NOT be used (caller signals pass failure).
// ---------------------------------------------------------------------------
static RoutingModeAttr resolveFusedRoutingMode(Create outCh, Create inCh,
                                               bool &errored) {
  errored = false;
  RoutingModeAttr outAttr = outCh.getRoutingModeAttr();
  RoutingModeAttr inAttr = inCh.getRoutingModeAttr();

  // Both absent: leave for downstream resolution.
  if (!outAttr && !inAttr)
    return RoutingModeAttr{};
  // Exactly one set: take the set one.
  if (!outAttr)
    return inAttr;
  if (!inAttr)
    return outAttr;
  // Both set: equal → that value.
  RoutingMode outMode = outAttr.getValue();
  RoutingMode inMode = inAttr.getValue();
  if (outMode == inMode)
    return outAttr;

  auto isCascade = [](RoutingMode m) { return m == RoutingMode::Cascade; };
  auto isShared = [](RoutingMode m) { return m == RoutingMode::SharedMemory; };
  auto isCircuit = [](RoutingMode m) { return m == RoutingMode::Circuit; };

  auto modeName = [](RoutingMode m) -> llvm::StringRef {
    switch (m) {
    case RoutingMode::Circuit:
      return "circuit";
    case RoutingMode::Packet:
      return "packet";
    case RoutingMode::Cascade:
      return "cascade";
    case RoutingMode::Stream:
      return "stream";
    case RoutingMode::SharedMemory:
      return "shared_memory";
    case RoutingMode::DMA:
      return "dma";
    }
    return "?";
  };

  // Cascade vs anything (other than equal cascade, handled above) → ERROR.
  if (isCascade(outMode) || isCascade(inMode)) {
    outCh.emitOpError("conduit-fuse-operators: incompatible routing_mode "
                      "for fused intermediate channel: producer=")
        << modeName(outMode) << ", consumer=" << modeName(inMode)
        << " (cascade cannot be fused with a non-cascade endpoint)";
    errored = true;
    return RoutingModeAttr{};
  }

  // SharedMemory vs Circuit (either direction) → ERROR.
  if ((isShared(outMode) && isCircuit(inMode)) ||
      (isCircuit(outMode) && isShared(inMode))) {
    outCh.emitOpError("conduit-fuse-operators: incompatible routing_mode "
                      "for fused intermediate channel: producer=")
        << modeName(outMode) << ", consumer=" << modeName(inMode)
        << " (shared_memory and circuit cannot be reconciled)";
    errored = true;
    return RoutingModeAttr{};
  }

  // Otherwise: producer wins.
  return outAttr;
}

// ---------------------------------------------------------------------------
// Helper: collect the maximum column index used by any tile in a DeviceOp.
// ---------------------------------------------------------------------------
static int64_t maxColInDevice(AIE::DeviceOp device) {
  int64_t maxCol = -1;
  device.walk([&](AIE::TileOp tile) {
    if (tile.getCol() > maxCol)
      maxCol = tile.getCol();
  });
  return maxCol;
}

// ---------------------------------------------------------------------------
// Helper: offset all tile coordinates in a DeviceOp by colOffset.
// Updates aie.tile(col, row) → aie.tile(col + colOffset, row).
// The TileOp's col attribute is updated in place.
// ---------------------------------------------------------------------------
static void offsetDeviceTiles(AIE::DeviceOp device, int64_t colOffset) {
  if (colOffset == 0)
    return;
  // Collect all TileOps first (avoid iterator invalidation).
  llvm::SmallVector<AIE::TileOp> tiles;
  device.walk([&](AIE::TileOp t) { tiles.push_back(t); });
  mlir::OpBuilder builder(device.getContext());
  for (AIE::TileOp tile : tiles) {
    int64_t newCol = tile.getCol() + colOffset;
    tile->setAttr("col",
                  builder.getI32IntegerAttr(static_cast<int32_t>(newCol)));
  }
  // Note: conduit.create producer_tile/consumer_tiles attrs are no longer
  // updated here — tile coordinates are inferred from TileOp structure
  // via inferAllTiles().
}

// ---------------------------------------------------------------------------
// Helper: extract (col, row) from a tile Value (aie.tile op result).
// Returns {-1, -1} if not a TileOp.
// ---------------------------------------------------------------------------
static std::pair<int64_t, int64_t> extractCoord(mlir::Value tileVal) {
  if (auto tileOp = tileVal.getDefiningOp<AIE::TileOp>())
    return {static_cast<int64_t>(tileOp.getCol()),
            static_cast<int64_t>(tileOp.getRow())};
  return {-1, -1};
}

// ---------------------------------------------------------------------------
// Helper: check if a conduit.create is an operator "output" channel.
// Criterion: no compute consumers and producer tile is a shim (row==0).
// Uses inferred tiles; falls back to producer_tile/consumer_tiles attrs
// for hand-written IR outside aie.core (cascade channels are now
// covered by Source 6 in inferAllTiles).
// ---------------------------------------------------------------------------
static bool isOutputChannel(Create op,
                            const llvm::StringMap<InferredTiles> &inferredMap) {
  std::string name = op.getName().str();
  auto tileIt = inferredMap.find(name);

  // Check no compute consumers.
  bool noComputeConsumers = true;
  if (tileIt != inferredMap.end() && !tileIt->second.consumerTiles.empty())
    noComputeConsumers = false;
  // No inferred consumer tiles — assume no compute consumers.

  // Check consumer is shim (output channels exit to LPDDR5 via shim DMA).
  bool hasShimConsumer =
      tileIt != inferredMap.end() && !tileIt->second.shimConsumerTiles.empty();

  return noComputeConsumers && hasShimConsumer;
}

// ---------------------------------------------------------------------------
// Helper: check if a conduit.create is an operator "input" channel.
// Criterion: producer tile is a shim tile (row == 0).
// Uses inferred tiles; falls back to producer_tile attr for hand-written IR.
// ---------------------------------------------------------------------------
static bool isInputChannel(Create op,
                           const llvm::StringMap<InferredTiles> &inferredMap) {
  std::string name = op.getName().str();
  auto tileIt = inferredMap.find(name);

  if (tileIt != inferredMap.end() && tileIt->second.producerTile) {
    auto [col, row] = extractCoord(tileIt->second.producerTile);
    return row == 0;
  }
  // No inferred producer tile — cannot determine if shim.
  return false;
}

// ---------------------------------------------------------------------------
// Helper: detect whether a conduit channel is a forward-chain endpoint
// ("Pattern E") inside the given device.  Implementation lives in
// DeviceMergeUtils so it can be shared with --conduit-fuse-core-bodies (which
// has the same Step 6 rename invariant and the same scatter/gather hazard).
// ---------------------------------------------------------------------------
using detail::isForwardChainEndpoint;

// ---------------------------------------------------------------------------
// Helper: read the discardable `fusion_index : i32` attribute that Track 3
// convergent fixtures stamp on producer/consumer channels to disambiguate K
// producers fanning into one consumer (e.g. SwiGLU's gate=0, up=1).
// Returns std::nullopt for 1:1 fusion IR (which does not carry fusion_index).
// ---------------------------------------------------------------------------
static std::optional<int64_t> getFusionIndex(Create op) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("fusion_index"))
    return attr.getInt();
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Helper: erase shim_dma_allocation ops for a given conduit name.
// Pass A emits aie.shim_dma_allocation with sym_name = @<name>_shim_alloc,
// so we check for both the raw name and the _shim_alloc suffixed form.
// ---------------------------------------------------------------------------
static void eraseShimAlloc(AIE::DeviceOp device, llvm::StringRef name) {
  std::string shimAllocName = name.str() + "_shim_alloc";
  llvm::SmallVector<mlir::Operation *> toErase;
  device.walk([&](AIE::ShimDMAAllocationOp alloc) {
    llvm::StringRef sym = alloc.getSymName();
    if (sym == name || sym == shimAllocName)
      toErase.push_back(alloc.getOperation());
  });
  for (auto *op : toErase)
    op->erase();
}

// ---------------------------------------------------------------------------
// Helper: erase DMA task ops in a runtime_sequence that reference the given
// conduit symbol name (or the _shim_alloc suffixed form that Pass A emits).
// aiex.dma_configure_task_for @<name>_shim_alloc { ... }
// aiex.dma_start_task(%tok) / aiex.dma_await_task(%tok) /
// aiex.dma_free_task(%tok): users of the task token.
//
// The aiex ops are unregistered; we match by op name string and inspect
// FlatSymbolRefAttr or symbol-use attributes.
// ---------------------------------------------------------------------------
static void eraseRuntimeDMAOpsForName(AIE::DeviceOp device,
                                      llvm::StringRef name) {
  std::string shimAllocName = name.str() + "_shim_alloc";

  // Collect DMA configure ops referencing our channel (either naming form).
  // Pass A emits: "aiex.dma_configure_task_for"() <{alloc =
  // @<name>_shim_alloc}> The symbol is stored as a FlatSymbolRefAttr under the
  // "alloc" property key.
  llvm::SmallVector<mlir::Operation *> configOps;
  device.walk([&](mlir::Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();
    if (opName != "aiex.dma_configure_task_for")
      return;
    // Check "alloc" property (stored as intrinsic property, falls through to
    // getAttrOfType for unregistered ops).
    auto checkSymAttr = [&](mlir::Attribute attr) -> bool {
      if (!attr)
        return false;
      if (auto flat = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr))
        return flat.getValue() == name || flat.getValue() == shimAllocName;
      if (auto sym = mlir::dyn_cast<mlir::SymbolRefAttr>(attr))
        return sym.getRootReference() == name ||
               sym.getRootReference() == shimAllocName;
      return false;
    };
    // Try the "alloc" key (the key used by aiex.dma_configure_task_for).
    // getInherentAttr returns std::optional<mlir::Attribute>; unwrap it.
    if (auto optAttr = op->getInherentAttr("alloc"))
      if (checkSymAttr(*optAttr)) {
        configOps.push_back(op);
        return;
      }
    // Fallback: scan all attrs.
    for (mlir::NamedAttribute na : op->getAttrs())
      if (checkSymAttr(na.getValue())) {
        configOps.push_back(op);
        return;
      }
  });

  // For each configure op, collect its token users (start/await/free) then
  // erase.
  for (mlir::Operation *configOp : configOps) {
    llvm::SmallVector<mlir::Operation *> userOps;
    if (configOp->getNumResults() > 0) {
      for (mlir::Operation *user : configOp->getResult(0).getUsers())
        userOps.push_back(user);
    }
    for (mlir::Operation *user : userOps)
      user->erase();
    configOp->erase();
  }
}

// ---------------------------------------------------------------------------
// Helpers: block-arg-to-channel grouping for runtime sequence reconstruction.
// ---------------------------------------------------------------------------

/// A group of channel names that share the same runtime_sequence block arg.
/// In multi-column operators, multiple per-column channels (e.g., ext_in_0,
/// ext_in_1) map to one host buffer and therefore one block arg.
struct ArgGroup {
  unsigned argIndex;
  llvm::SmallVector<std::string> channelNames;
  int64_t maxExtent = 0; // max(offset[0] + num_elems) across all ops in group
};

/// Build arg groups from a runtime_sequence body's put/get_memref ops.
///
/// Groups put/get_memref ops by their explicit `arg_index` attribute, which
/// --dma-task-to-conduit (FS7 fix) sets from the original aie.dma_bd's
/// BlockArgument index.  arg_index is the authoritative binding between a
/// BD and a host-buffer block arg.
///
/// History: this routine previously used an `offset == 0` heuristic to
/// detect arg-group boundaries on the implicit assumption that BDs sharing
/// a host buffer are emitted contiguously with the first BD at offset 0.
/// Multi-column lowerings violate that assumption — when the column-major
/// BD ordering is `(col0_argA, col0_argB, col0_argC, col1_argA, ...)`
/// every column's first BD hits `offset == 0` and starts a bogus new
/// group, while non-zero-offset column-1+ BDs absorb into the wrong
/// neighbour's group (Bug B / Matrix Row #1 4col_med compile failure).
/// arg_index-driven grouping eliminates that ordering dependency.
///
/// The returned groups are sorted by argIndex so positional indexing into
/// `groups[i]` (used by Step 8c's `computeFullBufferType` and
/// `computeDeadArgs`) matches the layout of the original block arguments.
static llvm::SmallVector<ArgGroup> buildArgGroupsFromSeq(mlir::Block &seqBody) {
  llvm::DenseMap<unsigned, unsigned> argIdxToGroupPos;
  llvm::SmallVector<ArgGroup> groups;
  for (mlir::Operation &op : seqBody) {
    llvm::StringRef opName = op.getName().getStringRef();
    if (opName != "conduit.put_memref" && opName != "conduit.get_memref" &&
        opName != "conduit.put_memref_async" &&
        opName != "conduit.get_memref_async")
      continue;
    auto nameAttr = op.getAttrOfType<mlir::FlatSymbolRefAttr>("name");
    if (!nameAttr)
      continue;
    auto argIdxAttr = op.getAttrOfType<mlir::IntegerAttr>("arg_index");
    if (!argIdxAttr)
      continue; // arg_index is required upstream (--dma-task-to-conduit);
                // absence means we cannot reliably group this op.

    int64_t argIdxSigned = argIdxAttr.getInt();
    if (argIdxSigned < 0)
      continue;
    unsigned argIdx = static_cast<unsigned>(argIdxSigned);

    auto offsetsAttr = op.getAttrOfType<mlir::DenseI64ArrayAttr>("offsets");
    int64_t offset = (offsetsAttr && !offsetsAttr.empty()) ? offsetsAttr[0] : 0;
    auto numElemsAttr = op.getAttrOfType<mlir::IntegerAttr>("num_elems");
    int64_t numElems = numElemsAttr ? numElemsAttr.getInt() : 0;
    int64_t extent = offset + numElems;

    auto it = argIdxToGroupPos.find(argIdx);
    if (it == argIdxToGroupPos.end()) {
      argIdxToGroupPos[argIdx] = static_cast<unsigned>(groups.size());
      groups.push_back({argIdx, {nameAttr.getValue().str()}, extent});
    } else {
      ArgGroup &g = groups[it->second];
      g.channelNames.push_back(nameAttr.getValue().str());
      g.maxExtent = std::max(g.maxExtent, extent);
    }
  }
  // Sort by argIndex so groups[i] corresponds positionally to block-arg i
  // (a precondition for the Step 8c trim's indexed walks).
  llvm::sort(groups, [](const ArgGroup &a, const ArgGroup &b) {
    return a.argIndex < b.argIndex;
  });
  return groups;
}

// ---------------------------------------------------------------------------
// Helpers: sync-group-aware runtime sequence interleaving.
// ---------------------------------------------------------------------------

/// Extract channel name from a conduit runtime sequence op.
/// Returns std::nullopt for non-conduit ops or ops without a name attribute.
static std::optional<std::string>
getConduitRuntimeChannelName(mlir::Operation *op) {
  llvm::StringRef opName = op->getName().getStringRef();
  if (opName != "conduit.put_memref" && opName != "conduit.get_memref" &&
      opName != "conduit.put_memref_async" &&
      opName != "conduit.get_memref_async")
    return std::nullopt;
  auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
  if (!nameAttr)
    return std::nullopt;
  return nameAttr.getValue().str();
}

/// Partitioned runtime sequence ops for sync-group-aware interleaving.
struct PartitionedSeqOps {
  /// Ops for channels that appear exactly once (one-shot, e.g., bias vector).
  llvm::SmallVector<mlir::Operation *> oneShot;
  /// Ops for channels that appear multiple times, grouped into sync-group
  /// chunks.  chunks[i] is the i-th sync group's ops.
  llvm::SmallVector<llvm::SmallVector<mlir::Operation *>> chunks;
};

/// Partition runtime sequence ops into one-shot ops and batched sync-group
/// chunks.
///
/// Chunk detection: walk ops in order; a new occurrence of an already-seen
/// channel name within the current chunk signals a new chunk boundary.
/// Ops without channel names (e.g., wait_all) are appended to the current
/// chunk.
static PartitionedSeqOps
partitionRuntimeOps(llvm::ArrayRef<mlir::Operation *> ops) {
  // 1. Count channel name occurrences across all ops.
  llvm::StringMap<unsigned> nameCounts;
  for (mlir::Operation *op : ops) {
    if (auto name = getConduitRuntimeChannelName(op))
      nameCounts[*name]++;
  }

  // 2. Separate one-shot ops from batched ops, preserving order.
  PartitionedSeqOps result;
  llvm::SmallVector<mlir::Operation *> batchedOps;
  for (mlir::Operation *op : ops) {
    auto name = getConduitRuntimeChannelName(op);
    if (name && nameCounts[*name] == 1) {
      result.oneShot.push_back(op);
    } else {
      batchedOps.push_back(op);
    }
  }

  // 3. Group batched ops into sync-group chunks.
  if (!batchedOps.empty()) {
    result.chunks.push_back({});
    llvm::StringSet<> seenInChunk;
    for (mlir::Operation *op : batchedOps) {
      auto name = getConduitRuntimeChannelName(op);
      if (name) {
        if (!seenInChunk.insert(*name).second) {
          // Name already seen in this chunk → start a new chunk.
          result.chunks.push_back({});
          seenInChunk.clear();
          seenInChunk.insert(*name);
        }
      }
      result.chunks.back().push_back(op);
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Main pass struct.
// ---------------------------------------------------------------------------
struct ConduitFuseOperatorsPass
    : public impl::ConduitFuseOperatorsBase<ConduitFuseOperatorsPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();
    mlir::OpBuilder builder(ctx);

    // Infer tile coordinates from IR structure (before any offsetting).
    auto inferredMap = inferAllTiles(module);

    // Collect all DeviceOps in module order.
    llvm::SmallVector<AIE::DeviceOp> devices;
    module.walk([&](AIE::DeviceOp dev) { devices.push_back(dev); });

    if (devices.size() < 2) {
      // Nothing to fuse.
      return;
    }

    int fuseCount = 0;

    // ---------------------------------------------------------------------
    // Track 3 pre-scan: detect Q2 (mixed convergent + 1:1 fusion_groups on
    // the same consumer device) and pre-allocate fused-channel names for
    // convergent groups so that consumer-side IR ends up with K stable
    // `fused_intermediate_K` names where K = `fusion_index` of each
    // producer.  Q5 placement decisions stay inside the per-pair loop;
    // this pre-scan only diagnoses + reserves names.
    //
    // A "convergent" fusion_group is one whose fusion_group symbol appears
    // on Create ops in >= 3 distinct devices (K producers + 1 consumer,
    // K >= 2).  A "1:1" fusion_group appears on exactly 2 devices (one
    // producer + one consumer).  A device that participates in BOTH a
    // convergent group AND a 1:1 group is rejected per the locked Track 3
    // design (CLAUDE.md USER-LOCKED 2026-04-26 Q2): the placement /
    // depth-promote interaction for that composition is not yet decided.
    // ---------------------------------------------------------------------
    struct FgInfo {
      // Distinct devices that contain at least one Create with this fg,
      // in module order.
      llvm::SmallVector<AIE::DeviceOp> devices;
      llvm::DenseSet<mlir::Operation *> deviceSet;
      // fusion_index values seen on consumer-side (input) channels.
      llvm::DenseSet<int64_t> consumerIndices;
    };
    llvm::StringMap<FgInfo> fgInfo;
    for (AIE::DeviceOp d : devices) {
      llvm::StringSet<> fgsSeenInDev;
      d.walk([&](Create op) {
        auto fg = op.getFusionGroup();
        if (!fg || fg->empty())
          return;
        llvm::StringRef fgKey = *fg;
        FgInfo &info = fgInfo[fgKey];
        if (fgsSeenInDev.insert(fgKey).second) {
          info.devices.push_back(d);
          info.deviceSet.insert(d.getOperation());
        }
        if (isInputChannel(op, inferredMap)) {
          if (auto idx = getFusionIndex(op))
            info.consumerIndices.insert(*idx);
        }
      });
    }

    // Q2 detection: any device participating in BOTH a convergent fg
    // (>=3 devices) AND a 1:1 fg (==2 devices) is rejected.
    for (AIE::DeviceOp d : devices) {
      llvm::SmallVector<llvm::StringRef> convergentFgs, oneToOneFgs;
      for (auto &kv : fgInfo) {
        if (!kv.second.deviceSet.contains(d.getOperation()))
          continue;
        if (kv.second.devices.size() >= 3)
          convergentFgs.push_back(kv.first());
        else if (kv.second.devices.size() == 2)
          oneToOneFgs.push_back(kv.first());
      }
      if (!convergentFgs.empty() && !oneToOneFgs.empty()) {
        llvm::sort(convergentFgs);
        llvm::sort(oneToOneFgs);
        d.emitError(
            "conduit-fuse-operators: consumer device participates in both "
            "convergent fusion_group \"")
            << convergentFgs.front() << "\" and 1:1 fusion_group \""
            << oneToOneFgs.front()
            << "\"; mixed convergent + 1:1 fusion is out of scope";
        signalPassFailure();
        return;
      }
    }

    // Pre-allocate convergent names: for each convergent fg (sorted by
    // fg name for determinism), reserve `fused_intermediate_<base+i>`
    // for each consumer-side fusion_index value in ascending order, and
    // advance fuseCount past all reservations so subsequent 1:1 fusions
    // use higher numbers (no collision).
    llvm::StringMap<llvm::DenseMap<int64_t, std::string>> convergentNameMap;
    {
      llvm::SmallVector<llvm::StringRef> convFgs;
      for (auto &kv : fgInfo)
        if (kv.second.devices.size() >= 3)
          convFgs.push_back(kv.first());
      llvm::sort(convFgs);
      for (llvm::StringRef fg : convFgs) {
        llvm::SmallVector<int64_t> indices(fgInfo[fg].consumerIndices.begin(),
                                           fgInfo[fg].consumerIndices.end());
        llvm::sort(indices);
        for (int64_t idx : indices) {
          convergentNameMap[fg][idx] =
              "fused_intermediate_" + std::to_string(fuseCount++);
        }
      }
    }

    // Build convergentDevices set ONCE.  A device is "convergent" if it
    // participates in any convergent fusion_group (fg with >=3 devices = K
    // producers + 1 consumer, K>=2).  Used by:
    //   1. Element-type fallback gate at L702 — element-type matching is
    //      meant for IR with no fusion_group annotations at all; convergent
    //      participants always carry fg, so an empty fg-match for them
    //      means "no valid pair" (e.g., two co-producers like devGate/devUp
    //      in SwiGLU), not "no fg at all".  Without the gate, the fallback
    //      mis-pairs co-producers (e.g. @inter_gate ↔ @ext_in_up because
    //      they share element_type), destroying surviving channels and
    //      collapsing K fused intermediates to 1.
    //   2. classifyConvergent helper to redirect (producer, consumer)
    //      pairing in the device-pair driver — the natural module order
    //      [p0, p1, ..., pK-1, consumer] is NOT producer-then-consumer
    //      adjacency for K>=2, so adjacency-based pairing is wrong.
    llvm::DenseSet<mlir::Operation *> convergentDevices;
    for (auto &kv : fgInfo)
      if (kv.second.devices.size() >= 3)
        for (AIE::DeviceOp d : kv.second.devices)
          convergentDevices.insert(d.getOperation());

    // classifyConvergent: determine devA's role in any convergent fg.
    // Returns a tri-state via the out-params:
    //   0 → not in any convergent fg (fall through to 1:1 / element-type
    //       pairing with devices[i+1]).
    //   1 → convergent CONSUMER (the device whose Creates carry fg +
    //       fusion_index on input-side channels).  Skip this iteration —
    //       producers will pair with the consumer when the driver reaches
    //       them.
    //   2 → convergent PRODUCER.  outConsumer / outConsumerIdx point at
    //       the still-live consumer device (which may not be at i+1 in the
    //       devices vector — for [p0, p1, consumer] the consumer is at
    //       index i+2 when i=0).
    // Note: after a convergent merge erases the consumer device and
    // accumulates into the producer, the absorber inherits the consumer's
    // unfused convergent input channels and is therefore re-classified as
    // a CONSUMER on the next iteration — which correctly causes us to
    // skip it and process the next producer (which then pairs with the
    // absorber).  This implements the iterative pairwise N-way reduction.
    auto classifyConvergent = [&](AIE::DeviceOp devA, size_t devAIdx,
                                  AIE::DeviceOp &outConsumer,
                                  size_t &outConsumerIdx) -> int {
      for (auto &kv : fgInfo) {
        if (kv.second.devices.size() < 3)
          continue;
        if (!kv.second.deviceSet.contains(devA.getOperation()))
          continue;
        // Is devA the consumer for this fg?  Consumer = device whose
        // Creates carry fg on input-side channels with fusion_index.
        bool devAIsConsumer = false;
        devA.walk([&](Create op) {
          auto opFG = op.getFusionGroup();
          if (!opFG || *opFG != kv.first())
            return;
          if (isInputChannel(op, inferredMap) && getFusionIndex(op))
            devAIsConsumer = true;
        });
        if (devAIsConsumer)
          return 1;
        // devA is a producer in this fg.  Find the consumer in the live
        // devices vector (the original consumer may have been absorbed
        // into a prior producer by an earlier convergent merge — in that
        // case the absorber now carries the consumer-side input channels
        // and is the live consumer for the next pairing).
        for (size_t j = 0; j < devices.size(); ++j) {
          if (j == devAIdx)
            continue;
          AIE::DeviceOp cand = devices[j];
          bool candIsConsumer = false;
          cand.walk([&](Create op) {
            auto opFG = op.getFusionGroup();
            if (!opFG || *opFG != kv.first())
              return;
            if (isInputChannel(op, inferredMap) && getFusionIndex(op))
              candIsConsumer = true;
          });
          if (candIsConsumer) {
            outConsumer = cand;
            outConsumerIdx = j;
            return 2;
          }
        }
        // No live consumer found for this fg — convergent processing
        // already finished (all K producers paired).  Fall through to
        // try other fgs (devA could in principle be in multiple fgs,
        // though Q2 disallows mixing convergent+1:1 on the same device).
      }
      return 0;
    };

    // Process device pairs (A, B).  Iteration is convergent-aware: when
    // devA is a convergent producer, devB is the convergent consumer
    // (which may not be adjacent); when devA is a convergent consumer,
    // skip (producers will pair with us).  Otherwise (1:1 fg or
    // element-type), devB = devices[i+1].
    for (size_t i = 0; i + 1 < devices.size(); ++i) {
      AIE::DeviceOp devA = devices[i];
      AIE::DeviceOp devB = devices[i + 1];
      size_t devBIdx = i + 1;

      // Convergent-aware devB selection.
      {
        AIE::DeviceOp convConsumer;
        size_t convConsumerIdx = 0;
        int role = classifyConvergent(devA, i, convConsumer, convConsumerIdx);
        if (role == 1) {
          // devA is a convergent consumer.  In the iterative pairwise
          // reduction, after the first pair-merge the absorber inherits the
          // consumer's unfused convergent input channels — re-classifying
          // it as a CONSUMER on the next iteration.  But the for-loop
          // index `i` may not advance past the absorber's slot on its own
          // (the absorber typically sits at devices[0] with the remaining
          // producers at higher indices), so simply `continue`-ing would
          // strand the remaining producers.
          //
          // Look for a still-live convergent producer for some convergent
          // fg containing devA at any other index j.  If found, swap
          // roles for this iteration: set devA = devices[j] (producer)
          // and devB = original devA (consumer at i).  Step 8 will then
          // absorb the consumer-absorber INTO the producer, leaving the
          // producer as the new accumulator.  Each subsequent iteration
          // peels off one more producer until all K have been merged.
          //
          // If no remaining producer is found, the convergent group is
          // fully reduced — `continue` past devA.
          AIE::DeviceOp prodCand;
          size_t prodCandIdx = 0;
          bool foundProd = false;
          for (auto &kv : fgInfo) {
            if (kv.second.devices.size() < 3)
              continue;
            if (!kv.second.deviceSet.contains(devA.getOperation()))
              continue;
            for (size_t j = 0; j < devices.size(); ++j) {
              if (j == i)
                continue;
              AIE::DeviceOp cand = devices[j];
              bool candIsProducer = false;
              cand.walk([&](Create op) {
                auto opFG = op.getFusionGroup();
                if (!opFG || *opFG != kv.first())
                  return;
                // Producer = output-side fg-tagged Create with fusion_index.
                if (isOutputChannel(op, inferredMap) && getFusionIndex(op))
                  candIsProducer = true;
              });
              if (candIsProducer) {
                prodCand = cand;
                prodCandIdx = j;
                foundProd = true;
                break;
              }
            }
            if (foundProd)
              break;
          }
          if (!foundProd)
            continue;
          // Swap: process this iteration with the producer as devA and the
          // consumer-absorber as devB.  Step 8 will erase devB at devBIdx.
          devB = devA;
          devBIdx = i;
          devA = prodCand;
          (void)prodCandIdx; // referenced only for clarity above
        } else if (role == 2) {
          devB = convConsumer;
          devBIdx = convConsumerIdx;
        }
      }

      // --- Step 1: Find output channels in device A. ---
      llvm::SmallVector<Create> outputChannels;
      devA.walk([&](Create op) {
        if (isOutputChannel(op, inferredMap))
          outputChannels.push_back(op);
      });

      // --- Step 2: Find input channels in device B. ---
      llvm::SmallVector<Create> inputChannels;
      devB.walk([&](Create op) {
        if (isInputChannel(op, inferredMap))
          inputChannels.push_back(op);
      });

      if (outputChannels.empty() || inputChannels.empty())
        continue;

      // --- Step 3: Match by fusion_group attribute. ---
      // Channels with matching fusion_group values are paired for fusion.
      // Falls back to element_type matching for IR without fusion_group attrs.
      // Each input channel is consumed at most once (1:1 pairing).
      //
      // Pattern E guard: a channel that participates in a `conduit.scatter`
      // / `conduit.gather` is a forward-chain endpoint (e.g., from
      // `aie.objectfifo.link`).  Step 6's rename walk only updates ops with
      // a `name` attr, so erasing/renaming such a channel would leave the
      // scatter/gather op pointing at a dangling FlatSymbolRefAttr.  Skip
      // with a remark — there is no compute body on this side to fuse.
      llvm::SmallVector<std::pair<Create, Create>> matched;
      {
        llvm::DenseSet<mlir::Operation *> consumedInputs;
        for (Create outCh : outputChannels) {
          auto outFG = outCh.getFusionGroup();
          if (!outFG || outFG->empty())
            continue;
          if (isForwardChainEndpoint(devA, outCh.getName())) {
            outCh.emitRemark("conduit-fuse-operators: skipping fusion_group "
                             "match for output channel @")
                << outCh.getName()
                << " — forward-chain / link-only endpoint (Pattern E); "
                   "scatter/gather references cannot be safely renamed";
            continue;
          }
          auto outIdx = getFusionIndex(outCh);
          for (Create inCh : inputChannels) {
            if (consumedInputs.contains(inCh.getOperation()))
              continue;
            auto inFG = inCh.getFusionGroup();
            if (!inFG || *outFG != *inFG)
              continue;
            // Track 3 convergent disambiguation: when both sides expose
            // `fusion_index`, require equality so each producer pairs with
            // the matching consumer-side input.  1:1 fusion IR carries no
            // index and falls through to the existing fg-only match.
            auto inIdx = getFusionIndex(inCh);
            if (outIdx && inIdx && *outIdx != *inIdx)
              continue;
            if (isForwardChainEndpoint(devB, inCh.getName())) {
              inCh.emitRemark("conduit-fuse-operators: skipping "
                              "fusion_group match for input channel @")
                  << inCh.getName()
                  << " — forward-chain / link-only endpoint (Pattern E); "
                     "scatter/gather references cannot be safely renamed";
              continue;
            }
            matched.push_back({outCh, inCh});
            consumedInputs.insert(inCh.getOperation());
            break;
          }
        }
      }
      // Fallback: match by element_type if no fusion_group attrs found.
      // Same Pattern E guard applies — element_type collision between a
      // forward chain and a Pattern A neighbor would trigger the identical
      // dangling-symbol bug.
      //
      // Convergent-fg gate: skip element-type fallback when EITHER device
      // participates in a convergent fusion_group.  Convergent participants
      // always carry fg, so an empty fg-match for them means "no valid pair
      // here" (e.g., two co-producers like devGate / devUp in the SwiGLU
      // fixture both have outputs but no matching inputs in each other),
      // NOT "this IR has no fusion_group annotations".  Without the gate,
      // the fallback mis-pairs by element_type alone — e.g.,
      // (@inter_gate, @ext_in_up) — destroying the surviving @ext_in_up
      // and burning a `fused_intermediate_N` slot, leaving K-1 fused
      // intermediates instead of K.
      if (matched.empty() &&
          !convergentDevices.contains(devA.getOperation()) &&
          !convergentDevices.contains(devB.getOperation())) {
        llvm::DenseSet<mlir::Operation *> consumedInputs;
        for (Create outCh : outputChannels) {
          if (isForwardChainEndpoint(devA, outCh.getName()))
            continue;
          mlir::Type outET = outCh.getElementType();
          for (Create inCh : inputChannels) {
            if (consumedInputs.contains(inCh.getOperation()))
              continue;
            if (isForwardChainEndpoint(devB, inCh.getName()))
              continue;
            mlir::Type inET = inCh.getElementType();
            if (outET == inET) {
              matched.push_back({outCh, inCh});
              consumedInputs.insert(inCh.getOperation());
              break;
            }
          }
        }
      }

      if (matched.empty()) {
        module.emitWarning(
            "conduit-fuse-operators: no matching channel pair found between "
            "device " +
            std::to_string(i) + " and device " + std::to_string(devBIdx) +
            " by fusion_group or element_type; skipping");
        continue;
      }

      // --- Step 3.5: Record original block arg types and channel→arg
      // mappings.
      //
      // After --dma-task-to-conduit, runtime_sequence block args are SSA-dead
      // but retain the correct full-buffer types (e.g. memref<256xbf16>).
      // Record these now — before any erasure — so Step 8c can reconstruct
      // the merged block args using full-buffer types instead of per-tile
      // num_elems.
      llvm::SmallVector<mlir::Type> origTypesA, origTypesB;
      llvm::SmallVector<ArgGroup> argGroupsA, argGroupsB;
      llvm::StringSet<> erasedChannelsA, erasedChannelsB;
      {
        for (mlir::Operation &op : devA.getBodyRegion().front()) {
          if (op.getName().getStringRef() == "aie.runtime_sequence" &&
              op.getNumRegions() > 0) {
            mlir::Block &body = op.getRegion(0).front();
            for (auto arg : body.getArguments())
              origTypesA.push_back(arg.getType());
            argGroupsA = buildArgGroupsFromSeq(body);
            break;
          }
        }
        for (mlir::Operation &op : devB.getBodyRegion().front()) {
          if (op.getName().getStringRef() == "aie.runtime_sequence" &&
              op.getNumRegions() > 0) {
            mlir::Block &body = op.getRegion(0).front();
            for (auto arg : body.getArguments())
              origTypesB.push_back(arg.getType());
            argGroupsB = buildArgGroupsFromSeq(body);
            break;
          }
        }
      }

      // --- Step 4: Compute column offset for device B. ---
      int64_t colMaxA = maxColInDevice(devA);
      int64_t colOffset = colMaxA + 1; // device B's col=0 → col=colMaxA+1
      offsetDeviceTiles(devB, colOffset);

      // --- Step 5: Emit module-level conduit.create for each matched pair. ---
      // Insert before the first DeviceOp so module-level conduits are easy to
      // find in Pass C's module-scope walk.
      builder.setInsertionPoint(devA);

      for (auto [outCh, inCh] : matched) {
        std::string outName = outCh.getName().str();
        std::string inName = inCh.getName().str();

        // Track erased channels for Step 8c block arg reconstruction.
        erasedChannelsA.insert(outName);
        erasedChannelsB.insert(inName);

        // Pick the fused channel name.  For a convergent merge (K producers
        // → 1 consumer) Track 3's pre-scan reserved K stable names keyed by
        // (fusion_group, fusion_index), so each producer's `outName` →
        // `fused_intermediate_K` where K = fusion_index of that producer
        // (Q4 of the locked design).  1:1 fusion has no fusion_index and
        // falls through to the running counter.
        std::string fusedName;
        if (auto outFGOpt = outCh.getFusionGroup()) {
          if (auto outIdxOpt = getFusionIndex(outCh)) {
            auto fgIt = convergentNameMap.find(*outFGOpt);
            if (fgIt != convergentNameMap.end()) {
              auto idxIt = fgIt->second.find(*outIdxOpt);
              if (idxIt != fgIt->second.end())
                fusedName = idxIt->second;
            }
          }
        }
        if (fusedName.empty())
          fusedName = "fused_intermediate_" + std::to_string(fuseCount++);

        // Gather attributes for the fused conduit.create.
        // Note: producer_tile/consumer_tiles are no longer emitted —
        // tile coordinates are inferred from IR structure via inferAllTiles()
        // after Step 6 renames channel references in the core bodies.

        // depth = 2: standard double-buffering (works for both shared-memory
        // and circuit DMA routing).
        mlir::IntegerAttr depthAttr = builder.getI64IntegerAttr(2);

        // element_type: copy from outCh (required in redesign 2).
        // The Create builder takes mlir::Type directly (not TypeAttr).
        mlir::Type elemType = outCh.getElementType();

        // producer_rates / consumer_rates: propagate if present.
        mlir::DenseI64ArrayAttr producerRatesAttr, consumerRatesAttr;
        if (auto attr =
                outCh->getAttrOfType<mlir::DenseI64ArrayAttr>("producer_rates"))
          producerRatesAttr = attr;
        if (auto attr =
                inCh->getAttrOfType<mlir::DenseI64ArrayAttr>("consumer_rates"))
          consumerRatesAttr = attr;

        // routing_mode: resolve from the two endpoints' attrs via the
        // conflict policy in resolveFusedRoutingMode.  When both endpoints
        // are absent the result is empty and --conduit-infer-modes resolves
        // it (adjacent tiles → shared_memory; non-adjacent → circuit).
        bool routingErrored = false;
        RoutingModeAttr routingModeAttr =
            resolveFusedRoutingMode(outCh, inCh, routingErrored);
        if (routingErrored) {
          signalPassFailure();
          return;
        }

        // Emit conduit.create INSIDE devA (not at module scope) so Pass C
        // sees it as a normal intra-device channel after the device merge.
        // Insert after the last existing conduit.create in devA.
        {
          mlir::Operation *insertPt = nullptr;
          for (mlir::Operation &op : devA.getBodyRegion().front())
            if (mlir::isa<Create>(op))
              insertPt = &op;
          if (insertPt) {
            builder.setInsertionPointAfter(insertPt);
          } else {
            auto &front = devA.getBodyRegion().front();
            if (front.mightHaveTerminator()) {
              if (mlir::Operation *term = front.getTerminator())
                builder.setInsertionPoint(term);
              else
                builder.setInsertionPointToEnd(&front);
            } else {
              builder.setInsertionPointToEnd(&front);
            }
          }
        }
        builder.create<Create>(devA.getLoc(),
                               mlir::StringAttr::get(ctx, fusedName),
                               /*element_type=*/elemType,
                               /*depth=*/depthAttr,
                               /*routing_mode=*/routingModeAttr,
                               /*sync_mode=*/SyncModeAttr{},
                               /*producer_rates=*/producerRatesAttr,
                               /*consumer_rates=*/consumerRatesAttr,
                               /*fusion_group=*/mlir::StringAttr{},
                               /*bd_repeat=*/mlir::IntegerAttr{},
                               /*dma_repeat=*/mlir::IntegerAttr{},
                               /*producer_dimensions=*/nullptr,
                               /*consumer_dimensions=*/nullptr);

        // --- Step 6: Rename channel references in core bodies. ---
        // The GEMV core has conduit.acquire/release on @outName (the old output
        // channel name).  Rename these to the fused channel name so Pass C can
        // resolve subview_access to the fused channel's allocated buffers.
        // Similarly for the ReLU core's @inName references.
        // (outName and inName already captured above for inference lookup.)
        auto renameChannelRefs = [&](AIE::DeviceOp device,
                                     llvm::StringRef oldName) {
          device.walk([&](mlir::Operation *op) {
            // Rename any op that has a "name" attribute matching oldName.
            // This covers conduit.acquire, conduit.release,
            // conduit.subview_access, conduit.acquire_async,
            // conduit.release_async.
            if (auto nameAttr =
                    op->getAttrOfType<mlir::FlatSymbolRefAttr>("name")) {
              if (nameAttr.getValue() == oldName)
                op->setAttr("name",
                            mlir::FlatSymbolRefAttr::get(ctx, fusedName));
            }
          });
        };

        renameChannelRefs(devA, outName);
        renameChannelRefs(devB, inName);

        // --- Step 6b: Erase dead put_memref/get_memref for the fused channel.
        // After renaming, the runtime sequences contain put_memref/get_memref
        // ops referencing @fused_intermediate_N.  These drove the shim DMA for
        // the intermediate LPDDR5 crossing; after spatial fusion the
        // intermediate lives in shared tile memory and needs no shim DMA.
        // Erase them now so they are not cloned into the merged sequence.
        //
        // For the `_async` variants, the op's token result may be consumed
        // by `conduit.wait_all` ops.  When fusion eliminates the
        // intermediate DMA, the wait_all on its token is also no longer
        // semantically meaningful (no separate transfer = no sync point),
        // so we erase those wait_alls first.  If a wait_all has a single
        // operand pointing at the to-be-erased token, erase the wait_all
        // entirely; if it has multiple operands, drop just the dead one
        // by rebuilding the wait_all with the surviving operands.
        auto eraseFusedMemrefOps = [&](AIE::DeviceOp device) {
          llvm::SmallVector<mlir::Operation *> toErase;
          llvm::SmallPtrSet<mlir::Operation *, 8> targetAsyncOps;
          device.walk([&](mlir::Operation *op) {
            llvm::StringRef opName = op->getName().getStringRef();
            if (opName != "conduit.put_memref" &&
                opName != "conduit.get_memref" &&
                opName != "conduit.put_memref_async" &&
                opName != "conduit.get_memref_async")
              return;
            auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
            if (nameAttr && nameAttr.getValue() == fusedName) {
              toErase.push_back(op);
              if (opName == "conduit.put_memref_async" ||
                  opName == "conduit.get_memref_async")
                targetAsyncOps.insert(op);
            }
          });
          // Pre-pass: drop dead token operands from wait_all consumers,
          // erasing the wait_all entirely if no operands survive.
          if (!targetAsyncOps.empty()) {
            llvm::SmallVector<WaitAll> waitAllsToErase;
            llvm::SmallVector<
                std::pair<WaitAll, llvm::SmallVector<mlir::Value, 4>>>
                waitAllsToShrink;
            device.walk([&](WaitAll wa) {
              llvm::SmallVector<mlir::Value, 4> survivors;
              bool hadDead = false;
              for (mlir::Value tok : wa.getTokens()) {
                mlir::Operation *defOp = tok.getDefiningOp();
                if (defOp && targetAsyncOps.contains(defOp)) {
                  hadDead = true;
                  continue;
                }
                survivors.push_back(tok);
              }
              if (!hadDead)
                return;
              if (survivors.empty())
                waitAllsToErase.push_back(wa);
              else
                waitAllsToShrink.push_back({wa, std::move(survivors)});
            });
            for (auto &[wa, survivors] : waitAllsToShrink) {
              mlir::OpBuilder builder(wa);
              auto rebuilt = WaitAll::create(builder, wa.getLoc(), survivors);
              if (auto tokenAttr = wa.getTokenAttr())
                rebuilt.setTokenAttr(tokenAttr);
              wa.erase();
            }
            for (WaitAll wa : waitAllsToErase)
              wa.erase();
          }
          for (auto *op : toErase)
            op->erase();
        };
        eraseFusedMemrefOps(devA);
        eraseFusedMemrefOps(devB);

        eraseShimAlloc(devA, outName);
        eraseShimAlloc(devB, inName);

        // --- Step 7: Delete runtime DMA ops for intermediate channels. ---
        eraseRuntimeDMAOpsForName(devA, outName);
        eraseRuntimeDMAOpsForName(devB, inName);

        outCh->erase();
        inCh->erase();
      }

      // --- Step 7b: Renumber shim DMA channels sequentially after erasure. ---
      // After fused shim_dma_allocation ops are erased, surviving allocations
      // may have non-contiguous channel numbers (e.g., channel 1 with channel 0
      // erased). Pass C expects channels numbered from 0.  Renumber each
      // (tile, direction) group sequentially.
      auto renumberShimAllocs = [&](AIE::DeviceOp device) {
        // Group by (tile SSA value, direction).
        using Key = std::pair<mlir::Value, int>;
        llvm::DenseMap<Key, llvm::SmallVector<AIE::ShimDMAAllocationOp>> groups;
        device.walk([&](AIE::ShimDMAAllocationOp alloc) {
          Key k = {alloc.getTile(), static_cast<int>(alloc.getChannelDir())};
          groups[k].push_back(alloc);
        });
        for (auto &[key, allocs] : groups) {
          // Sort by original channel index to preserve relative order.
          llvm::sort(allocs, [](AIE::ShimDMAAllocationOp a,
                                AIE::ShimDMAAllocationOp b) {
            return a.getChannelIndex() < b.getChannelIndex();
          });
          for (unsigned idx = 0; idx < allocs.size(); ++idx) {
            if (allocs[idx].getChannelIndex() != static_cast<int64_t>(idx))
              allocs[idx].setChannelIndex(static_cast<int64_t>(idx));
          }
        }
      };
      renumberShimAllocs(devA);
      renumberShimAllocs(devB);

      // --- Step 8: Merge device B into device A by physically moving ops. ---
      //
      // aie.device has IsolatedFromAbove semantics. SSA values (tile ops,
      // locks, buffers) defined in devB cannot be used in devA and vice versa.
      // aie.flow and conduit's DMA BD chains also reference tiles from both
      // devices. Therefore, both devices must be merged into one so that Pass C
      // can emit aie.flow, aie.lock, and aie.buffer referencing all tiles.
      //
      // We use op->remove() + builder.insert(op) to physically move each op
      // from bodyB into bodyA (before bodyA's terminator). Moving — not cloning
      // — preserves all SSA values: the TileOp result %tile_1_2 is the same
      // Value after the move as before, so all users (CoreOp, LockOp, ...) in
      // devB that were moved alongside it continue to reference the correct
      // definition. No IRMapping is needed.
      //
      // aie.runtime_sequence: both devices may each have one. Rather than
      // moving a second conflicting sequence into devA, we splice devB's
      // sequence body into devA's existing sequence before its terminator.
      //
      // After the move, bodyB contains only the aie.end terminator. devB is
      // then erased. Because devB's body is empty of non-terminator ops (and
      // no live symbol-table entries remain), the erase is safe and does not
      // corrupt the module symbol table.
      {
        mlir::Block &bodyA = devA.getBodyRegion().front();
        mlir::Block &bodyB = devB.getBodyRegion().front();

        // Find devA's runtime_sequence (may not exist).
        mlir::Operation *seqA = detail::findRuntimeSequence(bodyA);

        // Phase 1: move all non-sequence, non-terminator ops from bodyB into
        // bodyA. This establishes the tile/shim SSA values in devA's scope so
        // that the sequence body ops (which use those tile values) can be moved
        // safely in Phase 2.
        llvm::SmallVector<mlir::Operation *> seqOps =
            detail::movePhase1NonSequenceOps(bodyA, bodyB);

        // Phase 2: merge devB's runtime_sequence(s) into devA's sequence.
        //
        // aie.runtime_sequence is a Symbol with block-argument-typed host
        // memref parameters. Each block argument corresponds to a host-side
        // buffer (group_id N).  When devA and devB have different signatures
        // (e.g., GEMV takes (A, x, C) and ReLU takes (y)), we merge them into
        // a single combined sequence whose signature is the concatenation of
        // devA's args plus devB's args.
        //
        // Concretely:
        //   seqA body args: %arg0: memref<16384xbf16>, %arg1: memref<128xbf16>
        //   seqB body args: %arg0: memref<128xbf16>
        //   merged args:    %arg0: memref<16384xbf16>, %arg1: memref<128xbf16>,
        //                   %arg2: memref<128xbf16>   ← seqB's arg0 remapped
        //
        // We clone seqB's body ops into seqA using an IRMapping that maps
        // seqB's block arguments to new block arguments appended to seqA's
        // block.  The tile/shim SSA values referenced by seqB's DMA ops are
        // already in bodyA (moved in Phase 1), so no additional mapping is
        // needed for those.
        {
          for (mlir::Operation *seqB : seqOps) {
            if (seqA && seqA->getNumRegions() > 0 &&
                seqB->getNumRegions() > 0) {
              mlir::Block &seqBodyA = seqA->getRegion(0).front();
              mlir::Block &seqBodyB = seqB->getRegion(0).front();

              // Append seqB's block args to seqA, building the IRMapping.
              mlir::IRMapping argMapping;
              for (mlir::BlockArgument arg : seqBodyB.getArguments()) {
                mlir::BlockArgument newArg =
                    seqBodyA.addArgument(arg.getType(), arg.getLoc());
                argMapping.map(arg, newArg);
              }

              // Clone each op from seqBodyB into seqBodyA.
              // The IRMapping covers seqB's block args; tile/shim values need
              // no mapping because they were physically moved to bodyA in
              // Phase 1 and are the same SSA Value objects.
              // --- Sync-group-aware interleaving ---
              // Instead of simply appending all of seqB's ops after seqA's
              // ops (which is only correct for single-sync-group operators),
              // we partition both sequences into one-shot ops and batched
              // chunks, then interleave the chunks to preserve per-batch
              // correctness for multi-batch operators.
              //
              // Rule: one_shot_A + one_shot_B
              //       + [chunk_A[0] + chunk_B[0]]
              //       + [chunk_A[1] + chunk_B[1]] + ...

              // Collect surviving ops from both sequences.
              llvm::SmallVector<mlir::Operation *> opsA, opsB;
              for (mlir::Operation &op : seqBodyA) {
                if (!op.hasTrait<mlir::OpTrait::IsTerminator>())
                  opsA.push_back(&op);
              }
              for (mlir::Operation &op : seqBodyB) {
                if (!op.hasTrait<mlir::OpTrait::IsTerminator>())
                  opsB.push_back(&op);
              }

              // Partition into one-shot ops and batched chunks.
              PartitionedSeqOps partA = partitionRuntimeOps(opsA);
              PartitionedSeqOps partB = partitionRuntimeOps(opsB);

              // Check chunk count compatibility.
              if (!partA.chunks.empty() && !partB.chunks.empty() &&
                  partA.chunks.size() != partB.chunks.size()) {
                seqA->emitError()
                    << "conduit-fuse-operators: incompatible batch counts "
                       "in runtime sequences: device A has "
                    << partA.chunks.size() << " sync groups, device B has "
                    << partB.chunks.size();
                signalPassFailure();
                return;
              }

              // Detach all non-terminator ops from seqBodyA.
              for (mlir::Operation *op : opsA)
                op->remove();

              // Set insertion point before the terminator (or end of block).
              mlir::OpBuilder seqBuilder(ctx);
              if (seqBodyA.mightHaveTerminator()) {
                if (mlir::Operation *term = seqBodyA.getTerminator())
                  seqBuilder.setInsertionPoint(term);
                else
                  seqBuilder.setInsertionPointToEnd(&seqBodyA);
              } else {
                seqBuilder.setInsertionPointToEnd(&seqBodyA);
              }

              // Helper: tag a cloned put/get_memref op as B-side so the
              // post-trim arg_index reprojection (Step 8c followup) can
              // distinguish A-side originals (untagged) from B-side clones.
              // See `reprojectMemrefArgIndex` below.
              auto tagBOriginIfMemref = [&](mlir::Operation *cloned) {
                llvm::StringRef nm = cloned->getName().getStringRef();
                if (nm == "conduit.put_memref" || nm == "conduit.get_memref" ||
                    nm == "conduit.put_memref_async" ||
                    nm == "conduit.get_memref_async")
                  cloned->setAttr("_origin_device",
                                  mlir::StringAttr::get(ctx, "B"));
              };

              // Re-insert in interleaved order.
              // 1. One-shot ops from A (re-insert originals).
              for (mlir::Operation *op : partA.oneShot)
                seqBuilder.insert(op);
              // 2. One-shot ops from B (clone with arg mapping).
              for (mlir::Operation *op : partB.oneShot)
                tagBOriginIfMemref(seqBuilder.clone(*op, argMapping));
              // 3. Interleave batched chunks.
              size_t numChunks = partA.chunks.size();
              if (partB.chunks.size() > numChunks)
                numChunks = partB.chunks.size();
              for (size_t c = 0; c < numChunks; ++c) {
                if (c < partA.chunks.size())
                  for (mlir::Operation *op : partA.chunks[c])
                    seqBuilder.insert(op);
                if (c < partB.chunks.size())
                  for (mlir::Operation *op : partB.chunks[c])
                    tagBOriginIfMemref(seqBuilder.clone(*op, argMapping));
              }
              // seqB's body is now represented in seqA. seqB itself remains
              // in devB and will be erased with devB below.
            } else if (!seqA) {
              // devA has no sequence yet — move devB's as-is.
              detail::moveToEndOfDeviceBody(seqB, bodyA);
              seqA = seqB;
            }
          }
        }

        // bodyB is now empty except its aie.end terminator. All SSA values
        // defined in bodyB have been moved to bodyA, so devB->erase() will
        // not encounter live-use violations.
        //
        // Before erasing devB, rewrite host-orchestrator references: any
        // module-level `aiex.configure @<devB.sym_name> { ... aiex.run ... }`
        // block must be retargeted to devA (and folded into a sibling
        // `aiex.configure @<devA.sym_name>` if one exists in the same host
        // runtime_sequence) so the merged-device runtime semantics is
        // preserved. Without this, the module verifier will emit
        // "No such device: '@<devB>'" against the dangling reference.
        if (mlir::failed(detail::rewriteHostConfigureOnDeviceMerge(
                devA->getParentOfType<mlir::ModuleOp>(), devA, devB, seqA))) {
          signalPassFailure();
          return;
        }
        devB->erase();
        devices.erase(devices.begin() + devBIdx);
        // For 1:1 / element-type pairing devBIdx == i+1 (devA position
        // unchanged → --i + for-loop's ++i = stay at same i, retry devA).
        // For convergent pairing devBIdx may be > i+1 (still i unchanged,
        // same retry semantics) OR devBIdx < i when an earlier convergent
        // merge has rotated the absorber to a lower index — in that case
        // devA shifted down by one, so subtract an additional 1 from i so
        // that the for-loop's ++i lands us back on devA's new position.
        --i;
        if (devBIdx < i + 1)
          --i;

        // --- Step 8b: Sink cores and runtime sequences to end of device body.
        //
        // Pass C emits aie.lock / aie.buffer ops after all aie.tile ops (using
        // insertAfterTile). After merging devB's tiles into devA, those inserts
        // go after the LAST tile in the merged body. However, devA's original
        // aie.core ops appear BEFORE devB's tiles in the merged body, so locks
        // emitted by Pass C come after the cores — violating MLIR dominance
        // (locks must be defined before use inside cores).
        //
        // Fix: move all aie.core, aie.mem, and aie.runtime_sequence ops in
        // devA's body to just before the aie.end terminator. This places them
        // after any locks/buffers that Pass C will insert after the tile ops.
        detail::sinkCoresMemsAndSequences(bodyA);

        // --- Step 8c: Remove dead block args for fused intermediate channels.
        //
        // After --dma-task-to-conduit, all runtime_sequence block args are
        // SSA-dead but retain the correct full-buffer types (e.g.
        // memref<256xbf16>).  After the Phase 2 merge, the merged sequence
        // has origTypesA.size() + origTypesB.size() block args — some of
        // which correspond to fused-intermediate channels erased in Step 6b.
        //
        // Using the channel→arg-index mapping recorded in Step 3.5, identify
        // which original block args are fully intermediate (all channels in
        // their group were erased) and remove them, preserving the surviving
        // full-buffer types.
        if (seqA && seqA->getNumRegions() > 0) {
          mlir::Block &seqBody = seqA->getRegion(0).front();

          // Compute dead arg indices for each original sequence.
          auto computeDeadArgs =
              [](const llvm::SmallVector<ArgGroup> &groups,
                 const llvm::StringSet<> &erasedChannels,
                 unsigned numOrigArgs) -> llvm::DenseSet<unsigned> {
            llvm::DenseSet<unsigned> dead;
            // Only trust the grouping if it matches the block arg count.
            if (groups.size() != numOrigArgs)
              return dead;
            for (const auto &group : groups) {
              bool allErased = !group.channelNames.empty();
              for (const auto &name : group.channelNames) {
                if (!erasedChannels.contains(name)) {
                  allErased = false;
                  break;
                }
              }
              if (allErased)
                dead.insert(group.argIndex);
            }
            return dead;
          };

          llvm::DenseSet<unsigned> deadA =
              computeDeadArgs(argGroupsA, erasedChannelsA,
                              static_cast<unsigned>(origTypesA.size()));
          llvm::DenseSet<unsigned> deadB =
              computeDeadArgs(argGroupsB, erasedChannelsB,
                              static_cast<unsigned>(origTypesB.size()));

          // Build surviving arg types: non-dead from A + non-dead from B.
          //
          // Use the max of (BD-derived maxExtent, origType extent) to
          // reconstruct full-buffer types:
          //   * maxExtent fixes the per-tile arg type bug — IRON sometimes
          //     records per-tile block-arg types (e.g. memref<128xbf16>)
          //     even when the BDs span the full host buffer with offsets.
          //   * origType extent preserves IRON's recorded full-buffer type
          //     when it ALREADY spans the full buffer (e.g. multi-column
          //     lowerings that record memref<8192xbf16> directly).  Taking
          //     the max keeps both cases correct without a special-case.
          auto computeFullBufferType =
              [](unsigned i, const llvm::SmallVector<ArgGroup> &groups,
                 const llvm::SmallVector<mlir::Type> &origTypes) -> mlir::Type {
            if (i >= origTypes.size())
              return origTypes.empty() ? mlir::Type{} : origTypes.back();
            mlir::Type t = origTypes[i];
            auto memTy = mlir::dyn_cast<mlir::MemRefType>(t);
            if (!memTy)
              return t;
            int64_t bdExtent = (i < groups.size()) ? groups[i].maxExtent : 0;
            int64_t origExtent =
                (memTy.hasStaticShape() && memTy.getRank() == 1)
                    ? memTy.getNumElements()
                    : 0;
            int64_t finalExtent = std::max(bdExtent, origExtent);
            if (finalExtent <= 0)
              return t;
            return mlir::MemRefType::get({finalExtent}, memTy.getElementType());
          };

          llvm::SmallVector<mlir::Type> newArgTypes;
          for (unsigned i = 0; i < origTypesA.size(); ++i) {
            if (!deadA.contains(i))
              newArgTypes.push_back(
                  computeFullBufferType(i, argGroupsA, origTypesA));
          }
          for (unsigned i = 0; i < origTypesB.size(); ++i) {
            if (!deadB.contains(i))
              newArgTypes.push_back(
                  computeFullBufferType(i, argGroupsB, origTypesB));
          }

          // Reconstruct block args if the count changed.
          if (newArgTypes.size() != seqBody.getNumArguments()) {
            for (int idx = static_cast<int>(seqBody.getNumArguments()) - 1;
                 idx >= 0; --idx)
              seqBody.eraseArgument(static_cast<unsigned>(idx));
            for (mlir::Type ty : newArgTypes)
              seqBody.addArgument(ty, seqA->getLoc());
          }

          // --- Step 8c-bis: Re-project `arg_index` on every surviving
          // put/get_memref op in the merged sequence body.
          //
          // This is the device-side counterpart to Step 8d's host-side
          // `reconcileHostRunArgsAfterTrim`.  Step 8c just reshaped the
          // merged sequence's block-arg layout (drop deadA, then drop
          // deadB), but the surviving conduit.put_memref / get_memref ops
          // still carry their pre-merge `arg_index` attributes:
          //
          //   * A-side originals: arg_index references seqA's pre-trim
          //     position (== merged-seq pre-trim position, since A's args
          //     occupy [0, origArgCountA) without offset).
          //   * B-side clones (tagged `_origin_device = "B"` at clone
          //     time): arg_index still references seqB's ORIGINAL position
          //     (0..origArgCountB-1), with NO offset for the merged
          //     layout.
          //
          // --conduit-to-dma Step 8g (`ConduitToDMALower.cpp:1126-1148`)
          // consumes arg_index directly as `blockArgs[argIdx]`, so without
          // this reprojection BDs bind to the wrong block args (Bug A /
          // Matrix Row #1 1col_small numerical wrong-output: routing-
          // independent because routing is correct — block-arg binding
          // is wrong).
          //
          // Reprojection (oldIdx → newIdx):
          //   A-side: newIdx = oldIdx − |{d ∈ deadA : d < oldIdx}|
          //   B-side: newIdx = (origArgCountA − |deadA|)
          //                    + (oldIdx − |{d ∈ deadB : d < oldIdx}|)
          //
          // The `_origin_device` discardable tag is removed in the same
          // walk so it never leaks to downstream passes.
          {
            unsigned origArgCountA_v = static_cast<unsigned>(origTypesA.size());
            unsigned aSurvCount =
                origArgCountA_v - static_cast<unsigned>(deadA.size());
            bool reprojFailed = false;
            for (mlir::Operation &op : seqBody) {
              llvm::StringRef opName = op.getName().getStringRef();
              if (opName != "conduit.put_memref" &&
                  opName != "conduit.get_memref" &&
                  opName != "conduit.put_memref_async" &&
                  opName != "conduit.get_memref_async")
                continue;
              auto argIdxAttr =
                  op.getAttrOfType<mlir::IntegerAttr>("arg_index");
              bool isB = false;
              if (auto orig =
                      op.getAttrOfType<mlir::StringAttr>("_origin_device"))
                isB = (orig.getValue() == "B");
              op.removeAttr("_origin_device");
              if (!argIdxAttr)
                continue;
              int64_t oldIdx = argIdxAttr.getInt();
              if (oldIdx < 0)
                continue;
              const llvm::DenseSet<unsigned> &dead = isB ? deadB : deadA;
              if (dead.contains(static_cast<unsigned>(oldIdx))) {
                op.emitError("conduit-fuse-operators: ")
                    << (isB ? "B" : "A")
                    << "-side put/get_memref op survived Step 8c trim but "
                       "its arg_index "
                    << oldIdx << " is in the dead-arg set";
                reprojFailed = true;
                continue;
              }
              unsigned deadBefore = 0;
              for (unsigned d : dead)
                if (static_cast<int64_t>(d) < oldIdx)
                  ++deadBefore;
              int64_t newIdx =
                  isB ? (static_cast<int64_t>(aSurvCount) +
                         (oldIdx - static_cast<int64_t>(deadBefore)))
                      : (oldIdx - static_cast<int64_t>(deadBefore));
              op.setAttr("arg_index",
                         mlir::IntegerAttr::get(argIdxAttr.getType(), newIdx));
            }
            if (reprojFailed) {
              signalPassFailure();
              return;
            }
          }

          // --- Step 8d: Phase 2 of the host-orchestrator rewrite.
          //
          // Project the same drops Step 8c just applied to the merged
          // sequence's block args into the host-side `aiex.run` arg vectors.
          // After `rewriteHostConfigureOnDeviceMerge` (the Phase 1 fold),
          // each folded run carries the naive `runA.getArgs() ++
          // runB.getArgs()` concat — which now overruns the trimmed callee.
          // The helper drops `deadA` from the first `origArgCountA` positions
          // and `deadB` from the next `origArgCountB` positions, restoring
          // arity match.
          //
          // Non-folded `aiex.configure @devA` blocks (rewritten in place) are
          // handled by the helper too: the run there only carries devB's
          // args, so only `deadB` is projected.
          if (mlir::failed(detail::reconcileHostRunArgsAfterTrim(
                  devA->getParentOfType<mlir::ModuleOp>(), devA, seqA,
                  static_cast<unsigned>(origTypesA.size()),
                  static_cast<unsigned>(origTypesB.size()), deadA, deadB))) {
            signalPassFailure();
            return;
          }
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitFuseOperatorsPass() {
  return std::make_unique<ConduitFuseOperatorsPass>();
}

} // namespace xilinx::conduit
