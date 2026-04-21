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
/// After --dma-task-to-conduit, the block args are SSA-dead but remain in
/// the function signature with the correct full-buffer types.  The
/// put/get_memref ops are ordered so that ops sharing the same host buffer
/// (block arg) are contiguous, with the first op in each group having
/// offsets[0] == 0.  We use this structural invariant to recover the
/// channel-name → block-arg-index mapping.
static llvm::SmallVector<ArgGroup> buildArgGroupsFromSeq(mlir::Block &seqBody) {
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
    auto offsetsAttr = op.getAttrOfType<mlir::DenseI64ArrayAttr>("offsets");
    int64_t offset = (offsetsAttr && !offsetsAttr.empty()) ? offsetsAttr[0] : 0;

    // Compute extent = offset + num_elems for this op.
    auto numElemsAttr = op.getAttrOfType<mlir::IntegerAttr>("num_elems");
    int64_t numElems = numElemsAttr ? numElemsAttr.getInt() : 0;
    int64_t extent = offset + numElems;

    if (groups.empty() || offset == 0) {
      unsigned idx = groups.empty() ? 0 : groups.back().argIndex + 1;
      groups.push_back({idx, {nameAttr.getValue().str()}, extent});
    } else {
      groups.back().channelNames.push_back(nameAttr.getValue().str());
      groups.back().maxExtent = std::max(groups.back().maxExtent, extent);
    }
  }
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

    // Process consecutive device pairs (A, B).
    for (size_t i = 0; i + 1 < devices.size(); ++i) {
      AIE::DeviceOp devA = devices[i];
      AIE::DeviceOp devB = devices[i + 1];

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
      llvm::SmallVector<std::pair<Create, Create>> matched;
      {
        llvm::DenseSet<mlir::Operation *> consumedInputs;
        for (Create outCh : outputChannels) {
          auto outFG = outCh.getFusionGroup();
          if (!outFG || outFG->empty())
            continue;
          for (Create inCh : inputChannels) {
            if (consumedInputs.contains(inCh.getOperation()))
              continue;
            auto inFG = inCh.getFusionGroup();
            if (inFG && *outFG == *inFG) {
              matched.push_back({outCh, inCh});
              consumedInputs.insert(inCh.getOperation());
              break;
            }
          }
        }
      }
      // Fallback: match by element_type if no fusion_group attrs found.
      if (matched.empty()) {
        llvm::DenseSet<mlir::Operation *> consumedInputs;
        for (Create outCh : outputChannels) {
          mlir::Type outET = outCh.getElementType();
          for (Create inCh : inputChannels) {
            if (consumedInputs.contains(inCh.getOperation()))
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
            std::to_string(i) + " and device " + std::to_string(i + 1) +
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

        std::string fusedName =
            "fused_intermediate_" + std::to_string(fuseCount++);

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

        // routing_mode = absent ("any"): let --conduit-infer-modes resolve.
        // Adjacent tiles (after offset) get shared_memory; non-adjacent get
        // circuit.  Eliminates DMA channels for the intermediate when tiles
        // are neighbours.
        RoutingModeAttr routingModeAttr{};

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
        auto eraseFusedMemrefOps = [&](AIE::DeviceOp device) {
          llvm::SmallVector<mlir::Operation *> toErase;
          device.walk([&](mlir::Operation *op) {
            llvm::StringRef opName = op->getName().getStringRef();
            if (opName != "conduit.put_memref" &&
                opName != "conduit.get_memref" &&
                opName != "conduit.put_memref_async" &&
                opName != "conduit.get_memref_async")
              return;
            auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
            if (nameAttr && nameAttr.getValue() == fusedName)
              toErase.push_back(op);
          });
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
        mlir::Operation *seqA = nullptr;
        for (mlir::Operation &op : bodyA) {
          if (op.getName().getStringRef() == "aie.runtime_sequence") {
            seqA = &op;
            break;
          }
        }

        // Phase 1: move all non-sequence, non-terminator ops from bodyB into
        // bodyA. This establishes the tile/shim SSA values in devA's scope so
        // that the sequence body ops (which use those tile values) can be moved
        // safely in Phase 2.
        mlir::OpBuilder b(ctx);
        if (bodyA.mightHaveTerminator()) {
          if (mlir::Operation *term = bodyA.getTerminator())
            b.setInsertionPoint(term);
          else
            b.setInsertionPointToEnd(&bodyA);
        } else {
          b.setInsertionPointToEnd(&bodyA);
        }
        llvm::SmallVector<mlir::Operation *> seqOps;
        {
          llvm::SmallVector<mlir::Operation *> nonSeq;
          for (mlir::Operation &op : bodyB) {
            if (op.hasTrait<mlir::OpTrait::IsTerminator>())
              continue;
            if (op.getName().getStringRef() == "aie.runtime_sequence")
              seqOps.push_back(&op);
            else
              nonSeq.push_back(&op);
          }
          for (mlir::Operation *op : nonSeq) {
            op->remove();
            b.insert(op);
          }
        }

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

              // Re-insert in interleaved order.
              // 1. One-shot ops from A (re-insert originals).
              for (mlir::Operation *op : partA.oneShot)
                seqBuilder.insert(op);
              // 2. One-shot ops from B (clone with arg mapping).
              for (mlir::Operation *op : partB.oneShot)
                seqBuilder.clone(*op, argMapping);
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
                    seqBuilder.clone(*op, argMapping);
              }
              // seqB's body is now represented in seqA. seqB itself remains
              // in devB and will be erased with devB below.
            } else if (!seqA) {
              // devA has no sequence yet — move devB's as-is.
              seqB->remove();
              b.insert(seqB);
              seqA = seqB;
            }
          }
        }

        // bodyB is now empty except its aie.end terminator. All SSA values
        // defined in bodyB have been moved to bodyA, so devB->erase() will
        // not encounter live-use violations.
        devB->erase();

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
        {
          llvm::SmallVector<mlir::Operation *> toSink;
          for (mlir::Operation &op : bodyA) {
            llvm::StringRef name = op.getName().getStringRef();
            if (name == "aie.core" || name == "aie.mem" ||
                name == "aie.runtime_sequence")
              toSink.push_back(&op);
          }
          mlir::Operation *termA =
              bodyA.mightHaveTerminator() ? bodyA.getTerminator() : nullptr;
          for (mlir::Operation *op : toSink) {
            if (termA)
              op->moveBefore(termA);
            else
              op->moveBefore(&bodyA, bodyA.end());
          }
        }

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
          // Use computed max extents from buildArgGroupsFromSeq() to
          // reconstruct full-buffer types, fixing the per-tile arg type bug.
          auto computeFullBufferType =
              [](unsigned i, const llvm::SmallVector<ArgGroup> &groups,
                 const llvm::SmallVector<mlir::Type> &origTypes) -> mlir::Type {
            if (i < groups.size() && groups[i].maxExtent > 0 &&
                i < origTypes.size()) {
              if (auto memTy = mlir::dyn_cast<mlir::MemRefType>(origTypes[i])) {
                return mlir::MemRefType::get({groups[i].maxExtent},
                                             memTy.getElementType());
              }
            }
            return origTypes[i];
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
