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
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
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
  bool hasShimConsumer = tileIt != inferredMap.end() && !tileIt->second.shimConsumerTiles.empty();

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
      llvm::SmallVector<std::pair<Create, Create>> matched;
      for (Create outCh : outputChannels) {
        auto outFG = outCh.getFusionGroup();
        if (!outFG || outFG->empty())
          continue;
        for (Create inCh : inputChannels) {
          auto inFG = inCh.getFusionGroup();
          if (inFG && *outFG == *inFG) {
            matched.push_back({outCh, inCh});
            break;
          }
        }
      }
      // Fallback: match by element_type if no fusion_group attrs found.
      if (matched.empty()) {
        for (Create outCh : outputChannels) {
          mlir::Type outET = outCh.getElementType();
          for (Create inCh : inputChannels) {
            mlir::Type inET = inCh.getElementType();
            if (outET == inET) {
              matched.push_back({outCh, inCh});
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
                               /*dma_repeat=*/mlir::IntegerAttr{});

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
            auto nameAttr =
                op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
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
          Key k = {alloc.getTile(),
                   static_cast<int>(alloc.getChannelDir())};
          groups[k].push_back(alloc);
        });
        for (auto &[key, allocs] : groups) {
          // Sort by original channel index to preserve relative order.
          llvm::sort(allocs, [](AIE::ShimDMAAllocationOp a,
                                AIE::ShimDMAAllocationOp b) {
            return a.getChannelIndex() < b.getChannelIndex();
          });
          for (unsigned idx = 0; idx < allocs.size(); ++idx) {
            if (allocs[idx].getChannelIndex() !=
                static_cast<int64_t>(idx))
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
              mlir::OpBuilder seqBuilder(ctx);
              // Use setInsertionPoint(terminator) if one exists (e.g. aie.core
              // has aie.end), otherwise setInsertionPointToEnd for blocks
              // without terminators (aie.runtime_sequence).
              if (seqBodyA.mightHaveTerminator()) {
                if (mlir::Operation *term = seqBodyA.getTerminator())
                  seqBuilder.setInsertionPoint(term);
                else
                  seqBuilder.setInsertionPointToEnd(&seqBodyA);
              } else {
                seqBuilder.setInsertionPointToEnd(&seqBodyA);
              }
              for (mlir::Operation &inner : seqBodyB) {
                if (inner.hasTrait<mlir::OpTrait::IsTerminator>())
                  continue;
                seqBuilder.clone(inner, argMapping);
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

        // --- Step 8c: Eliminate dead block args from the merged sequence.
        //
        // After --dma-task-to-conduit, conduit.put/get_memref ops are purely
        // attribute-based (no SSA operands).  All runtime_sequence block args
        // are therefore SSA-dead.  After fusion, the merged sequence carries
        // extra block args from the intermediate channels (erased in Step 6b)
        // and unused pass-through buffers.
        //
        // Reconstruct the block args to have exactly one per surviving
        // put/get_memref op.  The type of each arg is derived from the op's
        // num_elems attribute and the scalar element type of the referenced
        // conduit.create.
        if (seqA && seqA->getNumRegions() > 0) {
          mlir::Block &seqBody = seqA->getRegion(0).front();

          // Collect surviving put/get_memref ops in body order.
          llvm::SmallVector<mlir::Operation *> survivingOps;
          for (mlir::Operation &op : seqBody) {
            llvm::StringRef n = op.getName().getStringRef();
            if (n == "conduit.put_memref" || n == "conduit.get_memref" ||
                n == "conduit.put_memref_async" ||
                n == "conduit.get_memref_async")
              survivingOps.push_back(&op);
          }

          // Build a name→element_type map from conduit.create ops in devA.
          llvm::StringMap<mlir::Type> channelElemTypes;
          devA.walk([&](Create create) {
            channelElemTypes[create.getName()] = create.getElementType();
          });

          // Derive block arg types from surviving ops.
          llvm::SmallVector<mlir::Type> newArgTypes;
          bool allResolved = true;
          for (mlir::Operation *op : survivingOps) {
            auto nameAttr =
                op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
            auto numElemsAttr =
                op->getAttrOfType<mlir::IntegerAttr>("num_elems");
            if (!nameAttr || !numElemsAttr) {
              allResolved = false;
              break;
            }
            auto it = channelElemTypes.find(nameAttr.getValue());
            if (it == channelElemTypes.end()) {
              allResolved = false;
              break;
            }
            mlir::Type scalarType;
            if (auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(it->second))
              scalarType = memrefTy.getElementType();
            if (!scalarType) {
              allResolved = false;
              break;
            }
            int64_t numElems = numElemsAttr.getInt();
            newArgTypes.push_back(
                mlir::MemRefType::get({numElems}, scalarType));
          }

          // Only reconstruct if we resolved all types and the count differs.
          if (allResolved &&
              newArgTypes.size() != seqBody.getNumArguments()) {
            // Erase existing args back-to-front (none have SSA uses).
            for (int idx = static_cast<int>(seqBody.getNumArguments()) - 1;
                 idx >= 0; --idx)
              seqBody.eraseArgument(static_cast<unsigned>(idx));
            // Add new args matching surviving ops.
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
