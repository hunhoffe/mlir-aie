//===- ConduitFuseCoreBodyPass.cpp - conduit-fuse-core-bodies ---*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --conduit-fuse-core-bodies: Loop-body IRON operator fusion via Conduit IR.
//
// Given two aie.core ops on the same tile connected by an intermediate
// conduit.create channel, this pass:
//
//   1. Identifies fusable core pairs (same tile, connected by intermediate
//      conduit, 1:1 rate-matched, matching loop structure).
//   2. Decides intermediate routing: L1 (memref.alloc) if intermediate fits
//      in tile L1 SRAM, MemTile relay (conduit.scatter/gather) if L1 pressure,
//      skip if neither suffices.
//   3. Composes the two core bodies into a single aie.core body by cloning
//      Core B's loop body into Core A after the intermediate produce/release.
//   4. Replaces intermediate conduit.acquire/release/subview_access ops with
//      direct memref.alloc references (L1) or relay buffer references
//      (MemTile).
//   5. Deletes the intermediate conduit.create, dead Core B, and associated
//      runtime sequence DMA ops.
//
// Prerequisites:
//   --objectfifo-to-conduit (Pass A) + --dma-task-to-conduit
//   IRON operators emit explicit scf.for tile loops (while_true=False for test)
//
// Run before:
//   --conduit-infer-modes --conduit-depth-promote --conduit-to-dma
//
// Limitations:
//   - Only 1:1 rate-matched (SDF period 1) edges are supported.
//   - Multi-consumer (broadcast) intermediates are not fused.
//   - Three-way chains are handled via iterative re-discovery.
//   - MemTile relay in loop-body fusion is an exposed stall (sequential core).
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "ConduitTileInference.h"
#include "DeviceMergeUtils.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITFUSECOREBODIES
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Step 0: Cross-device merge for fusion_group connections
// ---------------------------------------------------------------------------

/// Check if two devices are connected by a conduit channel pair with matching
/// fusion_group attributes.
///
/// Pattern E guard: a channel that participates in a `conduit.scatter` /
/// `conduit.gather` is a forward-chain endpoint (e.g., from
/// `aie.objectfifo.link`).  The Step 0 consumer→producer name unification
/// inside `mergeAndUnifyDevices` only updates ops carrying a `name`
/// FlatSymbolRefAttr (acquire/release/wait_window), so unifying such a
/// channel would leave the scatter/gather op pointing at a dangling
/// FlatSymbolRefAttr.  When a matched pair has Pattern E on either side,
/// emit a remark on the offending channel and treat the devices as not
/// connected for fusion.  `alreadyRemarked` deduplicates remarks across
/// outer-loop restarts in `mergeDevicesForFusion`.
static bool devicesConnectedByFusionGroup(AIE::DeviceOp devA,
                                          AIE::DeviceOp devB,
                                          llvm::StringSet<> &alreadyRemarked) {
  // Build fusion_group → Create map for devA so we can locate the producer-
  // side conduit.create when a match fires.
  llvm::StringMap<Create> groupsA;
  devA.walk([&](Create op) {
    auto fg = op.getFusionGroup();
    if (fg && !fg->empty())
      groupsA[*fg] = op;
  });

  bool found = false;
  devB.walk([&](Create opB) {
    if (found)
      return;
    auto fg = opB.getFusionGroup();
    if (!fg || fg->empty())
      return;
    auto it = groupsA.find(*fg);
    if (it == groupsA.end())
      return;
    Create opA = it->second;
    bool prodIsFwd = detail::isForwardChainEndpoint(devA, opA.getName());
    bool consIsFwd = detail::isForwardChainEndpoint(devB, opB.getName());
    if (prodIsFwd || consIsFwd) {
      if (prodIsFwd && alreadyRemarked.insert(opA.getName()).second)
        opA.emitRemark("conduit-fuse-core-bodies: skipping fusion_group "
                       "match for output channel @")
            << opA.getName()
            << " — forward-chain / link-only endpoint (Pattern E); "
               "scatter/gather references cannot be safely renamed";
      if (consIsFwd && alreadyRemarked.insert(opB.getName()).second)
        opB.emitRemark("conduit-fuse-core-bodies: skipping fusion_group "
                       "match for input channel @")
            << opB.getName()
            << " — forward-chain / link-only endpoint (Pattern E); "
               "scatter/gather references cannot be safely renamed";
      return; // Treat as not connected; keep searching for other matches.
    }
    found = true;
  });
  return found;
}

/// Dedup tile ops: if multiple aie.tile(col, row) ops have the same
/// coordinates, keep the first and replace uses of duplicates with it.
static void dedupTileOps(AIE::DeviceOp device) {
  using TileCoord = std::pair<int64_t, int64_t>;
  llvm::DenseMap<TileCoord, AIE::TileOp> tileMap;
  llvm::SmallVector<AIE::TileOp> duplicates;
  device.walk([&](AIE::TileOp tile) {
    TileCoord key{tile.getCol(), tile.getRow()};
    auto it = tileMap.find(key);
    if (it != tileMap.end()) {
      tile.getResult().replaceAllUsesWith(it->second.getResult());
      duplicates.push_back(tile);
    } else {
      tileMap[key] = tile;
    }
  });
  for (AIE::TileOp tile : duplicates)
    tile->erase();
}

/// Merge devB into devA (same-tile mode) and unify the fusion_group channel
/// pair into a single channel so that findFusableCorePairs can detect the
/// cross-device pair as a same-device pair.
///
/// Returns failure() if the host-orchestrator rewrite that runs immediately
/// before `devB->erase()` cannot preserve runtime semantics (e.g., arity
/// mismatch between the merged runtime_sequence and a host-side aiex.run).
static mlir::LogicalResult mergeAndUnifyDevices(AIE::DeviceOp devA,
                                                AIE::DeviceOp devB,
                                                mlir::MLIRContext *ctx) {
  // --- Find the matching fusion_group channel pair. ---
  Create producerChannel = nullptr;
  Create consumerChannel = nullptr;

  devA.walk([&](Create opA) {
    if (producerChannel)
      return;
    auto fgA = opA.getFusionGroup();
    if (!fgA || fgA->empty())
      return;
    devB.walk([&](Create opB) {
      if (consumerChannel)
        return;
      auto fgB = opB.getFusionGroup();
      if (fgB && *fgA == *fgB) {
        producerChannel = opA;
        consumerChannel = opB;
      }
    });
  });

  if (!producerChannel || !consumerChannel)
    return mlir::success();

  // Pattern E defensive guard: if either side participates in a
  // conduit.scatter/gather, the consumer→producer name unification below
  // would leave dangling FlatSymbolRefAttrs because the rename walk only
  // updates `name` attrs (not scatter/gather src/dsts).  The primary filter
  // lives in devicesConnectedByFusionGroup so the outer loop never asks us
  // to merge such a pair, but keep this as a hard backstop so a future
  // refactor cannot reintroduce the bug silently.
  if (detail::isForwardChainEndpoint(devA, producerChannel.getName()) ||
      detail::isForwardChainEndpoint(devB, consumerChannel.getName()))
    return mlir::success();

  std::string prodName = producerChannel.getName().str();
  std::string consName = consumerChannel.getName().str();

  // --- Deconflict channel names before merge. ---
  // If devB has channels with the same name as devA channels (other than
  // the matched consumer channel which will be unified), rename them in
  // devB before moving ops. This prevents the post-merge rename step from
  // corrupting unrelated channels that happen to share a name.
  {
    llvm::StringSet<> devANames;
    devA.walk([&](Create op) { devANames.insert(op.getName()); });

    // Collect conflicting devB channel names.
    llvm::SmallVector<std::pair<std::string, std::string>> renames;
    devB.walk([&](Create op) {
      std::string name = op.getName().str();
      if (name == consName)
        return; // Will be unified with producerChannel.
      if (!devANames.count(name))
        return; // No conflict.
      // Generate unique name.
      std::string newName = name + "_merged";
      int suffix = 0;
      while (devANames.count(newName))
        newName = name + "_merged_" + std::to_string(suffix++);
      devANames.insert(newName);
      renames.push_back({name, newName});
    });

    // Apply renames within devB before ops are moved.
    for (auto &[oldN, newN] : renames) {
      auto newRef = mlir::FlatSymbolRefAttr::get(ctx, newN);
      auto newStr = mlir::StringAttr::get(ctx, newN);
      devB.walk([&](mlir::Operation *op) {
        // conduit.create sym_name.
        if (auto createOp = mlir::dyn_cast<Create>(op)) {
          if (createOp.getName() == oldN)
            createOp.setSymNameAttr(newStr);
          return;
        }
        // ShimDMAAllocation sym_name and conduit_channel attr.
        if (auto alloc = mlir::dyn_cast<AIE::ShimDMAAllocationOp>(op)) {
          if (alloc.getSymName() == oldN)
            alloc.setSymNameAttr(newStr);
          std::string shimSuffix = oldN + "_shim_alloc";
          if (alloc.getSymName() == shimSuffix)
            alloc.setSymNameAttr(
                mlir::StringAttr::get(ctx, newN + "_shim_alloc"));
          auto cc =
              alloc->getAttrOfType<mlir::FlatSymbolRefAttr>("conduit_channel");
          if (cc && cc.getValue() == oldN)
            alloc->setAttr("conduit_channel", newRef);
          return;
        }
        // "name" attr on conduit ops (acquire, put_memref, get_memref, etc.).
        auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
        if (nameAttr && nameAttr.getValue() == oldN)
          op->setAttr("name", newRef);
      });
    }
  }

  // --- Merge devB body into devA (same-tile mode — no column offset). ---
  {
    mlir::Block &bodyA = devA.getBodyRegion().front();
    mlir::Block &bodyB = devB.getBodyRegion().front();

    // Find devA's runtime_sequence (may not exist).
    mlir::Operation *seqA = detail::findRuntimeSequence(bodyA);

    // Phase 1: move all non-sequence, non-terminator ops from bodyB into bodyA.
    llvm::SmallVector<mlir::Operation *> seqOps =
        detail::movePhase1NonSequenceOps(bodyA, bodyB);

    // Phase 2: merge devB's runtime_sequence(s) into devA's sequence.
    detail::mergeRuntimeSequencesSimple(seqA, seqOps, bodyA);

    // devB body is now empty except its aie.end terminator. Before
    // erasing devB, retarget any module-level `aiex.configure @<devB>`
    // host-orchestrator references onto devA — folding into a sibling
    // `aiex.configure @<devA>` if one exists in the same host
    // runtime_sequence — so the merged-device runtime semantics survive
    // the merge. Without this, the module verifier emits
    // "No such device: '@<devB>'" against the dangling reference.
    if (mlir::failed(detail::rewriteHostConfigureOnDeviceMerge(
            devA->getParentOfType<mlir::ModuleOp>(), devA, devB, seqA)))
      return mlir::failure();
    devB->erase();

    // Sink cores, mem, and runtime_sequences to end of device body
    // (before aie.end terminator) to maintain dominance.
    detail::sinkCoresMemsAndSequences(bodyA);
  }

  // --- Dedup tile ops (same coordinates → single SSA value). ---
  dedupTileOps(devA);

  // --- Unify fusion_group channels: rename consumer → producer. ---
  // Replace all core-body references to consumerChannel with producerChannel
  // so findFusableCorePairs sees a single intermediate channel.
  auto prodNameAttr = mlir::FlatSymbolRefAttr::get(ctx, prodName);

  devA.walk([&](Acquire op) {
    if (op.getName() == consName)
      op->setAttr("name", prodNameAttr);
  });
  devA.walk([&](AcquireAsync op) {
    if (op.getName() == consName)
      op->setAttr("name", prodNameAttr);
  });
  devA.walk([&](ReleaseAsync op) {
    if (op.getName() == consName)
      op->setAttr("name", prodNameAttr);
  });
  devA.walk([&](WaitWindow op) {
    if (op.getName() == consName)
      op->setAttr("name", prodNameAttr);
  });

  // --- Delete dead consumer channel ops. ---
  llvm::SmallVector<mlir::Operation *> toErase;

  // Delete the consumer conduit.create (now inside devA).
  devA.walk([&](Create op) {
    if (op.getName() == consName)
      toErase.push_back(op.getOperation());
  });

  // Delete runtime_sequence ops referencing the consumer channel.
  devA.walk([&](mlir::Operation *op) {
    llvm::StringRef opName = op->getName().getStringRef();
    if (opName != "conduit.put_memref" && opName != "conduit.get_memref" &&
        opName != "conduit.put_memref_async" &&
        opName != "conduit.get_memref_async")
      return;
    auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
    if (nameAttr && nameAttr.getValue() == consName)
      toErase.push_back(op);
  });

  // Delete ShimDMAAllocation referencing the consumer channel.
  std::string consShimName = consName + "_shim_alloc";
  devA.walk([&](AIE::ShimDMAAllocationOp alloc) {
    auto ccAttr =
        alloc->getAttrOfType<mlir::FlatSymbolRefAttr>("conduit_channel");
    if ((ccAttr && ccAttr.getValue() == consName) ||
        alloc.getSymName() == consName || alloc.getSymName() == consShimName)
      toErase.push_back(alloc.getOperation());
  });

  for (auto *op : toErase)
    op->erase();
  return mlir::success();
}

/// Pre-process: merge cross-device fusion_group connections into single
/// devices so that findFusableCorePairs can detect them.
///
/// Returns failure() if any underlying merge surfaces a host-orchestrator
/// rewrite error (see rewriteHostConfigureOnDeviceMerge).
static mlir::LogicalResult mergeDevicesForFusion(mlir::ModuleOp module,
                                                 mlir::MLIRContext *ctx) {
  // Channels we have already remarked on as Pattern E forward-chain endpoints
  // — survives outer-loop restarts so the diagnostic fires at most once per
  // channel even when other unrelated pairs successfully merge.
  llvm::StringSet<> alreadyRemarked;
  bool merged = true;
  while (merged) {
    merged = false;
    llvm::SmallVector<AIE::DeviceOp> devices;
    module.walk([&](AIE::DeviceOp dev) { devices.push_back(dev); });

    if (devices.size() < 2)
      return mlir::success();

    for (size_t i = 0; i + 1 < devices.size(); ++i) {
      AIE::DeviceOp devA = devices[i];
      AIE::DeviceOp devB = devices[i + 1];

      if (!devicesConnectedByFusionGroup(devA, devB, alreadyRemarked))
        continue;

      if (mlir::failed(mergeAndUnifyDevices(devA, devB, ctx)))
        return mlir::failure();
      merged = true;
      break; // Restart — device list is invalidated.
    }
  }
  return mlir::success();
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Describes a fusable core pair connected by an intermediate conduit.
struct FusableCorePair {
  AIE::CoreOp producerCore; // Core A: produces into intermediate
  AIE::CoreOp consumerCore; // Core B: consumes from intermediate
  Create intermediateConduit;
  mlir::Value tile; // Shared tile SSA value
};

/// Routing decision for the intermediate buffer.
enum class IntermediateRoute {
  L1,      // memref.alloc in tile L1 SRAM
  MemTile, // conduit.scatter/gather through MemTile
  Skip,    // Cannot route; skip fusion
};

// ---------------------------------------------------------------------------
// Step 1: Identify fusable core pairs
// ---------------------------------------------------------------------------

/// Collect all conduit channel names that a core produces into (Produce port).
static llvm::SmallVector<std::string> getProducedChannels(AIE::CoreOp core) {
  llvm::StringSet<> seen;
  llvm::SmallVector<std::string> channels;
  core.walk([&](Acquire acqOp) {
    if (acqOp.getPort() == Port::Produce) {
      std::string name = acqOp.getName().str();
      if (seen.insert(name).second)
        channels.push_back(name);
    }
  });
  core.walk([&](AcquireAsync acqOp) {
    if (acqOp.getPort() == Port::Produce) {
      std::string name = acqOp.getName().str();
      if (seen.insert(name).second)
        channels.push_back(name);
    }
  });
  return channels;
}

/// Collect all conduit channel names that a core consumes from (Consume port).
static llvm::SmallVector<std::string> getConsumedChannels(AIE::CoreOp core) {
  llvm::StringSet<> seen;
  llvm::SmallVector<std::string> channels;
  core.walk([&](Acquire acqOp) {
    if (acqOp.getPort() == Port::Consume) {
      std::string name = acqOp.getName().str();
      if (seen.insert(name).second)
        channels.push_back(name);
    }
  });
  core.walk([&](AcquireAsync acqOp) {
    if (acqOp.getPort() == Port::Consume) {
      std::string name = acqOp.getName().str();
      if (seen.insert(name).second)
        channels.push_back(name);
    }
  });
  return channels;
}

/// Check whether the intermediate conduit has 1:1 rate-matched edges.
/// Returns true if rates are compatible for loop-body fusion.
static bool isRateMatched(Create intermediateConduit) {
  // If no rate annotations, assume 1:1 (SDF default).
  auto prodRates = intermediateConduit.getProducerRates();
  auto consRates = intermediateConduit.getConsumerRates();
  if (!prodRates && !consRates)
    return true;
  // If rates present, check they are single-phase and equal.
  if (prodRates && consRates) {
    llvm::ArrayRef<int64_t> pRates = *prodRates;
    llvm::ArrayRef<int64_t> cRates = *consRates;
    // Single-phase (SDF period 1): one rate value each, and they must match.
    if (pRates.size() == 1 && cRates.size() == 1 && pRates[0] == cRates[0])
      return true;
    // Multi-phase: not yet supported for loop-body fusion.
    return false;
  }
  // One set present but not the other — asymmetric rates, not 1:1.
  return false;
}

/// Try to get the outermost scf.for from a core body.
/// Returns nullptr if no scf.for found (flat body).
static mlir::scf::ForOp getOutermostFor(AIE::CoreOp core) {
  mlir::scf::ForOp result = nullptr;
  for (mlir::Operation &op : core.getBody().front()) {
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      result = forOp;
      break;
    }
  }
  return result;
}

/// Check whether two cores have matching outermost loop structure.
/// Both must have the same scf.for trip count, or both must have flat bodies.
// TODO(#88): when consumer site is ready, lift the constant-UB requirement
// using xilinx::conduit::evaluateConstantsInMap from LoopAnalysisUtils.h.
// Currently pinned by infer_iter_count_rtp_then_fuse_core_bodies_relay_propagates_repeat.mlir
// which CHECK-NOTs the relay create + COUNT-2 cores; flips to positive
// @intermediate_relay when this lifts.
static bool hasMatchingLoopStructure(AIE::CoreOp coreA, AIE::CoreOp coreB) {
  mlir::scf::ForOp forA = getOutermostFor(coreA);
  mlir::scf::ForOp forB = getOutermostFor(coreB);

  // Both flat (no scf.for): match.
  if (!forA && !forB)
    return true;

  // One has a loop, the other doesn't: no match.
  if (!forA || !forB)
    return false;

  // Both have loops: check constant bounds and trip counts match.
  auto getConstVal = [](mlir::Value v) -> std::optional<int64_t> {
    if (auto cst = v.getDefiningOp<mlir::arith::ConstantIndexOp>())
      return cst.value();
    return std::nullopt;
  };

  auto lbA = getConstVal(forA.getLowerBound());
  auto ubA = getConstVal(forA.getUpperBound());
  auto stepA = getConstVal(forA.getStep());
  auto lbB = getConstVal(forB.getLowerBound());
  auto ubB = getConstVal(forB.getUpperBound());
  auto stepB = getConstVal(forB.getStep());

  if (!lbA || !ubA || !stepA || !lbB || !ubB || !stepB)
    return false;

  // Compute trip counts.
  if (*stepA == 0 || *stepB == 0)
    return false;
  int64_t tripA = (*ubA - *lbA + *stepA - 1) / *stepA;
  int64_t tripB = (*ubB - *lbB + *stepB - 1) / *stepB;
  return tripA == tripB;
}

/// Find all fusable core pairs within a single aie.device.
static llvm::SmallVector<FusableCorePair>
findFusableCorePairs(AIE::DeviceOp device,
                     const llvm::StringMap<InferredTiles> &inferredMap) {
  llvm::SmallVector<FusableCorePair> pairs;

  // 1. Collect all aie.core ops grouped by tile coordinates (col, row).
  // We group by coordinates rather than SSA value because after
  // --aie-combine-device, the same physical tile may have multiple
  // aie.tile SSA values.
  using TileCoord = std::pair<int64_t, int64_t>;
  llvm::DenseMap<TileCoord, llvm::SmallVector<AIE::CoreOp>> tileCores;
  device.walk([&](AIE::CoreOp core) {
    if (auto tileOp = core.getTile().getDefiningOp<AIE::TileOp>()) {
      TileCoord key{tileOp.getCol(), tileOp.getRow()};
      tileCores[key].push_back(core);
    }
  });

  // Build a name→Create map for conduit.create ops.
  llvm::StringMap<Create> createMap;
  device.walk(
      [&](Create createOp) { createMap[createOp.getName()] = createOp; });

  // Build a fusion_group→Create map for cross-device channel matching.
  llvm::StringMap<llvm::SmallVector<Create>> fusionGroupMap;
  device.walk([&](Create createOp) {
    auto fg = createOp.getFusionGroup();
    if (fg && !fg->empty())
      fusionGroupMap[*fg].push_back(createOp);
  });

  // 2. For each tile with >= 2 cores, find fusable pairs.
  for (auto &[coord, cores] : tileCores) {
    if (cores.size() < 2)
      continue;

    for (size_t i = 0; i < cores.size(); ++i) {
      for (size_t j = 0; j < cores.size(); ++j) {
        if (i == j)
          continue;

        AIE::CoreOp coreA = cores[i]; // potential producer
        AIE::CoreOp coreB = cores[j]; // potential consumer

        auto produced = getProducedChannels(coreA);
        auto consumed = getConsumedChannels(coreB);

        for (const std::string &prodCh : produced) {
          for (const std::string &consCh : consumed) {
            Create intermediateConduit = nullptr;

            if (prodCh == consCh) {
              // Same channel name: direct match.
              auto createIt = createMap.find(prodCh);
              if (createIt == createMap.end())
                continue;
              intermediateConduit = createIt->second;
            } else {
              // Different names: check if connected by matching fusion_group.
              auto prodIt = createMap.find(prodCh);
              auto consIt = createMap.find(consCh);
              if (prodIt == createMap.end() || consIt == createMap.end())
                continue;
              Create prodCreate = prodIt->second;
              Create consCreate = consIt->second;
              auto fgProd = prodCreate.getFusionGroup();
              auto fgCons = consCreate.getFusionGroup();
              if (!fgProd || fgProd->empty() || !fgCons || fgCons->empty() ||
                  *fgProd != *fgCons)
                continue;
              // Matched by fusion_group — use the producer channel as
              // the intermediate (consumer channel will also be cleaned up).
              intermediateConduit = prodCreate;
            }

            if (!intermediateConduit)
              continue;

            // Check the intermediate has only one consumer (no broadcast).
            std::string intermediateName = intermediateConduit.getName().str();
            auto inferIt = inferredMap.find(intermediateName);
            if (inferIt != inferredMap.end()) {
              if (inferIt->second.consumerTiles.size() > 1)
                continue; // Multi-consumer; skip.
            }

            // Check rate matching.
            if (!isRateMatched(intermediateConduit))
              continue;

            // Check matching loop structure.
            if (!hasMatchingLoopStructure(coreA, coreB))
              continue;

            FusableCorePair pair;
            pair.producerCore = coreA;
            pair.consumerCore = coreB;
            pair.intermediateConduit = intermediateConduit;
            pair.tile = coreA.getTile();
            pairs.push_back(pair);
          }
        }
      }
    }
  }

  return pairs;
}

// ---------------------------------------------------------------------------
// Step 2: Decide intermediate routing
// ---------------------------------------------------------------------------

/// Compute the byte size of the intermediate buffer.
static int64_t computeIntermediateSize(Create intermediateConduit) {
  mlir::Type elemType = intermediateConduit.getElementType();
  auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(elemType);
  if (!memrefTy)
    return 0;

  // Compute total elements = product of shape dimensions.
  int64_t numElements = 1;
  for (int64_t dim : memrefTy.getShape()) {
    if (mlir::ShapedType::isDynamic(dim))
      return 0; // Dynamic shape; cannot statically determine size.
    numElements *= dim;
  }

  // Get scalar element size in bytes.
  mlir::Type scalarType = memrefTy.getElementType();
  unsigned bitsPerElem = scalarType.getIntOrFloatBitWidth();
  int64_t bytesPerElem = (bitsPerElem + 7) / 8;

  return numElements * bytesPerElem;
}

/// Estimate available L1 SRAM on a tile after existing buffer allocations.
static int64_t estimateAvailableL1(mlir::Value tile, AIE::DeviceOp device) {
  // Query target model for tile L1 capacity.
  const AIE::AIETargetModel &tm = AIE::getTargetModel(device);
  int64_t capacity = static_cast<int64_t>(tm.getLocalMemorySize());

  // Walk all aie.buffer ops on this tile, sum allocated bytes.
  int64_t allocated = 0;
  device.walk([&](AIE::BufferOp bufOp) {
    if (bufOp.getTile() != tile)
      return;
    auto memrefTy = bufOp.getType();
    int64_t numElems = 1;
    for (int64_t dim : memrefTy.getShape()) {
      if (mlir::ShapedType::isDynamic(dim))
        return;
      numElems *= dim;
    }
    unsigned bitsPerElem = memrefTy.getElementType().getIntOrFloatBitWidth();
    allocated += numElems * ((bitsPerElem + 7) / 8);
  });

  return capacity - allocated;
}

/// Decide the routing for the intermediate buffer.
static IntermediateRoute decideRoute(Create intermediateConduit,
                                     mlir::Value tile, AIE::DeviceOp device) {
  int64_t intermediateSize = computeIntermediateSize(intermediateConduit);
  int64_t availableL1 = estimateAvailableL1(tile, device);

  if (intermediateSize > 0 && intermediateSize <= availableL1)
    return IntermediateRoute::L1;

  // MemTile route: check if the device has MemTiles with sufficient capacity.
  // Loop-body fusion through MemTile is an exposed stall (sequential core)
  // but is still preferable to not fusing at all.
  const AIE::AIETargetModel &tm = AIE::getTargetModel(device);
  int64_t memTileCapacity = static_cast<int64_t>(tm.getMemTileSize());
  if (memTileCapacity > 0 && intermediateSize > 0 &&
      intermediateSize <= memTileCapacity)
    return IntermediateRoute::MemTile;

  return IntermediateRoute::Skip;
}

// ---------------------------------------------------------------------------
// Step 3: Compose core bodies
// ---------------------------------------------------------------------------

/// Get the block containing the core body ops (before aie.end).
/// For flat bodies: the core's single block.
/// For loop bodies: the block inside the outermost scf.for.
static mlir::Block &getCoreBodyBlock(AIE::CoreOp core) {
  return core.getBody().front();
}

/// Emit the conduit.create relay channel and conduit.scatter op for MemTile
/// relay routing. Returns the relay channel name.
static std::string emitMemTileRelay(FusableCorePair &pair, AIE::DeviceOp device,
                                    mlir::OpBuilder &builder,
                                    mlir::MLIRContext *ctx) {
  Create intermediateConduit = pair.intermediateConduit;
  std::string channelName = intermediateConduit.getName().str();
  std::string relayName = channelName + "_relay";

  // Find MemTile column from the producer core's tile.
  int64_t col = 0;
  if (auto tileOp = pair.tile.getDefiningOp<AIE::TileOp>())
    col = tileOp.getCol();

  // Create relay conduit.create with same characteristics as intermediate.
  // Propagate `bd_repeat` and `dma_repeat` from the source intermediate so
  // Pass A's inferred replay annotations survive the relay split.  If the
  // source has no annotation (legitimate Dynamic-skip case from Pass A),
  // the getters return null and the relay create is left unannotated too.
  builder.setInsertionPointAfter(intermediateConduit);
  builder.create<Create>(
      intermediateConduit.getLoc(), mlir::StringAttr::get(ctx, relayName),
      intermediateConduit.getElementTypeAttr(),
      intermediateConduit.getDepthAttr(),
      /*routing_mode=*/intermediateConduit.getRoutingModeAttr(),
      /*sync_mode=*/SyncModeAttr{},
      /*producer_rates=*/nullptr,
      /*consumer_rates=*/nullptr,
      /*fusion_group=*/mlir::StringAttr{},
      /*bd_repeat=*/intermediateConduit.getBdRepeatAttr(),
      /*dma_repeat=*/intermediateConduit.getDmaRepeatAttr(),
      /*producer_dimensions=*/nullptr,
      /*consumer_dimensions=*/nullptr);

  // Emit conduit.scatter { src=@channel, dsts=[@channel_relay],
  //                        memtile="tile(col,1)" }.
  std::string memtileStr;
  {
    llvm::raw_string_ostream os(memtileStr);
    os << "tile(" << col << ",1)";
  }
  mlir::FlatSymbolRefAttr srcRef =
      mlir::FlatSymbolRefAttr::get(ctx, channelName);
  mlir::ArrayAttr dstsArr =
      mlir::ArrayAttr::get(ctx, {mlir::FlatSymbolRefAttr::get(ctx, relayName)});
  mlir::StringAttr memtileAttr = mlir::StringAttr::get(ctx, memtileStr);
  builder.create<ScatterOp>(intermediateConduit.getLoc(), srcRef, dstsArr,
                            memtileAttr, /*offsets=*/nullptr);

  return relayName;
}

/// Compose the consumer core body into the producer core body, replacing
/// intermediate conduit ops with direct memref references (L1) or renaming
/// consumer-side references to the relay channel (MemTile).
static mlir::LogicalResult composeCoresBodies(FusableCorePair &pair,
                                              IntermediateRoute route,
                                              mlir::OpBuilder &builder,
                                              mlir::MLIRContext *ctx,
                                              AIE::DeviceOp device) {

  if (route == IntermediateRoute::Skip)
    return mlir::failure();

  AIE::CoreOp producerCore = pair.producerCore;
  AIE::CoreOp consumerCore = pair.consumerCore;
  Create intermediateConduit = pair.intermediateConduit;
  std::string channelName = intermediateConduit.getName().str();

  // Get the memref type for the intermediate buffer.
  mlir::Type elemType = intermediateConduit.getElementType();
  auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(elemType);
  if (!memrefTy)
    return mlir::failure();

  // Determine the consumer-side channel name. For fusion_group pairs where
  // the consumer uses a different channel name, find that name.
  std::string consumerChannelName = channelName;
  auto consumed = getConsumedChannels(consumerCore);
  for (const std::string &ch : consumed) {
    if (ch == channelName) {
      consumerChannelName = channelName;
      break;
    }
    // Check if this channel is connected via fusion_group.
    auto consIt = llvm::StringMap<Create>{};
    device.walk([&](Create op) { consIt[op.getName()] = op; });
    auto consCreateIt = consIt.find(ch);
    if (consCreateIt == consIt.end())
      continue;
    auto fgCons = consCreateIt->second.getFusionGroup();
    auto fgProd = intermediateConduit.getFusionGroup();
    if (fgProd && fgCons && !fgProd->empty() && *fgProd == *fgCons) {
      consumerChannelName = ch;
      break;
    }
  }

  // ---------------------------------------------------------------
  // Step A: Determine if we have loop bodies or flat bodies.
  // ---------------------------------------------------------------
  mlir::scf::ForOp producerFor = getOutermostFor(producerCore);
  mlir::scf::ForOp consumerFor = getOutermostFor(consumerCore);

  // Find the consumer for loop that directly contains the intermediate
  // acquire. For 2-level nesting (outer num_invocations + inner tile loop),
  // this is the inner for, not the outermost. For 1-level nesting this
  // stays equal to consumerFor.
  mlir::scf::ForOp consumerCloneFrom = consumerFor;
  if (consumerFor) {
    bool found = false;
    consumerCore.walk([&](Acquire acq) {
      if (found)
        return;
      if (acq.getName() == consumerChannelName &&
          acq.getPort() == Port::Consume) {
        mlir::Operation *parent = acq->getParentOp();
        while (parent && parent != consumerCore.getOperation()) {
          if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
            consumerCloneFrom = forOp;
            found = true;
            return;
          }
          parent = parent->getParentOp();
        }
      }
    });
  }

  // ---------------------------------------------------------------
  // Step B': Find the insertion point for consumer ops.
  // Consumer ops must be inserted at the same nesting level as the
  // intermediate produce/release, not at the outermost loop scope.
  // This must happen BEFORE erasing producer-side intermediate ops.
  // ---------------------------------------------------------------
  mlir::Operation *consumerInsertionPt = nullptr;
  producerCore.walk([&](Release rel) {
    if (auto acq = rel.getWindow().getDefiningOp<Acquire>()) {
      if (acq.getName() == channelName && acq.getPort() == Port::Produce)
        consumerInsertionPt = rel->getNextNode();
    }
  });

  // ---------------------------------------------------------------
  // Step B: Route-specific setup.
  // ---------------------------------------------------------------
  mlir::Block &producerBlock = getCoreBodyBlock(producerCore);
  mlir::Value allocVal;  // L1 route only.
  std::string relayName; // MemTile route only.

  if (route == IntermediateRoute::L1) {
    // Create memref.alloc at top of producer core body.
    builder.setInsertionPointToStart(&producerBlock);
    auto allocOp = mlir::memref::AllocaOp::create(
        builder, intermediateConduit.getLoc(), memrefTy);
    allocVal = allocOp.getResult();

    // Erase producer-side intermediate conduit ops and replace with alloc.
    llvm::SmallVector<SubviewAccess> prodSubviews;
    producerCore.walk([&](SubviewAccess op) {
      if (auto acqOp = op.getWindow().getDefiningOp<Acquire>()) {
        if (acqOp.getName() == channelName)
          prodSubviews.push_back(op);
      }
    });
    for (SubviewAccess sva : prodSubviews)
      sva.getResult().replaceAllUsesWith(allocVal);

    llvm::SmallVector<Release> prodReleases;
    producerCore.walk([&](Release op) {
      if (auto acqOp = op.getWindow().getDefiningOp<Acquire>()) {
        if (acqOp.getName() == channelName)
          prodReleases.push_back(op);
      }
    });

    llvm::SmallVector<Acquire> prodAcquires;
    producerCore.walk([&](Acquire op) {
      if (op.getName() == channelName && op.getPort() == Port::Produce)
        prodAcquires.push_back(op);
    });

    for (Release rel : prodReleases)
      rel->erase();
    for (SubviewAccess sva : prodSubviews)
      sva->erase();
    for (Acquire acq : prodAcquires) {
      if (!acq.getWindow().use_empty())
        acq.getWindow().replaceAllUsesWith(allocVal);
      acq->erase();
    }
  } else if (route == IntermediateRoute::MemTile) {
    // Emit relay channel and scatter op in the device body.
    // Producer-side conduit ops stay as-is (write to @intermediate).
    // Consumer-side ops will be renamed to @intermediate_relay during cloning.
    relayName = emitMemTileRelay(pair, device, builder, ctx);
  }

  // ---------------------------------------------------------------
  // Step C: Clone consumer core body ops into producer core.
  // ---------------------------------------------------------------

  // Set insertion point at the intermediate release site (found in Step B').
  // This ensures consumer ops are at the correct loop nesting level.
  if (consumerInsertionPt) {
    builder.setInsertionPoint(consumerInsertionPt);
  } else if (producerFor) {
    builder.setInsertionPoint(producerFor.getBody()->getTerminator());
  } else {
    builder.setInsertionPoint(producerBlock.getTerminator());
  }

  // Build IRMapping: map consumer induction vars to producer induction vars
  // at each nesting level. Walk from the clone-from level outward so that
  // 2-level nests (outer num_invocations + inner tile loop) get both IVs
  // mapped correctly.
  mlir::IRMapping mapping;
  if (consumerCloneFrom) {
    mlir::scf::ForOp enclosingFor;
    if (consumerInsertionPt) {
      mlir::Operation *parent = consumerInsertionPt->getParentOp();
      while (parent) {
        if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(parent)) {
          enclosingFor = forOp;
          break;
        }
        parent = parent->getParentOp();
      }
    }
    // Map IVs from the clone-from level outward through all nesting levels.
    mlir::scf::ForOp cFor = consumerCloneFrom;
    mlir::scf::ForOp pFor = enclosingFor ? enclosingFor : producerFor;
    while (cFor && pFor) {
      mapping.map(cFor.getInductionVar(), pFor.getInductionVar());
      cFor = cFor->getParentOfType<mlir::scf::ForOp>();
      pFor = pFor->getParentOfType<mlir::scf::ForOp>();
    }
  }

  // For loop bodies, pre-clone any ops defined OUTSIDE the consumer's
  // clone-from for but used INSIDE it (e.g., arith.constant ops defined
  // in an outer for body or the core body). This ensures the IRMapping
  // has entries for these values before we clone the for body.
  if (consumerCloneFrom) {
    llvm::DenseSet<mlir::Operation *> alreadyCloned;
    for (mlir::Operation &innerOp : *consumerCloneFrom.getBody()) {
      for (mlir::Value operand : innerOp.getOperands()) {
        mlir::Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;
        // If defined outside the clone-from for body and not already in
        // the mapping, clone it into the producer.
        if (!consumerCloneFrom->isProperAncestor(defOp) &&
            defOp != consumerCloneFrom.getOperation() &&
            !mapping.contains(operand) && alreadyCloned.insert(defOp).second) {
          builder.clone(*defOp, mapping);
        }
      }
    }
  }

  // Collect ops to clone from the consumer's clone-from for body (inner
  // for when nested, outermost for when single-level).
  llvm::SmallVector<mlir::Operation *> opsToClone;
  if (consumerCloneFrom) {
    for (mlir::Operation &op : *consumerCloneFrom.getBody()) {
      if (op.hasTrait<mlir::OpTrait::IsTerminator>())
        continue;
      opsToClone.push_back(&op);
    }
  } else {
    for (mlir::Operation &op : consumerCore.getBody().front()) {
      if (op.hasTrait<mlir::OpTrait::IsTerminator>())
        continue;
      opsToClone.push_back(&op);
    }
  }

  // Clone each consumer op into the producer, handling intermediate channel
  // ops according to the routing decision.
  auto relayNameAttr = route == IntermediateRoute::MemTile
                           ? mlir::FlatSymbolRefAttr::get(ctx, relayName)
                           : mlir::FlatSymbolRefAttr{};

  for (mlir::Operation *op : opsToClone) {
    if (auto acqOp = mlir::dyn_cast<Acquire>(op)) {
      if (acqOp.getName() == consumerChannelName) {
        if (route == IntermediateRoute::L1) {
          // L1: map window result to allocVal, skip the acquire.
          mapping.map(acqOp.getWindow(), allocVal);
          continue;
        }
        // MemTile: clone but rename to relay channel.
        mlir::Operation *cloned = builder.clone(*op, mapping);
        cloned->setAttr("name", relayNameAttr);
        mapping.map(acqOp.getWindow(), cloned->getResult(0));
        continue;
      }
    }
    if (auto relOp = mlir::dyn_cast<Release>(op)) {
      if (auto acqDef = relOp.getWindow().getDefiningOp<Acquire>()) {
        if (acqDef.getName() == consumerChannelName) {
          if (route == IntermediateRoute::L1)
            continue; // L1: skip release of intermediate.
          // MemTile: clone (window operand already mapped to relay acquire).
          builder.clone(*op, mapping);
          continue;
        }
      }
    }
    if (auto svaOp = mlir::dyn_cast<SubviewAccess>(op)) {
      if (auto acqDef = svaOp.getWindow().getDefiningOp<Acquire>()) {
        if (acqDef.getName() == consumerChannelName) {
          if (route == IntermediateRoute::L1) {
            // L1: map subview result to allocVal, skip.
            mapping.map(svaOp.getResult(), allocVal);
            continue;
          }
          // MemTile: clone (window operand mapped to relay acquire result).
          builder.clone(*op, mapping);
          continue;
        }
      }
    }

    // Clone the op with mapping.
    builder.clone(*op, mapping);
  }

  // ---------------------------------------------------------------
  // Step D: (L1 uses alloca — no dealloc needed.)
  // ---------------------------------------------------------------

  return mlir::success();
}

// ---------------------------------------------------------------------------
// Step 4: Clean up dead ops
// ---------------------------------------------------------------------------

/// Remove a token operand from a wait_all or wait_all_async op by rebuilding
/// the op without that operand. If no operands remain, erase the op entirely
/// (and recursively handle wait_all_async's own token users).
static void removeTokenFromWaitOp(mlir::Operation *waitOp,
                                  mlir::Value deadToken,
                                  mlir::OpBuilder &builder) {
  // Collect surviving operands.
  llvm::SmallVector<mlir::Value> surviving;
  for (mlir::Value tok : waitOp->getOperands()) {
    if (tok != deadToken)
      surviving.push_back(tok);
  }

  if (surviving.empty()) {
    // No remaining operands — erase the wait op.
    // If it's a wait_all_async, its result token may have users.
    if (waitOp->getNumResults() > 0) {
      mlir::Value resultToken = waitOp->getResult(0);
      // Recursively remove from downstream waits.
      llvm::SmallVector<mlir::Operation *> users(resultToken.getUsers().begin(),
                                                 resultToken.getUsers().end());
      for (mlir::Operation *user : users)
        removeTokenFromWaitOp(user, resultToken, builder);
    }
    waitOp->erase();
    return;
  }

  if (surviving.size() == waitOp->getNumOperands())
    return; // Nothing changed.

  // Rebuild the wait op with fewer operands.
  builder.setInsertionPoint(waitOp);
  if (waitOp->getName().getStringRef() == "conduit.wait_all") {
    auto newWait = builder.create<WaitAll>(waitOp->getLoc(), surviving);
    (void)newWait;
    waitOp->erase();
  } else if (waitOp->getName().getStringRef() == "conduit.wait_all_async") {
    auto newWait = builder.create<WaitAllAsync>(
        waitOp->getLoc(), waitOp->getResult(0).getType(), surviving);
    waitOp->getResult(0).replaceAllUsesWith(newWait.getResult());
    waitOp->erase();
  }
}

/// Safely erase a put_memref_async or get_memref_async op by first handling
/// any downstream wait_all / wait_all_async consumers of its token result.
static void safeEraseAsyncOp(mlir::Operation *asyncOp,
                             mlir::OpBuilder &builder) {
  if (asyncOp->getNumResults() == 0) {
    asyncOp->erase();
    return;
  }

  mlir::Value token = asyncOp->getResult(0);
  llvm::SmallVector<mlir::Operation *> users(token.getUsers().begin(),
                                             token.getUsers().end());
  for (mlir::Operation *user : users)
    removeTokenFromWaitOp(user, token, builder);

  asyncOp->erase();
}

/// Delete the intermediate conduit.create, dead consumer core, and associated
/// runtime sequence and shim DMA ops. Also cleans up the consumer channel
/// (different name) if the pair was matched via fusion_group.
static void cleanUpDeadOps(FusableCorePair &pair, AIE::DeviceOp device,
                           mlir::OpBuilder &builder,
                           const std::string &consumerChannelName) {
  std::string channelName = pair.intermediateConduit.getName().str();

  // Collect both channel names to clean up (may be the same).
  llvm::SmallVector<std::string> deadNames;
  deadNames.push_back(channelName);
  if (consumerChannelName != channelName)
    deadNames.push_back(consumerChannelName);

  // 1. Erase consumer core's aie.core op (body already merged).
  pair.consumerCore->erase();

  // 2. Erase intermediate conduit.create op(s).
  pair.intermediateConduit->erase();
  if (consumerChannelName != channelName) {
    // Also erase the consumer-side conduit.create.
    // Collect first, then erase — walk is not safe for erasure.
    llvm::SmallVector<mlir::Operation *> consCreates;
    device.walk([&](Create op) {
      if (op.getName() == consumerChannelName)
        consCreates.push_back(op.getOperation());
    });
    for (auto *op : consCreates)
      op->erase();
  }

  // 3. Walk device for aie.shim_dma_allocation referencing dead channels.
  llvm::SmallVector<mlir::Operation *> toErase;
  for (const std::string &name : deadNames) {
    std::string shimAllocName = name + "_shim_alloc";
    device.walk([&](AIE::ShimDMAAllocationOp alloc) {
      llvm::StringRef sym = alloc.getSymName();
      if (sym == name || sym == shimAllocName)
        toErase.push_back(alloc.getOperation());
    });
  }
  for (auto *op : toErase)
    op->erase();

  // 4. Walk runtime_sequence for conduit.put_memref / conduit.get_memref
  //    (blocking and async) referencing dead channel names.
  //    For async variants, safely remove token uses from downstream waits.
  for (const std::string &name : deadNames) {
    std::string shimAllocName = name + "_shim_alloc";
    llvm::SmallVector<mlir::Operation *> dmaOps;
    device.walk([&](mlir::Operation *op) {
      llvm::StringRef opName = op->getName().getStringRef();
      if (opName != "conduit.put_memref" && opName != "conduit.get_memref" &&
          opName != "conduit.put_memref_async" &&
          opName != "conduit.get_memref_async")
        return;
      auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
      if (nameAttr &&
          (nameAttr.getValue() == name || nameAttr.getValue() == shimAllocName))
        dmaOps.push_back(op);
    });
    for (mlir::Operation *op : dmaOps) {
      if (op->getNumResults() > 0) {
        // Async variant — handle token users before erasing.
        safeEraseAsyncOp(op, builder);
      } else {
        // Blocking variant — safe to erase directly.
        op->erase();
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Step 5: Merge link_with from consumer into producer core
// ---------------------------------------------------------------------------

/// Merge link_with / link_files from the consumer core into the producer core.
/// After loop-body fusion, the producer core body calls functions from both
/// the producer's and consumer's kernel object files.  Collect all object
/// file paths from both cores, de-duplicate, and set link_files (StrArrayAttr)
/// on the producer core.  The deprecated link_with (single string) is removed
/// to avoid the "both specified" verifier error.
static void mergeLinkWith(AIE::CoreOp producerCore, AIE::CoreOp consumerCore,
                          mlir::MLIRContext *ctx) {
  llvm::SmallVector<std::string> allPaths;
  llvm::StringSet<> seen;

  auto addPath = [&](llvm::StringRef path) {
    llvm::StringRef trimmed = path.trim();
    if (!trimmed.empty() && seen.insert(trimmed).second)
      allPaths.push_back(trimmed.str());
  };

  // Helper to split a link_with value on commas (handles legacy
  // comma-separated values from a previous fusion invocation).
  auto addLinkWith = [&](llvm::StringRef lw) {
    llvm::SmallVector<llvm::StringRef> parts;
    lw.split(parts, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (llvm::StringRef p : parts)
      addPath(p);
  };

  // Helper to add entries from a link_files array attribute.
  auto addLinkFiles = [&](mlir::ArrayAttr arr) {
    for (mlir::Attribute a : arr) {
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(a))
        addPath(s.getValue());
    }
  };

  // Collect from producer.
  if (auto lw = producerCore.getLinkWith())
    addLinkWith(*lw);
  if (auto lf = producerCore.getLinkFiles())
    addLinkFiles(*lf);

  // Collect from consumer.
  if (auto lw = consumerCore.getLinkWith())
    addLinkWith(*lw);
  if (auto lf = consumerCore.getLinkFiles())
    addLinkFiles(*lf);

  if (allPaths.empty())
    return;

  // If only one path and it already matches the producer's, no change needed.
  if (allPaths.size() == 1 && producerCore.getLinkWith() &&
      *producerCore.getLinkWith() == allPaths[0])
    return;

  // Set link_files (StrArrayAttr) — one entry per object file.
  llvm::SmallVector<mlir::Attribute> fileAttrs;
  for (const std::string &p : allPaths)
    fileAttrs.push_back(mlir::StringAttr::get(ctx, p));
  producerCore.setLinkFilesAttr(mlir::ArrayAttr::get(ctx, fileAttrs));

  // Remove deprecated link_with to avoid "both specified" verifier error.
  if (producerCore.getLinkWith())
    producerCore->removeAttr("link_with");
}

// ---------------------------------------------------------------------------
// Step 6: (Removed — see note in runOnOperation.)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Main pass struct
// ---------------------------------------------------------------------------

struct ConduitFuseCoreBodyPass
    : public impl::ConduitFuseCoreBodiesBase<ConduitFuseCoreBodyPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();
    mlir::OpBuilder builder(ctx);

    // Step 0: Merge cross-device fusion_group connections into single devices.
    // This allows findFusableCorePairs to detect pairs that span separate
    // aie.device blocks (e.g., GEMV in device A → SiLU in device B).
    if (mlir::failed(mergeDevicesForFusion(module, ctx))) {
      signalPassFailure();
      return;
    }

    // Process each aie.device independently.
    module.walk([&](AIE::DeviceOp device) {
      // Track erased cores and conduits across all fusion iterations.
      llvm::DenseSet<mlir::Operation *> deadCores;
      llvm::DenseSet<mlir::Operation *> deadConduits;

      // Fuse-one-then-restart: after each successful fusion, re-run
      // findFusableCorePairs on the updated IR so chained edges (A→B→C)
      // pick up the fused A+B core as the new producer for the B→C edge.
      bool fusionHappened = true;
      while (fusionHappened) {
        fusionHappened = false;

        // Re-infer tile coordinates after each fusion (IR has changed).
        auto inferredMap = inferAllTiles(module);

        // Step 1: Find fusable core pairs (re-run each iteration).
        llvm::SmallVector<FusableCorePair> pairs =
            findFusableCorePairs(device, inferredMap);

        if (pairs.empty())
          break;

        for (FusableCorePair &pair : pairs) {
          // Skip stale pairs where a core or conduit was already erased.
          if (deadCores.count(pair.producerCore.getOperation()) ||
              deadCores.count(pair.consumerCore.getOperation()) ||
              deadConduits.count(pair.intermediateConduit.getOperation()))
            continue;

          // Step 2: Decide intermediate routing.
          IntermediateRoute route =
              decideRoute(pair.intermediateConduit, pair.tile, device);

          if (route == IntermediateRoute::Skip) {
            pair.intermediateConduit.emitRemark()
                << "conduit-fuse-core-bodies: skipping fusion — intermediate "
                   "too large for L1 and MemTile";
            continue;
          }

          // Determine consumer channel name (may differ from producer
          // channel name when matched via fusion_group).
          std::string channelName = pair.intermediateConduit.getName().str();
          std::string consumerChannelName = channelName;
          auto consumed = getConsumedChannels(pair.consumerCore);
          for (const std::string &ch : consumed) {
            if (ch == channelName)
              break;
            // Check fusion_group match.
            llvm::StringMap<Create> cMap;
            device.walk([&](Create op) { cMap[op.getName()] = op; });
            auto consIt = cMap.find(ch);
            if (consIt == cMap.end())
              continue;
            auto fgCons = consIt->second.getFusionGroup();
            auto fgProd = pair.intermediateConduit.getFusionGroup();
            if (fgProd && fgCons && !fgProd->empty() && *fgProd == *fgCons) {
              consumerChannelName = ch;
              break;
            }
          }

          // Step 3: Compose core bodies.
          if (mlir::failed(
                  composeCoresBodies(pair, route, builder, ctx, device))) {
            pair.producerCore.emitWarning()
                << "conduit-fuse-core-bodies: failed to compose core bodies";
            continue;
          }

          // Step 5: Merge link_with / link_files from consumer into producer.
          mergeLinkWith(pair.producerCore, pair.consumerCore, ctx);

          // Record dead ops before cleanup erases them.
          deadCores.insert(pair.consumerCore.getOperation());
          deadConduits.insert(pair.intermediateConduit.getOperation());

          // Step 4: Clean up dead ops.
          cleanUpDeadOps(pair, device, builder, consumerChannelName);

          // Restart: break out of the pairs loop and re-run
          // findFusableCorePairs on the updated IR.
          fusionHappened = true;
          break;
        }
      }

      // Step 6: Preserve runtime_sequence block arguments.
      // conduit.put_memref/get_memref ops don't reference block args as SSA
      // operands (buffer info is in attributes), so all args appear unused.
      // However, they are positionally significant — the host passes buffer
      // addresses at these positions.  Removing them breaks the host
      // invocation signature.  Leave cleanup to downstream passes.
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitFuseCoreBodyPass() {
  return std::make_unique<ConduitFuseCoreBodyPass>();
}

} // namespace xilinx::conduit
