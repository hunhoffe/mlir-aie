//===- ConduitDepthPromotion.cpp - conduit-depth-promote pass ----*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// NOTE: This pass is experimental and not validated on hardware. Use
// --conduit-depth-promote only as an opt-in flag.
//
// Two-phase pass that runs after Pass A/B and before Pass C:
//
// Phase 1 — Sentinel resolver (always runs):
//   Pass A/B emit depth=0 for channels whose depth is unknown at translation
//   time.  This phase resolves depth=0 to depth=1 (single-buffering minimum),
//   or to the CSDFa minimum depth when --conduit-depth-promote{csdf=true} and
//   producer_rates/consumer_rates are present.
//
// Phase 2 — Depth promotion (opt-in heuristic):
//   Promotes eligible depth-1 conduits to depth-2 (double-buffering) to enable
//   compute-DMA overlap.
//
// Exclusion criteria (any one disqualifies):
//   0. Cascade conduit (routing_mode = "cascade") — depth is architecturally
//      fixed at 1; hardware has no FIFO, only a blocking register.
//   1. CSDF / cyclostatic access pattern present
//   2. Linked conduit (appears in conduit.link srcs or dsts)
//   3. No surrounding loop (no overlap benefit without iteration)
//   4. Passthrough-only (acquire immediately followed by release, no compute)
//   5. Non-uniform acquire/release counts across uses
//   6. Memory budget exceeded on target tile
//   7. AIE1 lock budget exceeded (AIE1 has 16 locks per tile)
//   8. BD budget exceeded (each tile has limited BD slots)
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "ConduitTileInference.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITDEPTHPROMOTE
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Constants for budget heuristics
// ---------------------------------------------------------------------------

// AIE1 has 16 locks per tile.  Each depth-2 conduit uses 2 locks (prod+cons),
// so promoting adds 1 lock.  We refuse if the tile already uses >= this many.
static constexpr int64_t kAIE1MaxLocksPerTile = 16;

// Maximum number of BD slots per tile.  Promoting adds 1 BD per consumer.
static constexpr int64_t kMaxBDSlotsPerTile = 16;

// Memory budget per compute tile (bytes).  Promoting doubles buffer usage.
// 32 KiB is the typical AIE tile data memory size.
static constexpr int64_t kDefaultTileMemoryBytes = 32 * 1024;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect conduit names that appear in any relay op (src/dst of scatter,
/// gather, transpose).
static llvm::StringSet<> collectLinkedConduitNames(mlir::ModuleOp module) {
  llvm::StringSet<> linked;
  auto collect = [&](mlir::Operation *op) {
    // Array attrs: scatter.dsts, gather.srcs.
    if (auto srcsAttr = op->getAttrOfType<mlir::ArrayAttr>("srcs"))
      for (auto s : srcsAttr)
        if (auto str = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(s))
          linked.insert(str.getValue());
    if (auto dstsAttr = op->getAttrOfType<mlir::ArrayAttr>("dsts"))
      for (auto d : dstsAttr)
        if (auto str = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(d))
          linked.insert(str.getValue());
    // Scalar attrs: scatter.src (single), gather.dst (single).
    if (auto srcAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("src"))
      linked.insert(srcAttr.getValue());
    if (auto dstAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("dst"))
      linked.insert(dstAttr.getValue());
  };
  // Sprint 3+ relay ops.
  module.walk([&](ScatterOp op) { collect(op.getOperation()); });
  module.walk([&](GatherOp op) { collect(op.getOperation()); });
  module.walk([&](TransposeOp op) { collect(op.getOperation()); });
  return linked;
}

/// Check if an operation is inside a loop-like construct.
static bool isInsideLoop(mlir::Operation *op) {
  mlir::Operation *parent = op->getParentOp();
  while (parent) {
    if (mlir::isa<mlir::LoopLikeOpInterface>(parent))
      return true;
    parent = parent->getParentOp();
  }
  return false;
}

/// Check if an acquire is "passthrough" -- the window result has no
/// subview_access users (i.e., no compute consumes the buffer).
static bool isPassthroughAcquire(mlir::Operation *acqOp) {
  if (acqOp->getNumResults() == 0)
    return true; // no result -> trivially passthrough
  mlir::Value window = acqOp->getResult(0);
  for (mlir::Operation *user : window.getUsers())
    if (mlir::isa<SubviewAccess>(user))
      return false; // has a subview user — not passthrough
  return true;      // no subview users — is passthrough
}

/// Estimate single-slot buffer size in bytes from element type.
static int64_t estimateSingleSlotBytes(mlir::Type elemType) {
  auto mref = mlir::dyn_cast<mlir::MemRefType>(elemType);
  if (!mref)
    return 4; // default 4 bytes per element
  int64_t elemBits = mref.getElementTypeBitWidth();
  int64_t elemCount = 1;
  for (int64_t d : mref.getShape()) {
    if (mlir::ShapedType::isDynamic(d))
      return 4; // can't determine statically
    elemCount *= d;
  }
  return (elemBits / 8) * elemCount;
}

// ---------------------------------------------------------------------------
// Main pass
// ---------------------------------------------------------------------------

struct ConduitDepthPromotePass
    : impl::ConduitDepthPromoteBase<ConduitDepthPromotePass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    mlir::OpBuilder builder(module.getContext());

    // Step 1: Collect linked conduit names (exclusion criterion #2).
    auto linkedNames = collectLinkedConduitNames(module);

    // Step 1.5: Resolve depth=0 sentinel.
    // Pass A/B emit depth=0 for channels where depth is unknown at translation
    // time.  Resolve to the CSDFa minimum depth (if --conduit-depth-promote
    // {csdf=true} and producer_rates/consumer_rates are present), or to depth=1
    // (safe single-buffering minimum) otherwise.  After resolution the standard
    // promotion loop (Step 5) may further promote depth=1 → depth=2+.
    module.walk([&](Create op) {
      auto depthAttr = op->getAttrOfType<mlir::IntegerAttr>("depth");
      if (!depthAttr || depthAttr.getInt() != 0)
        return;
      // Cascade channels are architecturally fixed at depth=1; pass the
      // sentinel through as 1 and let Criterion 0 in Step 5 prevent promotion.
      if (auto rm = op.getRoutingMode()) {
        if (*rm == RoutingMode::Cascade) {
          op->setAttr("depth", builder.getI64IntegerAttr(1));
          return;
        }
      }
      int64_t resolvedDepth = 1; // safe default
      if (csdfa) {
        auto prodRates =
            op->getAttrOfType<mlir::DenseI64ArrayAttr>("producer_rates");
        auto consRates =
            op->getAttrOfType<mlir::DenseI64ArrayAttr>("consumer_rates");
        if (prodRates && consRates) {
          int64_t P = 0, C = 0;
          for (int64_t r : prodRates.asArrayRef())
            P += r;
          for (int64_t r : consRates.asArrayRef())
            C += r;
          if (P > 0 && C > 0) {
            int64_t maxPC = std::max(P, C);
            int64_t minPC = std::min(P, C);
            resolvedDepth = static_cast<int64_t>(std::ceil(
                static_cast<double>(maxPC) * eta / static_cast<double>(minPC)));
            if (resolvedDepth < 1)
              resolvedDepth = 1;
            op->emitRemark("conduit-depth-promote: sentinel depth=0 → depth=")
                << resolvedDepth << " (CSDFa) for '" << op.getSymName() << "'";
          }
        }
      }
      op->setAttr("depth", builder.getI64IntegerAttr(resolvedDepth));
    });

    // Step 2: Collect all conduit.create ops with depth == 1.
    llvm::SmallVector<mlir::Operation *> candidates;
    module.walk([&](Create op) {
      auto depthAttr = op->getAttrOfType<mlir::IntegerAttr>("depth");
      if (!depthAttr || depthAttr.getInt() != 1)
        return;
      candidates.push_back(op.getOperation());
    });

    if (candidates.empty())
      return;

    // Step 3: Collect acquire/release ops (Tier 2) and put/get_memref ops
    // (Tier 3) per conduit name for uniformity checks (exclusion criteria
    // #3, #4, #5).
    //
    // Tier 2 (ObjectFIFO-originated): conduit.acquire/release
    // Tier 3 (air.channel-originated):
    // conduit.put_memref[_async]/get_memref[_async]
    //
    // Separate maps for Tier 3 num_elems to avoid cross-tier confusion
    // (a cross-tier channel can have both acquire{count=1} and
    // put_memref{num_elems=64} — mixing them would break uniformity checks).
    llvm::StringMap<llvm::SmallVector<int64_t>> acquireCounts;
    llvm::StringMap<llvm::SmallVector<int64_t>> releaseCounts;
    llvm::StringMap<llvm::SmallVector<int64_t>> putMemrefNumElems;
    llvm::StringMap<llvm::SmallVector<int64_t>> getMemrefNumElems;
    llvm::StringMap<bool> nameHasLoopAcquire;
    llvm::StringMap<bool> nameAllPassthrough;

    module.walk([&](mlir::Operation *op) {
      // --- Tier 2: acquire/release ---
      if (mlir::isa<Acquire, AcquireAsync>(op)) {
        auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
        auto countAttr = op->getAttrOfType<mlir::IntegerAttr>("count");
        if (!nameAttr || !countAttr)
          return;
        llvm::StringRef name = nameAttr.getValue();
        acquireCounts[name].push_back(countAttr.getInt());
        if (isInsideLoop(op))
          nameHasLoopAcquire[name] = true;

        // Check passthrough
        if (nameAllPassthrough.find(name) == nameAllPassthrough.end())
          nameAllPassthrough[name] = true; // assume true until proven false
        if (!isPassthroughAcquire(op))
          nameAllPassthrough[name] = false;
      }
      if (mlir::isa<Release, ReleaseAsync>(op)) {
        auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
        auto countAttr = op->getAttrOfType<mlir::IntegerAttr>("count");
        if (!nameAttr || !countAttr)
          return;
        releaseCounts[nameAttr.getValue()].push_back(countAttr.getInt());
      }

      // --- Tier 3: put_memref[_async] / get_memref[_async] ---
      if (mlir::isa<PutMemref, PutMemrefAsync>(op)) {
        auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
        if (!nameAttr)
          return;
        llvm::StringRef name = nameAttr.getValue();
        if (auto numElemsAttr =
                op->getAttrOfType<mlir::IntegerAttr>("num_elems"))
          putMemrefNumElems[name].push_back(numElemsAttr.getInt());
        if (isInsideLoop(op))
          nameHasLoopAcquire[name] = true;
        // Tier 3 ops always perform real DMA work — never passthrough.
        nameAllPassthrough[name] = false;
      }
      if (mlir::isa<GetMemref, GetMemrefAsync>(op)) {
        auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
        if (!nameAttr)
          return;
        llvm::StringRef name = nameAttr.getValue();
        if (auto numElemsAttr =
                op->getAttrOfType<mlir::IntegerAttr>("num_elems"))
          getMemrefNumElems[name].push_back(numElemsAttr.getInt());
        if (isInsideLoop(op))
          nameHasLoopAcquire[name] = true;
        nameAllPassthrough[name] = false;
      }
    });

    // Infer tile coordinates from IR structure for budget checks.
    auto inferredMap = inferAllTiles(module);
    auto extractCoord = [](mlir::Value tileVal) -> std::pair<int64_t, int64_t> {
      if (auto tileOp = tileVal.getDefiningOp<AIE::TileOp>())
        return {static_cast<int64_t>(tileOp.getCol()),
                static_cast<int64_t>(tileOp.getRow())};
      return {-1, -1};
    };

    // Step 4: Per-tile resource counters for budget checks.
    // key = (col, row) packed as int64_t
    auto tileKey = [](int64_t col, int64_t row) -> int64_t {
      return (col << 32) | (row & 0xFFFFFFFF);
    };
    llvm::DenseMap<int64_t, int64_t> tileLockCount;
    llvm::DenseMap<int64_t, int64_t> tileBDCount;
    llvm::DenseMap<int64_t, int64_t> tileMemUsed;

    // Pre-populate from existing conduit.create ops using inferred tiles.
    module.walk([&](Create op) {
      // Cascade conduits use no buffers, locks, or BDs — skip resource
      // counting.
      if (auto rm = op.getRoutingMode())
        if (*rm == RoutingMode::Cascade)
          return;

      auto depthAttr = op->getAttrOfType<mlir::IntegerAttr>("depth");
      int64_t depth = depthAttr ? depthAttr.getInt() : 1;
      auto capAttr = op->getAttrOfType<mlir::IntegerAttr>("slot_elems");
      auto elemTypeAttr = op->getAttrOfType<mlir::TypeAttr>("element_type");

      // Get consumer tile coords: prefer inference, fallback to attribute
      // for channels outside aie.core (e.g. in func.func or hand-written IR).
      llvm::SmallVector<std::pair<int64_t, int64_t>> consCoords;
      auto tileIt = inferredMap.find(op.getName().str());
      if (tileIt != inferredMap.end() && !tileIt->second.consumerTiles.empty()) {
        for (mlir::Value tv : tileIt->second.consumerTiles) {
          auto [col, row] = extractCoord(tv);
          if (col >= 0)
            consCoords.push_back({col, row});
        }
      }

      // Estimate per-consumer resources.
      for (auto [col, row] : consCoords) {
        int64_t key = tileKey(col, row);
        tileLockCount[key] += 2; // prod + cons lock pair
        tileBDCount[key] += depth;
        if (capAttr && elemTypeAttr) {
          int64_t perSlotBytes =
              estimateSingleSlotBytes(elemTypeAttr.getValue());
          tileMemUsed[key] += perSlotBytes * depth;
        }
      }
      // Producer tile also uses resources for non-shim.
      std::pair<int64_t, int64_t> prodCoord = {-1, -1};
      if (tileIt != inferredMap.end() && tileIt->second.producerTile) {
        prodCoord = extractCoord(tileIt->second.producerTile);
      }
      if (prodCoord.first >= 0 && prodCoord.second != 0) { // non-shim
        int64_t key = tileKey(prodCoord.first, prodCoord.second);
        tileLockCount[key] += 2;
        tileBDCount[key] += depth;
      }
    });

    // Detect AIE1 vs AIE2 from device op.
    bool isAIE1 = false;
    module.walk([&](xilinx::AIE::DeviceOp deviceOp) {
      const AIE::AIETargetModel &tm = AIE::getTargetModel(deviceOp);
      isAIE1 = (tm.getTargetArch() == AIE::AIEArch::AIE1);
    });

    // Step 5: Evaluate each candidate.
    int promoted = 0;

    for (mlir::Operation *createOp : candidates) {
      auto typedCreate = mlir::dyn_cast<Create>(createOp);
      if (!typedCreate)
        continue;
      llvm::StringRef name = typedCreate.getSymName();
      if (name.empty())
        continue;

      // Criterion 0: cascade conduits — depth is architecturally fixed at 1.
      // The cascade stream is a hardware register (rendezvous channel), not a
      // FIFO. Promoting to depth-2 would emit an incorrect depth attribute that
      // Pass C cannot implement. Skip silently; do not emit a remark (this is
      // expected).
      if (auto typedOp = mlir::dyn_cast<Create>(createOp)) {
        if (auto rm = typedOp.getRoutingMode())
          if (*rm == RoutingMode::Cascade)
            continue;
      }

      // Criterion 1: CSDF / cyclostatic access pattern.
      if (createOp->getAttrOfType<mlir::DenseI64ArrayAttr>("access_pattern")) {
        createOp->emitRemark("conduit-depth-promote: skipping '")
            << name << "' -- CSDF access pattern";
        continue;
      }

      // Determine target depth.  Default: fixed depth-2 heuristic.
      // When --conduit-depth-promote{csdf=true} is set and both
      // producer_rates/consumer_rates are present, use the CSDFa minimum
      // buffer depth formula (Denolf 2007, Koek 2016 §4):
      //
      //   min_depth = ceil(max(P, C) * η / min(P, C))
      //
      // where P = sum(producer_rates), C = sum(consumer_rates),
      // η = target efficiency (default 1.0 = stall-free).
      int64_t targetDepth = 2;
      auto prodRatesAttr =
          createOp->getAttrOfType<mlir::DenseI64ArrayAttr>("producer_rates");
      auto consRatesAttr =
          createOp->getAttrOfType<mlir::DenseI64ArrayAttr>("consumer_rates");
      if (prodRatesAttr || consRatesAttr) {
        if (!csdfa || !prodRatesAttr || !consRatesAttr) {
          // Flag disabled or incomplete rate annotation — skip as before.
          createOp->emitRemark("conduit-depth-promote: skipping '")
              << name << "' -- CSDF rates present";
          continue;
        }
        // Compute P = sum(producer_rates), C = sum(consumer_rates).
        int64_t P = 0, C = 0;
        for (int64_t r : prodRatesAttr.asArrayRef())
          P += r;
        for (int64_t r : consRatesAttr.asArrayRef())
          C += r;
        if (P <= 0 || C <= 0) {
          createOp->emitRemark("conduit-depth-promote: skipping '")
              << name << "' -- zero or negative rate sum";
          continue;
        }
        // CSDFa minimum buffer depth formula.
        int64_t maxPC = std::max(P, C);
        int64_t minPC = std::min(P, C);
        double numerator = static_cast<double>(maxPC) * eta;
        targetDepth = static_cast<int64_t>(
            std::ceil(numerator / static_cast<double>(minPC)));
        if (targetDepth <= 1) {
          // Rates are balanced and η ≤ 1 — no promotion needed.
          continue;
        }
      }

      // Criterion 2: linked conduit.
      if (linkedNames.count(name)) {
        createOp->emitRemark("conduit-depth-promote: skipping '")
            << name << "' -- linked conduit";
        continue;
      }

      // Criterion 3: no surrounding loop.
      // conduit.create is always at device-body level — never inside a loop.
      // We check whether any acquire for this conduit name is inside a loop.
      if (!nameHasLoopAcquire.count(name) || !nameHasLoopAcquire[name]) {
        createOp->emitRemark("conduit-depth-promote: skipping '")
            << name << "' -- no loop context";
        continue;
      }

      // Criterion 4: passthrough-only.
      if (nameAllPassthrough.count(name) && nameAllPassthrough[name]) {
        createOp->emitRemark("conduit-depth-promote: skipping '")
            << name << "' -- passthrough-only (no compute)";
        continue;
      }

      // Criterion 5: non-uniform acquire/release counts.
      bool uniform = true;
      if (acquireCounts.count(name)) {
        auto &counts = acquireCounts[name];
        if (!counts.empty()) {
          int64_t first = counts[0];
          for (int64_t c : counts) {
            if (c != first) {
              uniform = false;
              break;
            }
          }
        }
      }
      if (uniform && releaseCounts.count(name)) {
        auto &counts = releaseCounts[name];
        if (!counts.empty()) {
          int64_t first = counts[0];
          for (int64_t c : counts) {
            if (c != first) {
              uniform = false;
              break;
            }
          }
        }
      }
      // Also check Tier 3 num_elems uniformity.
      if (uniform && putMemrefNumElems.count(name)) {
        auto &nums = putMemrefNumElems[name];
        if (!nums.empty()) {
          int64_t first = nums[0];
          for (int64_t n : nums) {
            if (n != first) {
              uniform = false;
              break;
            }
          }
        }
      }
      if (uniform && getMemrefNumElems.count(name)) {
        auto &nums = getMemrefNumElems[name];
        if (!nums.empty()) {
          int64_t first = nums[0];
          for (int64_t n : nums) {
            if (n != first) {
              uniform = false;
              break;
            }
          }
        }
      }
      if (!uniform) {
        createOp->emitRemark("conduit-depth-promote: skipping '")
            << name << "' -- non-uniform acquire/release counts";
        continue;
      }

      // Look up consumer tile coordinates: prefer inference, fallback to
      // attribute for channels outside aie.core.
      llvm::SmallVector<std::pair<int64_t, int64_t>> consCoords;
      {
        auto tileIt = inferredMap.find(name);
        if (tileIt != inferredMap.end() &&
            !tileIt->second.consumerTiles.empty()) {
          for (mlir::Value tv : tileIt->second.consumerTiles) {
            auto [col, row] = extractCoord(tv);
            if (col >= 0)
              consCoords.push_back({col, row});
          }
        }
      }

      // Criterion 6: memory budget.
      auto capAttr = createOp->getAttrOfType<mlir::IntegerAttr>("slot_elems");
      auto elemTypeAttr =
          createOp->getAttrOfType<mlir::TypeAttr>("element_type");
      bool memOverBudget = false;
      if (!consCoords.empty() && capAttr && elemTypeAttr) {
        int64_t bufBytes = estimateSingleSlotBytes(elemTypeAttr.getValue());
        for (auto [col, row] : consCoords) {
          int64_t key = tileKey(col, row);
          if (tileMemUsed[key] + bufBytes * targetDepth >
              kDefaultTileMemoryBytes) {
            memOverBudget = true;
            break;
          }
        }
      }
      if (memOverBudget) {
        createOp->emitRemark("conduit-depth-promote: skipping '")
            << name << "' -- memory budget exceeded";
        continue;
      }

      // Criterion 7: AIE1 lock budget.
      if (isAIE1 && !consCoords.empty()) {
        bool lockOverBudget = false;
        for (auto [col, row] : consCoords) {
          int64_t key = tileKey(col, row);
          if (tileLockCount[key] + 1 > kAIE1MaxLocksPerTile) {
            lockOverBudget = true;
            break;
          }
        }
        if (lockOverBudget) {
          createOp->emitRemark("conduit-depth-promote: skipping '")
              << name << "' -- AIE1 lock budget exceeded";
          continue;
        }
      }

      // Criterion 8: BD budget.
      if (!consCoords.empty()) {
        bool bdOverBudget = false;
        for (auto [col, row] : consCoords) {
          int64_t key = tileKey(col, row);
          if (tileBDCount[key] + (targetDepth - 1) > kMaxBDSlotsPerTile) {
            bdOverBudget = true;
            break;
          }
        }
        if (bdOverBudget) {
          createOp->emitRemark("conduit-depth-promote: skipping '")
              << name << "' -- BD budget exceeded";
          continue;
        }
      }

      // All checks passed — promote to targetDepth.
      createOp->setAttr("depth", builder.getI64IntegerAttr(targetDepth));

      // Scale capacity proportionally (slot_elems = depth * elemCount).
      if (capAttr) {
        createOp->setAttr("slot_elems", builder.getI64IntegerAttr(
                                            capAttr.getInt() * targetDepth));
      }

      // Update per-tile resource counters.
      for (auto [col, row] : consCoords) {
        int64_t key = tileKey(col, row);
        tileLockCount[key] += 1;
        tileBDCount[key] += (targetDepth - 1);
        if (capAttr && elemTypeAttr) {
          int64_t perSlotBytes =
              estimateSingleSlotBytes(elemTypeAttr.getValue());
          tileMemUsed[key] += perSlotBytes * targetDepth;
        }
      }

      ++promoted;
      createOp->emitRemark("conduit-depth-promote: promoted '")
          << name << "' from depth-1 to depth-" << targetDepth;
    }

    if (promoted > 0) {
      // Summary remark on the module op for test visibility.
      module.emitRemark("conduit-depth-promote: promoted ")
          << promoted << " conduit(s)";
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitDepthPromotePass() {
  return std::make_unique<ConduitDepthPromotePass>();
}

} // namespace xilinx::conduit
