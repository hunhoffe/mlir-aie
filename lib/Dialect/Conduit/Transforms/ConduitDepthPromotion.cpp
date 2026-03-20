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
// Promotes eligible depth-1 conduits to depth-2 (double-buffering) to enable
// compute-DMA overlap.  Runs after Pass A/B and before Pass C.
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

/// Collect conduit names that appear in any conduit.link (src or dst).
static llvm::StringSet<>
collectLinkedConduitNames(mlir::ModuleOp module) {
  llvm::StringSet<> linked;
  module.walk([&](Link op) {
    if (auto srcsAttr = op->getAttrOfType<mlir::ArrayAttr>("srcs")) {
      for (auto s : srcsAttr)
        if (auto str = mlir::dyn_cast<mlir::StringAttr>(s))
          linked.insert(str.getValue());
    }
    if (auto dstsAttr = op->getAttrOfType<mlir::ArrayAttr>("dsts")) {
      for (auto d : dstsAttr)
        if (auto str = mlir::dyn_cast<mlir::StringAttr>(d))
          linked.insert(str.getValue());
    }
  });
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
  return true; // no subview users — is passthrough
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

    // Step 1: Collect linked conduit names (exclusion criterion #2).
    auto linkedNames = collectLinkedConduitNames(module);

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
    // Tier 3 (air.channel-originated): conduit.put_memref[_async]/get_memref[_async]
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
        auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name");
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
        auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name");
        auto countAttr = op->getAttrOfType<mlir::IntegerAttr>("count");
        if (!nameAttr || !countAttr)
          return;
        releaseCounts[nameAttr.getValue()].push_back(countAttr.getInt());
      }

      // --- Tier 3: put_memref[_async] / get_memref[_async] ---
      if (mlir::isa<PutMemref, PutMemrefAsync>(op)) {
        auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name");
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
        auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name");
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

    // Step 4: Per-tile resource counters for budget checks.
    // key = (col, row) packed as int64_t
    auto tileKey = [](int64_t col, int64_t row) -> int64_t {
      return (col << 32) | (row & 0xFFFFFFFF);
    };
    llvm::DenseMap<int64_t, int64_t> tileLockCount;
    llvm::DenseMap<int64_t, int64_t> tileBDCount;
    llvm::DenseMap<int64_t, int64_t> tileMemUsed;

    // Pre-populate from existing conduit.create ops.
    module.walk([&](Create op) {
      // Cascade conduits use no buffers, locks, or BDs — skip resource counting.
      if (auto rm = op->getAttrOfType<mlir::StringAttr>("routing_mode"))
        if (rm.getValue() == "cascade")
          return;

      auto depthAttr = op->getAttrOfType<mlir::IntegerAttr>("depth");
      int64_t depth = depthAttr ? depthAttr.getInt() : 1;
      auto consTiles = op->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "consumer_tiles");
      auto prodTile = op->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "producer_tile");
      auto capAttr = op->getAttrOfType<mlir::IntegerAttr>("capacity");
      auto elemTypeAttr = op->getAttrOfType<mlir::TypeAttr>("element_type");

      // Estimate per-consumer resources.
      if (consTiles) {
        auto tiles = consTiles.asArrayRef();
        for (size_t i = 0; i + 1 < tiles.size(); i += 2) {
          int64_t key = tileKey(tiles[i], tiles[i + 1]);
          tileLockCount[key] += 2; // prod + cons lock pair
          tileBDCount[key] += depth;
          if (capAttr && elemTypeAttr) {
            int64_t perSlotBytes = estimateSingleSlotBytes(elemTypeAttr.getValue());
            int64_t newDepth = 2;
            tileMemUsed[key] += perSlotBytes * newDepth;
          }
        }
      }
      // Producer tile also uses resources for non-shim.
      if (prodTile) {
        auto pt = prodTile.asArrayRef();
        if (pt.size() >= 2 && pt[1] != 0) { // non-shim
          int64_t key = tileKey(pt[0], pt[1]);
          tileLockCount[key] += 2;
          tileBDCount[key] += depth;
        }
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
    mlir::OpBuilder builder(module.getContext());

    for (mlir::Operation *createOp : candidates) {
      auto nameAttr = createOp->getAttrOfType<mlir::StringAttr>("name");
      if (!nameAttr)
        continue;
      llvm::StringRef name = nameAttr.getValue();

      // Criterion 0: cascade conduits — depth is architecturally fixed at 1.
      // The cascade stream is a hardware register (rendezvous channel), not a FIFO.
      // Promoting to depth-2 would emit an incorrect depth attribute that Pass C
      // cannot implement. Skip silently; do not emit a remark (this is expected).
      if (auto routingMode =
              createOp->getAttrOfType<mlir::StringAttr>("routing_mode")) {
        if (routingMode.getValue() == "cascade")
          continue;
      }

      // Criterion 1: CSDF / cyclostatic access pattern.
      if (createOp->getAttrOfType<mlir::DenseI64ArrayAttr>(
              "access_pattern")) {
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
      auto prodRatesAttr = createOp->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "producer_rates");
      auto consRatesAttr = createOp->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "consumer_rates");
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

      // Criterion 6: memory budget.
      auto consTiles = createOp->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "consumer_tiles");
      auto capAttr = createOp->getAttrOfType<mlir::IntegerAttr>("capacity");
      auto elemTypeAttr = createOp->getAttrOfType<mlir::TypeAttr>(
          "element_type");
      bool memOverBudget = false;
      if (consTiles && capAttr && elemTypeAttr) {
        int64_t bufBytes = estimateSingleSlotBytes(elemTypeAttr.getValue());
        auto tiles = consTiles.asArrayRef();
        for (size_t i = 0; i + 1 < tiles.size(); i += 2) {
          int64_t key = tileKey(tiles[i], tiles[i + 1]);
          if (tileMemUsed[key] + bufBytes * targetDepth > kDefaultTileMemoryBytes) {
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
      if (isAIE1 && consTiles) {
        bool lockOverBudget = false;
        auto tiles = consTiles.asArrayRef();
        for (size_t i = 0; i + 1 < tiles.size(); i += 2) {
          int64_t key = tileKey(tiles[i], tiles[i + 1]);
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
      if (consTiles) {
        bool bdOverBudget = false;
        auto tiles = consTiles.asArrayRef();
        for (size_t i = 0; i + 1 < tiles.size(); i += 2) {
          int64_t key = tileKey(tiles[i], tiles[i + 1]);
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
      createOp->setAttr("depth",
          builder.getI64IntegerAttr(targetDepth));

      // Scale capacity proportionally (capacity = depth * elemCount).
      if (capAttr) {
        createOp->setAttr("capacity",
            builder.getI64IntegerAttr(capAttr.getInt() * targetDepth));
      }

      // Update per-tile resource counters.
      if (consTiles) {
        auto tiles = consTiles.asArrayRef();
        for (size_t i = 0; i + 1 < tiles.size(); i += 2) {
          int64_t key = tileKey(tiles[i], tiles[i + 1]);
          tileLockCount[key] += 1;
          tileBDCount[key] += (targetDepth - 1);
          if (capAttr && elemTypeAttr) {
            int64_t perSlotBytes = estimateSingleSlotBytes(elemTypeAttr.getValue());
            tileMemUsed[key] += perSlotBytes * targetDepth;
          }
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
