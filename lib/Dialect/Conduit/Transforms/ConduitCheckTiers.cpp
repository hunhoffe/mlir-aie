//===- ConduitCheckTiers.cpp - conduit-check-tiers pass ----------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// M-12: Mixed-tier verifier.
//
// Rejects programs that use BOTH Tier 2 (acquire/release) AND Tier 3
// (get_memref/put_memref) ops for the same channel name within the same
// aie.core region.
//
// Background:
//   Tier 2 ops (acquire, release, subview_access, acquire_async, release_async)
//   implement explicit buffer slot ownership with a rotation counter maintained
//   inside the core body.  The rotation counter selects which physical buffer
//   corresponds to the current logical window slot.
//
//   Tier 3 ops (get_memref, put_memref, get_memref_async, put_memref_async)
//   are fire-and-forget DMA ops that bypass the rotation counter entirely.
//   Pass C generates a fixed BD chain for Tier 3 without updating the counter.
//
//   If a single aie.core mixes both tiers for the SAME channel name:
//     - The Tier 3 op issues a DMA transfer that fills (or drains) a buffer slot
//       without advancing the rotation counter.
//     - The subsequent Tier 2 acquire uses the stale counter value and selects
//       the WRONG physical buffer.
//     - Result: silent read of stale or uninitialized data — hardware correctness
//       bug with no compile-time or runtime signal.
//
//   This pass catches the bug at compile time with a hard error.
//
// Valid patterns (NOT rejected):
//   - Cross-endpoint: Tier 3 on the shim producer, Tier 2 on a compute core
//     consumer.  This is the canonical ObjectFIFO shim-to-core pattern and is
//     explicitly supported.
//   - Same channel name, different aie.core regions: no rotation counter is
//     shared across cores, so there is no correctness hazard.
//   - Tier 2-only or Tier 3-only use within any single core: the single-tier
//     lowering path is well-defined.
//
// The check is performed per-core, not per-function, because the rotation
// counter is local to each aie.core region.
//
// Run with:  aie-opt --conduit-check-tiers <input.mlir>
//
// This pass is OPT-IN and NOT part of the default pipeline.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITCHECKTIERS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// Tier classification for an op found inside a core region.
enum class Tier { T2, T3 };

// Check one aie.core region for mixed-tier channel usage.
// Returns true if any error was emitted.
static bool checkMixedTiersInCore(mlir::Region &coreRegion) {
  // Map channel name -> first op using it at that tier.
  llvm::StringMap<mlir::Operation *> channelTier2;
  llvm::StringMap<mlir::Operation *> channelTier3;

  // Track channels already reported to suppress duplicate errors when multiple
  // ops of the same (wrong) tier reference the same channel.
  llvm::StringSet<> alreadyErrored;

  bool anyError = false;

  // Helper: try to get a channel name from a window SSA value by tracing
  // back through its defining op (Acquire or WaitWindow).
  auto nameFromWindow = [](mlir::Value win) -> llvm::StringRef {
    mlir::Operation *defOp = win.getDefiningOp();
    if (!defOp)
      return {};
    if (auto aDef = mlir::dyn_cast<Acquire>(defOp))
      return aDef.getName();
    if (auto wDef = mlir::dyn_cast<WaitWindow>(defOp))
      return wDef.getName();
    return {};
  };

  // Walk every op nested inside this core region (any nesting depth).
  coreRegion.walk([&](mlir::Operation *op) {
    // Determine if this op is Tier 2 or Tier 3 and extract the channel name.
    llvm::StringRef chanName;
    Tier tier;

    if (auto acqOp = mlir::dyn_cast<Acquire>(op)) {
      chanName = acqOp.getName();
      tier = Tier::T2;
    } else if (auto relOp = mlir::dyn_cast<Release>(op)) {
      // Release takes the window SSA value, not a name attr.
      chanName = nameFromWindow(relOp.getWindow());
      tier = Tier::T2;
    } else if (auto subOp = mlir::dyn_cast<SubviewAccess>(op)) {
      chanName = nameFromWindow(subOp.getWindow());
      tier = Tier::T2;
    } else if (auto acqAsOp = mlir::dyn_cast<AcquireAsync>(op)) {
      chanName = acqAsOp.getName();
      tier = Tier::T2;
    } else if (auto relAsOp = mlir::dyn_cast<ReleaseAsync>(op)) {
      chanName = relAsOp.getName();
      tier = Tier::T2;
    } else if (auto waitWin = mlir::dyn_cast<WaitWindow>(op)) {
      chanName = waitWin.getName();
      tier = Tier::T2;
    } else if (auto getOp = mlir::dyn_cast<GetMemref>(op)) {
      chanName = getOp.getName();
      tier = Tier::T3;
    } else if (auto putOp = mlir::dyn_cast<PutMemref>(op)) {
      chanName = putOp.getName();
      tier = Tier::T3;
    } else if (auto getAsOp = mlir::dyn_cast<GetMemrefAsync>(op)) {
      chanName = getAsOp.getName();
      tier = Tier::T3;
    } else if (auto putAsOp = mlir::dyn_cast<PutMemrefAsync>(op)) {
      chanName = putAsOp.getName();
      tier = Tier::T3;
    } else {
      return; // not a Conduit channel op
    }

    if (chanName.empty())
      return;

    // Record the op in its tier map.  If the other tier already has an entry
    // for this channel, emit the mixed-tier error (once per channel).
    if (tier == Tier::T2) {
      // Record first T2 occurrence (idempotent if already present).
      if (!channelTier2.count(chanName))
        channelTier2[chanName] = op;

      // Check if T3 already saw this channel.
      if (channelTier3.count(chanName) && !alreadyErrored.count(chanName)) {
        op->emitError("conduit channel '")
            << chanName
            << "' mixed Tier 2 (acquire/release) and Tier 3 "
               "(get_memref/put_memref) ops in same aie.core -- undefined "
               "behavior; use one tier per endpoint";
        alreadyErrored.insert(chanName);
        anyError = true;
      }
    } else {
      // Tier::T3 — record first T3 occurrence (idempotent if already present).
      if (!channelTier3.count(chanName))
        channelTier3[chanName] = op;

      // Check if T2 already saw this channel.
      if (channelTier2.count(chanName) && !alreadyErrored.count(chanName)) {
        op->emitError("conduit channel '")
            << chanName
            << "' mixed Tier 2 (acquire/release) and Tier 3 "
               "(get_memref/put_memref) ops in same aie.core -- undefined "
               "behavior; use one tier per endpoint";
        alreadyErrored.insert(chanName);
        anyError = true;
      }
    }
  });

  return anyError;
}

struct ConduitCheckTiersPass
    : public impl::ConduitCheckTiersBase<ConduitCheckTiersPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    bool anyFailure = false;

    module.walk([&](AIE::CoreOp coreOp) {
      if (checkMixedTiersInCore(coreOp.getBody()))
        anyFailure = true;
    });

    if (anyFailure)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCheckTiersPass() {
  return std::make_unique<ConduitCheckTiersPass>();
}

} // namespace xilinx::conduit
