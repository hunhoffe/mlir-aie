//===- ConduitFuseRelay.cpp - conduit-fuse-relay pass -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --conduit-fuse-relay: Fuse gather→scatter relay chains into
// conduit.transpose.
//
// Background
// ----------
// A gather→scatter relay chain routes N source channels through a MemTile
// into M destination channels.  The intermediate channel between the gather
// and scatter is a pure relay — it carries data through the MemTile without
// any compute-side consumer.  Fusing the pair into a single conduit.transpose
// eliminates the intermediate channel and halves the MemTile DMA channel
// usage (one S2MM+MM2S pair instead of two).
//
// Match condition
// ---------------
// A gather and scatter are fuseable when:
//   1. gather.dst == scatter.src  (connected via the same intermediate channel)
//   2. gather.memtile == scatter.memtile  (same MemTile relay point)
//   3. The intermediate channel has no other users (no acquire/release/
//      put_memref/get_memref ops reference that channel name)
//   4. srcs.size() + dsts.size() <= 12  (hardware port budget)
//   5. srcs.size() * dsts.size() <= 32  (AIE2 packet ID budget)
//
// Rewrite
// -------
// The fused conduit.transpose has:
//   - srcs = gather's srcs
//   - dsts = scatter's dsts
//   - memtile = shared memtile string
//   - offsets = N×M cross product of gather.offsets × scatter.offsets
//   - lock_id, sync_mode copied from gather if present
//
// The scatter, gather, and intermediate conduit.create are erased.
//
// Run with:  aie-opt --conduit-fuse-relay <input.mlir>
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITFUSERELAY
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check whether any Conduit activity op (acquire, release, put_memref,
/// get_memref, or their async variants) references 'name' in the module.
static bool hasActivityUsers(mlir::ModuleOp module, llvm::StringRef name) {
  bool found = false;
  module.walk([&](mlir::Operation *op) {
    if (found)
      return;
    if (!mlir::isa<Acquire, AcquireAsync, Release, ReleaseAsync, WaitWindow,
                   PutMemref, GetMemref, PutMemrefAsync, GetMemrefAsync>(op))
      return;
    // Release takes a window SSA operand — check the defining acquire's name.
    if (auto relOp = mlir::dyn_cast<Release>(op)) {
      mlir::Operation *def = relOp.getWindow().getDefiningOp();
      if (auto acq = mlir::dyn_cast_or_null<Acquire>(def))
        if (acq.getName() == name)
          found = true;
      return;
    }
    // All other ops carry an explicit 'name' attribute.
    if (auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name"))
      if (nameAttr.getValue() == name)
        found = true;
  });
  return found;
}

/// Compute the N×M cross-product offset array from gather offsets (size N)
/// and scatter offsets (size M).  If either is absent, treat as all-zeros.
static llvm::SmallVector<int64_t> computeCrossProductOffsets(
    std::optional<llvm::ArrayRef<int64_t>> gatherOffsets, unsigned N,
    std::optional<llvm::ArrayRef<int64_t>> scatterOffsets, unsigned M) {
  llvm::SmallVector<int64_t> result(N * M, 0);
  for (unsigned i = 0; i < N; ++i) {
    int64_t gOff =
        (gatherOffsets && i < gatherOffsets->size()) ? (*gatherOffsets)[i] : 0;
    for (unsigned j = 0; j < M; ++j) {
      int64_t sOff = (scatterOffsets && j < scatterOffsets->size())
                         ? (*scatterOffsets)[j]
                         : 0;
      result[i * M + j] = gOff + sOff;
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct ConduitFuseRelayPass
    : public impl::ConduitFuseRelayBase<ConduitFuseRelayPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();

    // Step 1: Collect all scatter ops, indexed by their $src name.
    llvm::StringMap<llvm::SmallVector<ScatterOp, 2>> scattersBySrc;
    module.walk([&](ScatterOp scatterOp) {
      scattersBySrc[scatterOp.getSrc()].push_back(scatterOp);
    });

    if (scattersBySrc.empty())
      return;

    // Step 2: Collect all conduit.create ops by name for intermediate erasure.
    llvm::StringMap<Create> createsByName;
    module.walk([&](Create createOp) {
      createsByName[createOp.getName().str()] = createOp;
    });

    // Step 3: Walk all gathers and attempt fusion.
    llvm::SmallVector<GatherOp> gathers;
    module.walk([&](GatherOp gatherOp) { gathers.push_back(gatherOp); });

    for (GatherOp gatherOp : gathers) {
      llvm::StringRef intermediateName = gatherOp.getDst();

      // Find a scatter whose src matches the gather's dst.
      auto it = scattersBySrc.find(intermediateName);
      if (it == scattersBySrc.end() || it->second.empty())
        continue;

      ScatterOp scatterOp = it->second.front();

      // Check same memtile.
      if (gatherOp.getMemtile() != scatterOp.getMemtile())
        continue;

      // Check no activity users on the intermediate channel.
      if (hasActivityUsers(module, intermediateName))
        continue;

      // Budget checks.
      unsigned N = gatherOp.getSrcs().size();
      unsigned M = scatterOp.getDsts().size();
      if (N + M > 12)
        continue;
      if (N * M > 32)
        continue;

      // --- Fuse: create conduit.transpose ---
      auto offsets = computeCrossProductOffsets(gatherOp.getOffsets(), N,
                                                scatterOp.getOffsets(), M);

      mlir::OpBuilder builder(gatherOp);

      // Build srcs and dsts arrays.
      auto srcs = gatherOp.getSrcs();
      auto dsts = scatterOp.getDsts();

      auto transposeOp = builder.create<TransposeOp>(
          gatherOp.getLoc(), srcs, dsts, gatherOp.getMemtile(),
          mlir::DenseI64ArrayAttr::get(ctx, offsets));
      (void)transposeOp;

      // Erase scatter, gather, and intermediate conduit.create.
      // Remove the scatter from the index first.
      auto &scatterVec = it->second;
      scatterVec.erase(scatterVec.begin());

      scatterOp->erase();
      gatherOp->erase();

      // Erase the intermediate conduit.create if it exists.
      auto createIt = createsByName.find(intermediateName);
      if (createIt != createsByName.end()) {
        createIt->second->erase();
        createsByName.erase(createIt);
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitFuseRelayPass() {
  return std::make_unique<ConduitFuseRelayPass>();
}

} // namespace xilinx::conduit
