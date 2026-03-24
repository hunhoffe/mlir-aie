//===- AIECheckCascadePairing.cpp -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Validation pass: checks that every aie.cascade_flow op has a corresponding
// aie.put_cascade in the source tile's core body and aie.get_cascade in the
// destination tile's core body.  Also checks the inverse: every core with a
// aie.put_cascade must have a cascade_flow naming that tile as source.
//
// Run with:  aie-opt --aie-check-cascade-pairing <input.mlir>
//
// Errors (not warnings) are emitted; signalPassFailure() is called if any
// violation is found.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIECHECKCASCADEPAIRING
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-check-cascade-pairing"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIECheckCascadePairingPass
    : xilinx::AIE::impl::AIECheckCascadePairingBase<
          AIECheckCascadePairingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    bool failed = false;

    // -----------------------------------------------------------------------
    // Step 1: Collect all cascade_flow ops; record source and dest tile Values.
    // -----------------------------------------------------------------------
    // cascadeFlowSources: set of TileOp Values that are named as a source in
    //   some cascade_flow.
    // cascadeFlowDests:   set of TileOp Values that are named as a dest.
    // Keep the CascadeFlowOp so we can emit errors on it.
    DenseMap<Value, CascadeFlowOp> sourceToFlow; // tile value → flow op
    DenseMap<Value, CascadeFlowOp> destToFlow;   // tile value → flow op

    for (CascadeFlowOp flowOp : device.getOps<CascadeFlowOp>()) {
      Value srcTile = flowOp.getSourceTile();
      Value dstTile = flowOp.getDestTile();
      sourceToFlow[srcTile] = flowOp;
      destToFlow[dstTile] = flowOp;
    }

    // -----------------------------------------------------------------------
    // Step 2: Walk all CoreOps; build tile → (hasPut, hasGet) map.
    //   Also collect: for each core with a put_cascade, record the tile Value
    //   and the put op (for the inverse check).
    // -----------------------------------------------------------------------
    // Map from tile Value to whether that core has a put/get cascade op.
    DenseMap<Value, bool> coreHasPut;
    DenseMap<Value, bool> coreHasGet;
    // Map from tile Value to the first put_cascade op in that core (for
    // error annotation when no flow exists).
    DenseMap<Value, PutCascadeOp> tileToPutOp;

    for (CoreOp coreOp : device.getOps<CoreOp>()) {
      Value tileval = coreOp.getTile();

      coreOp.walk([&](PutCascadeOp putOp) {
        coreHasPut[tileval] = true;
        if (!tileToPutOp.count(tileval))
          tileToPutOp[tileval] = putOp;
      });

      coreOp.walk([&](GetCascadeOp) { coreHasGet[tileval] = true; });
    }

    // -----------------------------------------------------------------------
    // Step 3: Forward check — for each cascade_flow verify put/get in cores.
    // -----------------------------------------------------------------------
    for (CascadeFlowOp flowOp : device.getOps<CascadeFlowOp>()) {
      Value srcTile = flowOp.getSourceTile();
      Value dstTile = flowOp.getDestTile();

      // Check: source tile's core must have a put_cascade.
      if (!coreHasPut.count(srcTile) || !coreHasPut[srcTile]) {
        flowOp.emitError(
            "'aie.cascade_flow' source tile has no 'aie.put_cascade' in its "
            "core body");
        failed = true;
      }

      // Check: dest tile's core must have a get_cascade.
      if (!coreHasGet.count(dstTile) || !coreHasGet[dstTile]) {
        flowOp.emitError(
            "'aie.cascade_flow' dest tile has no 'aie.get_cascade' in its "
            "core body");
        failed = true;
      }
    }

    // -----------------------------------------------------------------------
    // Step 4: Inverse check — each core with a put_cascade must have a
    //   cascade_flow naming that tile as source.
    // -----------------------------------------------------------------------
    for (auto &[tileval, putOp] : tileToPutOp) {
      if (!sourceToFlow.count(tileval)) {
        putOp.emitError(
            "'aie.put_cascade' in core body has no corresponding "
            "'aie.cascade_flow' naming this tile as source");
        failed = true;
      }
    }

    if (failed)
      signalPassFailure();
  }
};

std::unique_ptr<OperationPass<DeviceOp>>
AIE::createAIECheckCascadePairingPass() {
  return std::make_unique<AIECheckCascadePairingPass>();
}
