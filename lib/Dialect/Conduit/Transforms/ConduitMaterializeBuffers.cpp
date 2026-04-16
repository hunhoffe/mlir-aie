//===- ConduitMaterializeBuffers.cpp - emit aie.buffer per channel
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
// --conduit-materialize-buffers: for each conduit.create whose consumer tile
// can be inferred from conduit.acquire{port=Consume} inside aie.core bodies,
// emit aie.buffer × depth on the consumer tile.
//
// This decouples buffer allocation from Pass C so that --conduit-place-buffers
// can set mem_bank attributes before --conduit-to-dma runs.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/StringSet.h"

using namespace xilinx::conduit;
using namespace xilinx::AIE;

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITMATERIALIZEBUFFERS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

struct ConduitMaterializeBuffersPass
    : impl::ConduitMaterializeBuffersBase<ConduitMaterializeBuffersPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();
    mlir::OpBuilder builder(ctx);

    module.walk([&](AIE::DeviceOp device) {
      // ----------------------------------------------------------------
      // Step 1: Build consumer tile map.
      // Walk aie.core bodies for conduit.acquire{port=Consume} ops — the
      // same inference logic as Phase 5a in ConduitToDMACollect.cpp.
      // ----------------------------------------------------------------
      llvm::StringMap<llvm::SmallVector<mlir::Value>> channelToConsumerTiles;

      device.walk([&](AIE::CoreOp coreOp) {
        AIE::TileOp tileOp = coreOp.getTile().getDefiningOp<AIE::TileOp>();
        if (!tileOp)
          return;
        mlir::Value tileVal = tileOp.getResult();

        coreOp.walk([&](Acquire acqOp) {
          if (acqOp.getPort() != Port::Consume)
            return;
          std::string name = acqOp.getName().str();
          auto &vec = channelToConsumerTiles[name];
          if (llvm::find(vec, tileVal) == vec.end())
            vec.push_back(tileVal);
        });
      });

      // ----------------------------------------------------------------
      // Step 1.5: Build set of relay-managed channel names.
      // Scatter dst channels and gather src channels get their buffers
      // allocated by Pass C (ConduitToDMALink), so we must not emit
      // duplicates here.
      // ----------------------------------------------------------------
      llvm::StringSet<> relayDstChannels;
      device.walk([&](ScatterOp scatterOp) {
        for (auto d : scatterOp.getDsts())
          relayDstChannels.insert(
              mlir::cast<mlir::FlatSymbolRefAttr>(d).getValue());
      });
      device.walk([&](GatherOp gatherOp) {
        for (auto s : gatherOp.getSrcs())
          relayDstChannels.insert(
              mlir::cast<mlir::FlatSymbolRefAttr>(s).getValue());
      });

      // ----------------------------------------------------------------
      // Step 2: For each conduit.create, emit aie.buffer ops.
      // ----------------------------------------------------------------
      llvm::SmallVector<Create> creates;
      device.walk([&](Create createOp) { creates.push_back(createOp); });

      for (Create createOp : creates) {
        std::string name = createOp.getName().str();

        // Skip relay-managed channels (Pass C allocates their buffers).
        if (relayDstChannels.count(name))
          continue;

        // Skip if depth absent or = 0 — run --conduit-depth-promote first.
        if (!createOp.getDepth() || *createOp.getDepth() <= 0)
          continue;
        int64_t depth = *createOp.getDepth();

        // Skip if no compute consumer inferred from IR walk.
        auto consIt = channelToConsumerTiles.find(name);
        if (consIt == channelToConsumerTiles.end())
          continue;

        // element_type is required — always present.
        mlir::Type elemType = createOp.getElementType();
        auto bufTy = mlir::dyn_cast<mlir::MemRefType>(elemType);
        if (!bufTy)
          continue;

        // Buffer count equals depth (window_size removed from conduit.create).
        int64_t bufCount = depth;

        mlir::Location loc = createOp.getLoc();

        // Emit buffers on each consumer tile.
        for (mlir::Value tileVal : consIt->second) {
          // Insert before the device body terminator so all emitted ops
          // appear at device scope (not inside any core or mem region).
          builder.setInsertionPoint(device.getBody()->getTerminator());

          for (int64_t i = 0; i < bufCount; ++i) {
            std::string symName = name + "_cons_buff_" + std::to_string(i);
            builder.create<AIE::BufferOp>(
                loc, bufTy, tileVal, mlir::StringAttr::get(ctx, symName),
                /*address=*/mlir::IntegerAttr{},
                /*initial_value=*/mlir::ElementsAttr{},
                /*mem_bank=*/mlir::IntegerAttr{});
          }
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitMaterializeBuffersPass() {
  return std::make_unique<ConduitMaterializeBuffersPass>();
}

} // namespace xilinx::conduit
