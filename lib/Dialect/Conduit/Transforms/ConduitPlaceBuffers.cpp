//===- ConduitPlaceBuffers.cpp - DMA-aware mem_bank assignment ----*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --conduit-place-buffers: for each conduit.register_buffers op, assign
// mem_bank = i % numBanks on the i-th buffer so that consecutive FIFO slots
// land in different SRAM banks.
//
// Runs between --conduit-materialize-buffers and --aie-assign-buffer-addresses.
// Eliminates DMA↔core bank conflicts that degrade throughput when producer
// and consumer access the same bank simultaneously.
//
// AIE2 compute tiles have 4 SRAM banks (banks 0-3).
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace xilinx::conduit;
using namespace xilinx::AIE;

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITPLACEBUFFERS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// AIE2 compute tiles: 4 SRAM banks.
static constexpr int64_t kDefaultNumBanks = 4;

struct ConduitPlaceBuffersPass
    : impl::ConduitPlaceBuffersBase<ConduitPlaceBuffersPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();

    module.walk([&](RegisterBuffersOp rb) {
      int64_t bankIdx = 0;
      for (mlir::Value bufVal : rb.getBuffers()) {
        auto bufOp = bufVal.getDefiningOp<AIE::BufferOp>();
        if (!bufOp) {
          ++bankIdx;
          continue;
        }
        // Only assign mem_bank if not already set.
        if (!bufOp.getMemBank()) {
          bufOp.setMemBankAttr(mlir::IntegerAttr::get(
              mlir::IntegerType::get(ctx, 32), bankIdx % kDefaultNumBanks));
        }
        ++bankIdx;
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitPlaceBuffersPass() {
  return std::make_unique<ConduitPlaceBuffersPass>();
}

} // namespace xilinx::conduit
