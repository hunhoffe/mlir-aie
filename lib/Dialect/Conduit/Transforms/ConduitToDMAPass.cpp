//===- ConduitToDMAPass.cpp - Pass class + runOnOperation shell --*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Pass C of the Conduit lowering pipeline: lower Conduit IR to raw AIE
// hardware programming ops (aie.dma_bd, aie.lock, aie.buffer, aie.flow).
//
// This file contains only the pass class definition and runOnOperation()
// shell.  The actual lowering logic is split across per-phase files:
//
//   ConduitToDMACollect.cpp  — Phase 1-2.5: conduitMap, tile cache, depth
//   ConduitToDMAAlloc.cpp    — Phase 3: buffer + lock allocation
//   ConduitToDMARoute.cpp    — Phase 4-4.5a: shim DMA, flow emission
//   ConduitToDMALink.cpp     — Phase 5-5.5: objectfifo_link + BD chains
//   ConduitToDMALower.cpp    — Phase 6-8: acquire/release/async + erasure
//
//===----------------------------------------------------------------------===//

#include "ConduitToDMACommon.h"
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITTODMA
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

struct ConduitToDMAPass : impl::ConduitToDMABase<ConduitToDMAPass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());

    // Initialize shared state.
    ConduitToDMAState state;
    state.module = module;
    state.builder = &builder;
    state.ctx = module.getContext();

    // Phase 1-2.5: Collect conduit metadata, find device, build tile cache,
    // compute effective depths, gather link source names.
    collectPhase(state);
    if (state.passFailed) { signalPassFailure(); return; }

    // Phase 3: Allocate buffers and locks for each conduit.
    allocPhase(state);
    if (state.passFailed) { signalPassFailure(); return; }

    // Phase 4-4.5a: Shim DMA allocation, symbol rewriting, flow emission.
    routePhase(state);
    if (state.passFailed) { signalPassFailure(); return; }

    // Phase 5-5.5: ObjectFifoLink lowering, aie.mem BD chains, fused chains.
    linkPhase(state);
    if (state.passFailed) { signalPassFailure(); return; }

    // Phase 6-8: Acquire/release lowering, op erasure, async path.
    lowerPhase(state);
    if (state.passFailed) { signalPassFailure(); return; }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConduitToDMAPass() {
  return std::make_unique<ConduitToDMAPass>();
}

} // namespace xilinx::conduit
