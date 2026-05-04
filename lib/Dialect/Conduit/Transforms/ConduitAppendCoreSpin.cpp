//===- ConduitAppendCoreSpin.cpp - conduit-append-core-spin pass -*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --conduit-append-core-spin: append an empty infinite spin loop just before
// every aie.core's terminating aie.end op.
//
// Background
// ----------
// Llama decode-hang (Task #72/#83/#89/#90) is localized to op18_GEMV (LM-head,
// ni=1) at the LAST orchestrator-position.  Cores with finite inner-loop trip
// counts reach aie.end after their work, exposing a firmware-runtime hang at
// orchestrator-position-LAST.  Workaround proven via Task #89
// (LLAMA_DECODE_RUNLIST_APPEND_OP_K=1 → decode completes): if the orchestrator
// has any work after op18, the hang does not surface.
//
// Inserting an infinite spin in every aie.core prevents the core from ever
// reaching aie.end.  Equivalent in spirit to upstream stateful's `while(true)`
// outer-loop wrap that --dynamic-objFifos=true relies on but that conduit's
// existing emit does not preserve for finite trip-count cases.
//
// Emission shape per aie.core
// ---------------------------
//
//   aie.core(%tile) {
//     <existing body>
//     %c0 = arith.constant 0 : index
//     %max = arith.constant 9223372036854775806 : index   // i64 max - 1
//     %c1 = arith.constant 1 : index
//     scf.for %i = %c0 to %max step %c1 {
//       %t = arith.constant 0 : index   // single trivial op
//     }
//     aie.end                             // unreachable
//   }
//
// The single trivial op inside the scf.for body is required because scf.for
// needs a non-empty body region with an scf.yield terminator (auto-inserted
// by the builder when no yields are produced).
//
// Pipeline placement
// ------------------
// AFTER --conduit-to-dma, BEFORE Step 5 device-level passes (aiecc.cpp:1541).
// Runs always under --use-conduit (not opt-in).
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITAPPENDCORESPIN
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

struct ConduitAppendCoreSpinPass
    : public impl::ConduitAppendCoreSpinBase<ConduitAppendCoreSpinPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    module.walk([&](xilinx::AIE::CoreOp coreOp) {
      mlir::Region &body = coreOp.getBody();
      if (body.empty())
        return;
      mlir::Block &block = body.front();
      auto endOp =
          mlir::dyn_cast_or_null<xilinx::AIE::EndOp>(block.getTerminator());
      if (!endOp)
        return; // No aie.end terminator — skip (malformed core).

      mlir::OpBuilder builder(endOp);
      mlir::Location loc = endOp.getLoc();

      auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      // Task #105 (Task #90 follow-up): bumped from 16777214 (0xFFFFFE,
      // ~16 ms at AIE2 ~1 GHz) to i64 max-1 (effectively infinite). The
      // 16777214 bound finished spinning in ~10% of a 150 ms decode tick,
      // letting cores hit aie.end early and re-trigger the orchestrator-tail
      // hang Task #90 was meant to suppress (decode produced ~30 tokens
      // bit-identical to stateful, then ERT_CMD_STATE_TIMEOUT mid-token).
      auto cmax = builder.create<mlir::arith::ConstantIndexOp>(
          loc, 9223372036854775806LL); // i64 max - 1
      auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);

      auto forOp = builder.create<mlir::scf::ForOp>(loc, c0, cmax, c1);
      mlir::OpBuilder bodyBuilder =
          mlir::OpBuilder::atBlockBegin(forOp.getBody());
      // Single trivial op inside the loop body so the body is non-empty
      // (scf.for builder auto-inserts the scf.yield terminator).
      bodyBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitAppendCoreSpinPass() {
  return std::make_unique<ConduitAppendCoreSpinPass>();
}

} // namespace xilinx::conduit
