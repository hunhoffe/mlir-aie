//===- ConduitPasses.h - Conduit transformation pass declarations -*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Declares the two Conduit lowering passes:
//
//   Pass A: --objectfifo-to-conduit
//     Lifts aie.objectfifo.* ops into Conduit IR.  This is the entry point
//     for the unified lowering pipeline.
//
//   Pass C: --conduit-to-dma
//     Lowers Conduit IR to raw AIE hardware ops (aie.dma_bd, aie.lock,
//     aie.buffer, aie.flow).  Replaces the existing
//     --aie-objectFifo-stateful-transform path.
//
// The intended three-pass pipeline is:
//
//   aie.objectfifo.* ──┐
//                      ├──► Conduit IR ──► aie.dma_bd / aie.lock / aie.buffer
//   air.channel.*    ──┘
//
//   Pass A                   Pass C
//   (--objectfifo-to-conduit) (--conduit-to-dma)
//
// Pass B (--air-channel-to-conduit) lowers AIR Channel ops into Conduit
// Tier 3 memref-DMA ops (conduit.put_memref_async / conduit.get_memref_async).
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITPASSES_H
#define AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITPASSES_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/Pass/Pass.h"

namespace xilinx::conduit {

//===----------------------------------------------------------------------===//
// Pass declarations (generated from Passes.td)
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#define GEN_PASS_DECL_OBJECTFIFOTOCONDUIT
#define GEN_PASS_DECL_CONDUITTODMA
#define GEN_PASS_DECL_AIRCHANNELTOCONDUIT
#define GEN_PASS_DECL_CONDUITDEPTHPROMOTE
#define GEN_PASS_DECL_CONDUITPAIRINGCHECK
#define GEN_PASS_DECL_CONDUITLIVENESSCHECK
#define GEN_PASS_DECL_CONDUITFUSECHANNELS
#define GEN_PASS_DECL_CONDUITCHECKCHANNELS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

//===----------------------------------------------------------------------===//
// Factory functions
//===----------------------------------------------------------------------===//

/// Pass A: lift aie.objectfifo.* ops into Conduit IR.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createObjectFifoToConduitPass();

/// Pass B: lift air.channel.put/get ops into Conduit Tier 3 memref-DMA ops.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAirChannelToConduitPass();

/// Pass C: lower Conduit IR to aie.dma_bd / aie.lock / aie.buffer / aie.flow.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConduitToDMAPass();

/// Depth promotion: promote eligible depth-1 conduits to depth-2.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitDepthPromotePass();

/// M9 Phase 2 pairing check: warn when acquire has no matching release in block.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitPairingCheckPass();

/// M11 liveness check: error when a window lock grant is never released.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitLivenessCheckPass();

/// Channel fusion: annotate non-overlapping conduits on the same tile for
/// DMA channel sharing (addresses DMA channel exhaustion gap).
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitFuseChannelsPass();

/// Channel check: validate that no tile exceeds its hardware DMA channel limit.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCheckChannelsPass();

//===----------------------------------------------------------------------===//
// Pass registration (generated from Passes.td)
// Generates registerConduitPasses(), registerConduitToDMA(), etc.
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

} // namespace xilinx::conduit

#endif // AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITPASSES_H
