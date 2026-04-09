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
#define GEN_PASS_DECL_AIRCHANNELINDEXFLATTENER
#define GEN_PASS_DECL_CONDUITDEPTHPROMOTE
#define GEN_PASS_DECL_CONDUITPAIRINGCHECK
#define GEN_PASS_DECL_CONDUITLIVENESSCHECK
#define GEN_PASS_DECL_CONDUITFUSECHANNELS
#define GEN_PASS_DECL_CONDUITCHECKCHANNELS
#define GEN_PASS_DECL_CONDUITINFERMODES
#define GEN_PASS_DECL_CONDUITCHECKDEPS
#define GEN_PASS_DECL_CONDUITINFERRATES
#define GEN_PASS_DECL_CONDUITCHECKORDERING
#define GEN_PASS_DECL_CONDUITCHECKTIERS
#define GEN_PASS_DECL_CONDUITCHECKLOOPBALANCE
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

/// Air channel index flattener: flatten multi-dimensional air.channel
/// declarations and their put/get ops to scalar channels.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAirChannelIndexFlattenerPass();

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

/// Mode inference: resolve routing_mode="any" conduits to "circuit" or "packet"
/// using the R3 + Step 3.5 decision procedure.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitInferModesPass();

/// M12 dep-token DAG check: error when the $deps token DAG contains a cycle
/// (static deadlock — circular completion dependency between DMA ops).
/// Requires PASSB-DEP-001 fix (Task #24) for complete coverage.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCheckDepsPass();

/// Rate inference: infer CSDF producer_rates/consumer_rates from num_elems
/// attributes on conduit.put_memref_async / conduit.get_memref_async ops.
/// After attachment, M6/M7 verifiers fire automatically on next verify step.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitInferRatesPass();

/// CSDFa static-ordering verifier: warn when DMA-only channels on the same
/// tile fire at overlapping CSDF phases (ambiguous ordering).
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCheckOrderingPass();

/// M-12 tier check: error when Tier 2 and Tier 3 ops reference the same
/// channel in the same aie.core region (rotation counter invariant violation).
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCheckTiersPass();

/// MVE-1 loop balance check: warn when a channel's DMA repeat count is
/// exceeded by the static trip count of the enclosing scf.for consumer loop.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCheckLoopBalancePass();

/// Spatial IRON operator fusion: replace LPDDR5 intermediates between two
/// consecutive aie.device ops with shared-memory conduit.create channels.
/// Offsets tile coordinates in device B, emits module-level fused conduit,
/// deletes matched shim output/input channel pairs and their runtime DMA ops.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitFuseOperatorsPass();

/// Buffer materialization: emit aie.buffer × max(depth, window_size+1) on
/// each consumer tile and conduit.register_buffers linking them to the channel.
/// Enables --conduit-place-buffers to run before --conduit-to-dma.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitMaterializeBuffersPass();

/// Buffer placement: assign mem_bank = i % 4 on aie.buffer ops emitted by
/// --conduit-materialize-buffers for DMA-aware SRAM bank staggering.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitPlaceBuffersPass();

//===----------------------------------------------------------------------===//
// Pass registration (generated from Passes.td)
// Generates registerConduitPasses(), registerConduitToDMA(), etc.
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

} // namespace xilinx::conduit

#endif // AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITPASSES_H
