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
// Declares the Conduit lowering passes:
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
// The intended pipeline is:
//
//   aie.objectfifo.* ──► Conduit IR ──► aie.dma_bd / aie.lock / aie.buffer
//
//   Pass A                   Pass C
//   (--objectfifo-to-conduit) (--conduit-to-dma)
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
#define GEN_PASS_DECL_CONDUITDEPTHPROMOTE
#define GEN_PASS_DECL_CONDUITPAIRINGCHECK
#define GEN_PASS_DECL_CONDUITLIVENESSCHECK
#define GEN_PASS_DECL_CONDUITFUSECHANNELS
#define GEN_PASS_DECL_CONDUITFUSERELAY
#define GEN_PASS_DECL_CONDUITCHECKCHANNELS
#define GEN_PASS_DECL_CONDUITINFERMODES
#define GEN_PASS_DECL_CONDUITCHECKDEPS
#define GEN_PASS_DECL_CONDUITINFERRATES
#define GEN_PASS_DECL_CONDUITCHECKORDERING
#define GEN_PASS_DECL_CONDUITCHECKTIERS
#define GEN_PASS_DECL_CONDUITCHECKLOOPBALANCE
#define GEN_PASS_DECL_CONDUITDMATASKTOCONDUIT
#define GEN_PASS_DECL_CONDUITFUSECOREBODIES
#define GEN_PASS_DECL_CONDUITCOMBINEDEVICE
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

//===----------------------------------------------------------------------===//
// Factory functions
//===----------------------------------------------------------------------===//

/// Pass A: lift aie.objectfifo.* ops into Conduit IR.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createObjectFifoToConduitPass();

/// Pass C: lower Conduit IR to aie.dma_bd / aie.lock / aie.buffer / aie.flow.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createConduitToDMAPass();

/// Depth promotion: promote eligible depth-1 conduits to depth-2.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitDepthPromotePass();

/// M9 Phase 2 pairing check: warn when acquire has no matching release in
/// block.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitPairingCheckPass();

/// M11 liveness check: error when a window lock grant is never released.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitLivenessCheckPass();

/// Channel fusion: annotate non-overlapping conduits on the same tile for
/// DMA channel sharing (addresses DMA channel exhaustion gap).
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitFuseChannelsPass();

/// Relay fusion: fuse gather→scatter relay chains through the same MemTile
/// into conduit.transpose, eliminating the intermediate channel.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitFuseRelayPass();

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

/// DMA task → Conduit: convert aiex.dma_configure_task_for / dma_start_task /
/// dma_await_task / dma_free_task in aie.runtime_sequence into
/// conduit.put_memref or conduit.get_memref ops.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitDmaTaskToConduitPass();

/// Loop-body fusion: compose aie.core bodies on the same tile connected by
/// intermediate conduit channels, replacing intermediate with L1 memref.alloc
/// or MemTile relay.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitFuseCoreBodyPass();

/// Device merge: merge two aie.device ops connected by matching fusion_group
/// attributes into one. tile-offset mode offsets devB tiles; same-tile mode
/// keeps coordinates. Does NOT rewrite channels — leaves that to downstream
/// fusion passes.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCombineDevicePass();

//===----------------------------------------------------------------------===//
// Pass registration (generated from Passes.td)
// Generates registerConduitPasses(), registerConduitToDMA(), etc.
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

} // namespace xilinx::conduit

#endif // AIE_DIALECT_CONDUIT_TRANSFORMS_CONDUITPASSES_H
