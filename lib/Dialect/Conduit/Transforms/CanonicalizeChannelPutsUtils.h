//===- CanonicalizeChannelPutsUtils.h - shared canon helpers --*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Shared analysis helpers for --conduit-canonicalize-channel-puts and
// --conduit-expand-channel-puts.  Behavior is unchanged from the legacy
// monolithic implementation; this header simply makes the helpers
// reachable from the patterns/ subdirectory.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_CONDUIT_TRANSFORMS_CANONICALIZECHANNELPUTSUTILS_H
#define AIE_DIALECT_CONDUIT_TRANSFORMS_CANONICALIZECHANNELPUTSUTILS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir {
class Operation;
} // namespace mlir

namespace xilinx::conduit::detail {

// Read the optional channel-level dma_repeat as int64.  Absent → 0.
//
// Convention (USER-LOCKED 2026-05-01 via Task #39 / Bug #98): dma_repeat is
// 0-INDEXED — the field encodes "additional fires beyond the initial one,"
// matching IRON's `aiex.dma_configure_task_for.repeat_count` semantic
// (see `aiex.py:289-291` where IRON sets `repeat_count = sizes[0] - 1`).
// Total fires = 1 + dma_repeat.  Absent attribute = 0 = single fire (the
// default DMA dispatch).  Pass C surfaces dma_repeat verbatim onto
// configure_task.repeat_count, which firmware reads as "BD fires value+1
// times" via NpuPushQueueOp (AIEDmaToNpu.cpp:180-183).
int64_t getDmaRepeatOr0(::xilinx::conduit::Create createOp);

// Returns true iff two PutMemrefAsync ops are structurally identical for the
// purpose of loop-unroll collapse.
bool putsAreStructurallyIdentical(::xilinx::conduit::PutMemrefAsync a,
                                  ::xilinx::conduit::PutMemrefAsync b);

// Returns true iff two GetMemrefAsync ops are structurally identical for the
// purpose of loop-unroll collapse.
bool getsAreStructurallyIdentical(::xilinx::conduit::GetMemrefAsync a,
                                  ::xilinx::conduit::GetMemrefAsync b);

// Collect the WaitAll ops that consume a token (a put/get's getToken()).
std::optional<llvm::SmallVector<::xilinx::conduit::WaitAll>>
collectSyncChain(mlir::Value tok);

// Return the bool-attr "shape" of a sync chain — list of `token` attribute
// values.  Two chains with the same shape can be collapsed together.
llvm::SmallVector<bool>
chainShape(llvm::ArrayRef<::xilinx::conduit::WaitAll> chain);

// Find the parent DeviceOp for a Create op.
::xilinx::AIE::DeviceOp findEnclosingDevice(mlir::Operation *op);

// Look up the producer tile SSA Value for a channel using inferAllTiles.
mlir::Value lookupProducerTile(mlir::Operation *scope,
                               llvm::StringRef channelName);

// Look up the first non-shim consumer tile SSA Value for a channel.
mlir::Value lookupConsumerTile(mlir::Operation *scope,
                               llvm::StringRef channelName);

// Return the BD cap for `tile` on the active target model, or nullopt when
// the cap cannot be determined.
std::optional<uint32_t> tileBDCap(mlir::Operation *scope, mlir::Value tile);

// Return the per-BD data-layout-dim cap for `tile` (3 for compute/core tiles,
// 4 for MemTile and Shim, nullopt when the tile type cannot be determined).
// Mirrors AIEDialect.cpp:2233-2236 (compute/MemTile dma_bd verifier) and
// AIEDMATasksToNPU.cpp:347-350 (shim runtime-sequence cap).
std::optional<uint32_t> tileBDDimCap(mlir::Operation *scope, mlir::Value tile);

/// Returns true iff any `wait_all` in the per-put/per-get sync chain has
/// `token = true` set (the IR-level signal that IRON requested a per-issue
/// `aiex.dma_await_task` ack).  Pass A's `--dma-task-to-conduit`
/// (ConduitDmaTaskToConduit.cpp:551-558) ALWAYS stamps `setTokenAttr(true)`
/// when lowering an `aiex.dma_await_task`; absence (`token=false` or
/// elided default) means the IR carries no per-issue ack request.
/// `WaitAllOp.token` is `DefaultValuedOptionalAttr<BoolAttr, "true">` per
/// Conduit.td:1116-1117 — the printer ELIDES the default `true`, so
/// `wait_all %0 :` (no explicit attr) parses back as token=true.
///
/// Channels whose chain shape contains any `true` MUST NOT be collapsed
/// by canon (Homogeneous- or ArithProgression-).  A collapsed
/// `(1 configure × dma_repeat=N-1)` form produces ONE consolidated
/// firmware ack at the end of the configure, which starves the per-chunk
/// consumer-side ack request encoded in the original IR.  Empirically
/// (see `conduit_canon_no_collapse_on_link/` smoke and
/// `conduit_to_dma_b_channel_consolidation.mlir` lit pin) the consolidated
/// form stalls HW (XRT timeout or all-zero downstream invocations) on
/// channels that still expect per-issue ack semantics — which includes
/// every linked-relay path AND every IRON path that emits
/// `aiex.dma_await_task`.  The per-issue-ack failure mode is symmetric to
/// the linked-channel failure mode and is enforced via the same predicate
/// pattern.
bool chainHasAwait(llvm::ArrayRef<bool> chainShape);

/// Returns true if `chanName` participates in any aie.objectfifo.link
/// (lowered to conduit.scatter/gather/transpose by Pass A) within `scope`.
/// Channels on the linked path MUST NOT be collapsed by canon — the
/// downstream Pass C link path (ConduitToDMALink.cpp) emits N separate
/// paced configures for the linked-MM2S→memtile shape, and a collapsed
/// (1 configure × dma_repeat=N-1) form does not compose with multi-round
/// consumer pacing on the linked path. Verified empirically 2026-05-03 by
/// hand-patch experiment on Llama prefill attn_scores GEMM
/// (M=2048 K=2048 N=512); collapsed form hangs at HW dispatch with XRT
/// timeout, paced form completes in <1ms with bf16 max_abs_diff 0.003261.
/// Inspection pattern mirrors ConduitDepthPromotion.cpp:79-103
/// (`collectLinkedConduitNames`): walk Scatter/Gather/Transpose ops and
/// read their `src`/`srcs`/`dst`/`dsts` symbol-ref attrs.
bool isLinkedChannel(mlir::Operation *scope, llvm::StringRef chanName);

} // namespace xilinx::conduit::detail

#endif // AIE_DIALECT_CONDUIT_TRANSFORMS_CANONICALIZECHANNELPUTSUTILS_H
