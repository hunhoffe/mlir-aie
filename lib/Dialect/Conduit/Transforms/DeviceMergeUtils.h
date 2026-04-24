//===- DeviceMergeUtils.h - Device-merge host-orchestrator helpers --------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Shared helpers for passes that physically merge one aie.device (devB) into
// another (devA) and then erase devB.  Without these helpers a host
// orchestrator like
//
//   aie.device(npu2) {
//     aie.runtime_sequence(...) {
//       aiex.configure @devA { ... aiex.run @sequence(...) }
//       aiex.configure @devB { ... aiex.run @sequence(...) }
//     }
//   }
//
// breaks after the merge: the host's `aiex.configure @devB` block is left
// dangling and the verifier emits "No such device: '@devB'".
//
// Used by: --conduit-fuse-operators, --conduit-fuse-core-bodies,
//          --aie-combine-device.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_CONDUIT_TRANSFORMS_DEVICEMERGEUTILS_H
#define AIE_DIALECT_CONDUIT_TRANSFORMS_DEVICEMERGEUTILS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Block;
} // namespace mlir

namespace xilinx::conduit::detail {

// ---------------------------------------------------------------------------
// Device-body merge primitives shared between --aie-combine-device,
// --conduit-fuse-core-bodies, and --conduit-fuse-operators.
//
// These helpers physically merge the body of devB into the body of devA
// (preserving SSA values via remove/insert; never cloning tile/lock/buffer
// ops).  They are pure deduplication of identical code that previously lived
// in three transform files; behavior is byte-identical to the prior
// per-pass implementations.
//
// The runtime_sequence merge differs across callers:
//   - --aie-combine-device and --conduit-fuse-core-bodies use the SIMPLE
//     append-and-clone strategy implemented in mergeRuntimeSequencesSimple.
//   - --conduit-fuse-operators uses sync-group-aware chunk interleaving
//     (kept inline in that pass) and additional dead-block-arg removal.
// ---------------------------------------------------------------------------

/// Find the first `aie.runtime_sequence` op in `body`, or nullptr.
mlir::Operation *findRuntimeSequence(mlir::Block &body);

/// Move all non-terminator, non-`aie.runtime_sequence` ops from `bodyB` to
/// just before `bodyA`'s terminator (or to the end of `bodyA` if no
/// terminator is present).
///
/// Returns the `aie.runtime_sequence` ops found in `bodyB` — these are NOT
/// moved by this helper; the caller decides how to merge them (simple
/// append-and-clone vs sync-group-aware interleaving).
///
/// Used as "Phase 1" of the device-body merge: it establishes the tile/shim
/// SSA values in bodyA's scope so any subsequent runtime_sequence body merge
/// (whose DMA ops reference those tiles) operates against valid definitions.
llvm::SmallVector<mlir::Operation *>
movePhase1NonSequenceOps(mlir::Block &bodyA, mlir::Block &bodyB);

/// Move `op` to just before `bodyA`'s terminator (or to the end of `bodyA`
/// if no terminator is present).  Used to promote devB's runtime_sequence
/// into devA when devA has none of its own.
void moveToEndOfDeviceBody(mlir::Operation *op, mlir::Block &bodyA);

/// "Phase 2 simple" — append-and-clone runtime_sequence merge for callers
/// that do not need sync-group-aware chunk interleaving (used by
/// --aie-combine-device and --conduit-fuse-core-bodies).
///
/// For each op in `seqOpsB`:
///   - If `seqA` already exists with a body region, append fresh block-args
///     to seqA matching seqB's, build an IRMapping, and clone seqB's
///     non-terminator body ops into seqA before its terminator.
///   - Otherwise (no seqA yet), promote seqB into bodyA via
///     moveToEndOfDeviceBody and update `seqA` (passed by reference) so
///     subsequent iterations see the promoted op.
///
/// `bodyA` is required for the seqB-promotion path.
void mergeRuntimeSequencesSimple(mlir::Operation *&seqA,
                                 llvm::ArrayRef<mlir::Operation *> seqOpsB,
                                 mlir::Block &bodyA);

/// Move every `aie.core`, `aie.mem`, and `aie.runtime_sequence` op currently
/// in `bodyA` to just before `bodyA`'s terminator (or end of bodyA if no
/// terminator).
///
/// Required after the body merge: Pass C emits aie.lock / aie.buffer ops
/// after the last aie.tile op in the merged body, but devA's pre-merge
/// aie.core ops appear BEFORE devB's tiles — so without this sink, locks
/// emitted later would land after the cores that use them, violating MLIR
/// dominance.  Sinking the cores/mems/runtime_sequences past the
/// soon-to-be-emitted lock/buffer insertion point fixes that.
void sinkCoresMemsAndSequences(mlir::Block &bodyA);

/// Rewrite host-orchestrator references to a soon-to-be-erased device.
///
/// Call this immediately BEFORE `devB->erase()` in any pass that physically
/// merges devB into devA.  The function walks the parent module looking for
/// `aiex.configure @<devB.sym_name> { ... aiex.run @<seq>(args) ... }` blocks
/// and rewrites them so the host orchestrator targets devA after the merge.
///
/// Strategy (per confB found):
///
///   1. If a sibling `aiex.configure @<devA.sym_name>` (confA) exists in the
///      same parent block (same kernel-launch boundary in the host
///      runtime_sequence), confB is FOLDED INTO confA:
///        - confB's non-RunOp body ops are moved before confA's RunOp.
///        - confA's single RunOp is rewritten to take the concatenation of
///          A's existing args followed by B's args.  Its symbol is updated
///          to the surviving runtime_sequence's name (`seqA`).
///        - confB is erased.
///      This collapses two LoadPDI cycles into one — the runtime semantics
///      that fusion is meant to produce.
///
///   2. Otherwise (no sibling confA in the same parent block), confB stays
///      where it is, but its `symbol` attribute is rewritten to devA and its
///      inner aiex.run's `runtime_sequence_symbol` is retargeted to seqA's
///      name.  The host still drives the merged device, just from this
///      independent kernel-launch boundary.
///
/// Pre-conditions:
///   - `devA` and `devB` are the operations being merged; the body merge has
///     already happened (tiles, cores, sequence body of devB now live in
///     devA).
///   - `seqA` is the surviving `aie.runtime_sequence` op in devA after the
///     body merge — i.e. either devA's pre-existing sequence (with seqB's
///     body spliced in) or seqB itself if devA had no sequence and seqB was
///     promoted.  May be null if neither device exposed a host-callable
///     sequence; in that case no `aiex.run` retargeting is performed.
///   - Caller has not yet erased devB.
///
/// IMPORTANT — split-phase contract:
/// This Phase 1 step rewrites the configure-block topology and reconciles
/// `aiex.run` symbols, but it does NOT validate the run-op arg vector against
/// the merged callee's arity.  At the moment this runs, the surviving sequence
/// still carries its post-Phase-2 (pre-Step-8c) block-arg count — i.e. the
/// concatenation of the two pre-merge sequences' args.  The fold branch below
/// emits the naive `runA.getArgs() ++ runB.getArgs()` concat for the same
/// reason: that vector matches the sequence at this point.
///
/// Callers (e.g. --conduit-fuse-operators) that subsequently TRIM the merged
/// sequence's block args (Step 8c: drop fused-internal-channel endpoints) MUST
/// follow up with `reconcileHostRunArgsAfterTrim` to project the same drops
/// into the host-side `aiex.run` arg vectors.  Without that follow-up, the
/// run callsite arity will disagree with the trimmed callee.
mlir::LogicalResult rewriteHostConfigureOnDeviceMerge(mlir::ModuleOp module,
                                                      AIE::DeviceOp devA,
                                                      AIE::DeviceOp devB,
                                                      mlir::Operation *seqA);

/// Phase 2 of the host-orchestrator rewrite — only needed by callers that trim
/// the merged sequence's block args after `rewriteHostConfigureOnDeviceMerge`.
///
/// Walks every `aiex.run` op inside an `aiex.configure @<devA.sym_name>` block
/// in `module` whose `runtime_sequence_symbol` matches `seqA`'s name, and
/// projects the Step-8c trim into its arg vector.
///
/// The run's argument vector is treated as the concatenation of two segments,
/// in this fixed order:
///   - segment A: indices `[0, origArgCountA)` correspond to devA's pre-merge
///     sequence block args (positions 0..origArgCountA-1 in seqA's pre-trim
///     signature).
///   - segment B: indices `[origArgCountA, origArgCountA + origArgCountB)`
///     correspond to devB's pre-merge sequence block args.
///
/// For each run, `deadA` indices are dropped from segment A and `deadB` indices
/// are dropped from segment B; the result must equal the surviving sequence's
/// current block-arg count.
///
/// Pre-conditions:
///   - `rewriteHostConfigureOnDeviceMerge` has already run; folded run-ops
///     therefore carry the naive concat layout described above.
///   - The Step-8c trim on `seqA`'s block args has already been applied, so
///     `seqA`'s current block-arg count equals
///     `(origArgCountA - |deadA|) + (origArgCountB - |deadB|)`.
///   - For runs inside a non-folded `aiex.configure @devA` block (i.e. one
///     that was rewritten in place rather than folded), the segment-A range
///     is empty (the run only carried devB's args).  The helper detects this
///     by run-arity and projects only `deadB`.
///
/// Returns failure() if any reconciled run's arg count disagrees with the
/// surviving sequence's arity, or if a run's pre-trim arg count is neither
/// `origArgCountA + origArgCountB` (folded) nor `origArgCountB` (rewritten in
/// place).
mlir::LogicalResult
reconcileHostRunArgsAfterTrim(mlir::ModuleOp module, AIE::DeviceOp devA,
                              mlir::Operation *seqA, unsigned origArgCountA,
                              unsigned origArgCountB,
                              const llvm::DenseSet<unsigned> &deadA,
                              const llvm::DenseSet<unsigned> &deadB);

/// Detect whether a conduit channel is a forward-chain endpoint
/// ("Pattern E") inside the given device.
///
/// A Pattern E endpoint is a channel that participates in a `conduit.scatter`
/// or `conduit.gather` op (as `src`, `dst`, or member of `dsts`).  These ops
/// are emitted by Pass A from `aie.objectfifo.link` to express data motion
/// that is not driven by a compute core body.
///
/// Why fusion passes must skip these matches:
///   The post-rewrite channel-rename walks (in fuse-operators / fuse-core-
///   bodies) only update ops carrying a `name` FlatSymbolRefAttr (acquire /
///   release / subview_access / put_memref / ...).  `conduit.scatter` /
///   `conduit.gather` reference channels through `src` / `dst` / `dsts`
///   symbol attributes, NOT `name` — so an erase + rename leaves the
///   scatter/gather pointing at the deleted symbol (silent dangling
///   FlatSymbolRefAttr).  Beyond the symbol bookkeeping, fusing a
///   forward-chain endpoint is semantically meaningless: there is no
///   compute body to merge with.
///
/// Used by: --conduit-fuse-operators, --conduit-fuse-core-bodies.
bool isForwardChainEndpoint(AIE::DeviceOp device, llvm::StringRef channelName);

} // namespace xilinx::conduit::detail

#endif // AIE_DIALECT_CONDUIT_TRANSFORMS_DEVICEMERGEUTILS_H
