//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// HW smoke for the canon refuse-to-collapse-on-await predicate
// (`chainHasAwait` in CanonicalizeChannelPutsUtils.{h,cpp}, landed
// 2026-05-03 in this commit) — the puts-side variant.  Companion to:
//   test/Dialect/Conduit/conduit_canon_no_collapse_on_token_true.mlir
//     (lit-only at-site pin for the predicate)
//   test/npu-xrt/conduit_canon_no_collapse_on_link/aie.mlir
//     (link-refusal HW smoke; structural sibling, different predicate)
//   test/npu-xrt/conduit_canon_no_collapse_on_gets_with_await/aie.mlir
//     (the gets-side counterpart of THIS smoke)
//
// Pattern (geometry mirrors the canon-link smoke, scaled identically;
// the only differences are (a) NON-linked input path so the link
// predicate doesn't fire, and (b) per-issue ack request on the puts):
//   shim(0,0) ──@in──> compute(0,2) ──@out──> shim(0,0)
//
// Runtime sequence: 4 IDENTICAL MM2S puts on @in, EACH carrying both
// `aiex.dma_await_task` (per-issue ack) and `aiex.dma_free_task`.  After
// dma-task-to-conduit, the per-put sync chain shape is `[true, false]`
// — exactly what the new `chainHasAwait` predicate matches.
//
// Pre-fix, canon's HomogeneousRepeatPattern collapses the 4 puts into
// 1 put + dma_repeat = 3 (passes structural-identity, no deps, no
// existing dma_repeat, under tile BD cap).  Pass C surfaces that
// verbatim onto a single shim configure_task with `repeat_count = 3`
// and `issue_token = true` (because the per-issue ack request was on
// the original chain, the merged single configure inherits it).  The
// firmware fires that BD 4 times back-to-back without waiting for
// per-chunk consumer acks — the consumer-side ack starves and HW
// stalls.  Symptom on this smoke (before the fix): XRT timeout / HW
// hang on dispatch.  Same root-cause class as the link-refusal landed
// in commit 375b0e5233.
//
// Post-fix (this working tree): canon's HomogeneousRepeatPattern (and
// ArithProgressionPattern, symmetric) calls
// `xilinx::conduit::detail::chainHasAwait(refShape)` (declared in
// CanonicalizeChannelPutsUtils.h) which returns true iff any element
// of the chain shape is `true` (i.e. any per-issue ack request).
// Canon emits a remark and returns false from tryCollapsePuts.  Pass C
// therefore emits 4 separate paced shim configures (each with its own
// per-issue ack), HW dispatch completes, and the per-byte
// byte-equivalence check in test.cpp passes.
//
// Compute kernel: in-core bf16-by-bf16 copy from the acquired input
// slice to the acquired output slot.  Each iteration consumes one input
// slice and produces one output slot, so 4 puts → 4 dispatches → 4
// output slots populated.
//
// Reference (test.cpp computes this):
//   output[i*64 + j] = input[j]  for i in [0,4), j in [0,64)
// All input values 0..63 are bf16-exact (integers fit in sign + 8 exp +
// 7 mantissa).  Output is 4 identical replicas of the input ramp.
//
// HW-shape note: slice = 64 bf16 (128 bytes) clears npu2 shim BD
// min-length comfortably.  N = 4 is well under any tile BD cap so the
// only canon refusal that fires on the puts side is the new
// chain-await refusal (NOT the cap-refusal).  Channel is NOT linked
// (no aie.objectfifo.link), so the link refusal also doesn't fire —
// chainHasAwait is the sole gate exercised here.

module {
  aie.device(NPUDEVICE) {
    %shim_0   = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Direct shim → compute (NOT linked); chainHasAwait is the only
    // gate that should fire on the puts side here.
    aie.objectfifo @in  (%shim_0,   {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @out (%tile_0_2, {%shim_0},   2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Compute core: per-iteration copy from acquired @in slice to
    // acquired @out slot.  Trip = ∞ sentinel; the host runtime sequence
    // controls the actual dispatch count via shim puts/gets.
    %core_0_2 = aie.core(%tile_0_2) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 0xFFFFFE : index
      scf.for %niter = %c0 to %cmax step %c1 {
        %sv_in = aie.objectfifo.acquire @in (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_in = aie.objectfifo.subview.access %sv_in[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_out = aie.objectfifo.acquire @out (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_out = aie.objectfifo.subview.access %sv_out[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_in[%j]  : memref<64xbf16>
          memref.store %v, %elem_out[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @in  (Consume, 1)
        aie.objectfifo.release @out (Produce, 1)
      }
      aie.end
    }

    // Runtime sequence: 4 STRUCTURALLY IDENTICAL puts on @in, EACH
    // with per-issue `dma_await_task` (chain shape `[true, false]` post
    // dma-task-to-conduit).  Matches HomogeneousRepeatPattern's
    // structural-identity check exactly; the new chainHasAwait
    // predicate must fire and refuse the collapse.
    //
    // Output: 4 S2MM gets on @out at offsets [0, 64, 128, 192] with
    // standard issue_token = true / await / free pattern.  Symmetric
    // chainHasAwait refusal applies to the gets side too — verifies
    // the predicate fires on both code paths in one IR.
    aie.runtime_sequence @canon_no_collapse_on_puts_with_await(
        %input  : memref<64xbf16>,
        %output : memref<256xbf16>) {
      // ---- 4x identical MM2S puts on @in, each with await + free ----
      %ti0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti0)
      aiex.dma_await_task(%ti0)
      aiex.dma_free_task(%ti0)

      %ti1 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti1)
      aiex.dma_await_task(%ti1)
      aiex.dma_free_task(%ti1)

      %ti2 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti2)
      aiex.dma_await_task(%ti2)
      aiex.dma_free_task(%ti2)

      %ti3 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti3)
      aiex.dma_await_task(%ti3)
      aiex.dma_free_task(%ti3)

      // ---- 4x S2MM gets on @out, each with issue_token + await + free ----
      %to0 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%output : memref<256xbf16>, 0,   64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to0)
      aiex.dma_await_task(%to0)
      aiex.dma_free_task(%to0)

      %to1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%output : memref<256xbf16>, 64,  64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to1)
      aiex.dma_await_task(%to1)
      aiex.dma_free_task(%to1)

      %to2 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%output : memref<256xbf16>, 128, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to2)
      aiex.dma_await_task(%to2)
      aiex.dma_free_task(%to2)

      %to3 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%output : memref<256xbf16>, 192, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to3)
      aiex.dma_await_task(%to3)
      aiex.dma_free_task(%to3)
    }
  }
}
