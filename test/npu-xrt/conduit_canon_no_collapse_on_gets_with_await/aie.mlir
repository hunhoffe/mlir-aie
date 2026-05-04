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
// 2026-05-03 in this commit) — the gets-side variant.  Companion to:
//   test/Dialect/Conduit/conduit_canon_no_collapse_on_token_true.mlir
//     (lit-only at-site pin for the predicate)
//   test/npu-xrt/conduit_canon_no_collapse_on_link/aie.mlir
//     (link-refusal HW smoke; structural sibling, different predicate)
//   test/npu-xrt/conduit_canon_no_collapse_on_puts_with_await/aie.mlir
//     (puts-side counterpart of THIS smoke)
//
// Pattern (geometry mirrors the puts-side smoke; the only difference is
// which side carries the per-issue ack request):
//   shim(0,0) ──@in──> compute(0,2) ──@out──> shim(0,0)
//
// To ISOLATE the gets-side `chainHasAwait` refusal:
//   * Puts on @in: 4 IDENTICAL MM2S puts with `aiex.dma_free_task` ONLY
//     (NO `aiex.dma_await_task`).  After dma-task-to-conduit, chain
//     shape is `[false]` only — collapse-eligible.  Canon's
//     HomogeneousRepeatPattern collapses these 4 puts into 1 put +
//     dma_repeat = 3 (the LEGITIMATE collapse path; chain has no
//     per-issue ack request, so the consolidated form does not starve
//     the consumer ack).  This branch verifies the predicate doesn't
//     over-refuse on the puts side.
//
//   * Gets on @out: 4 sliding-offset S2MM gets with
//     `aiex.dma_await_task` (per-issue ack) + `aiex.dma_free_task`
//     (release marker).  Chain shape `[true, false]` after
//     dma-task-to-conduit.  Per the new chainHasAwait predicate,
//     ArithProgressionPattern (and HomogeneousRepeatPattern, symmetric)
//     refuses to collapse these gets — the consolidated form would
//     inherit the per-issue ack on a single configure that fires N
//     times back-to-back, starving the producer-side ack pacing on
//     the compute → shim flow and stalling HW.
//
// Pre-fix, canon's ArithProgressionPattern collapses the 4 sliding-
// offset gets into 1 get + producer_dimensions outer wrap
// <size = 4, stride = 64>.  Pass C surfaces that verbatim onto a
// single shim configure_task with `repeat_count = 0` + the 4-step
// outer dim and `issue_token = true` (because the per-issue ack
// request was on the original chain).  The firmware fires that BD
// 4 times back-to-back without per-chunk producer ack pacing → HW
// stalls.  Same root-cause class as the link-refusal landed in
// commit 375b0e5233 and as the puts-side smoke.
//
// Post-fix (this working tree): canon's ArithProgressionPattern (and
// HomogeneousRepeatPattern, symmetric) calls
// `xilinx::conduit::detail::chainHasAwait(refShape)` (declared in
// CanonicalizeChannelPutsUtils.h) which returns true iff any element
// of the chain shape is `true`.  Canon emits a remark and returns
// false from tryCollapseArithGets / tryCollapseGets.  Pass C therefore
// emits 4 separate paced shim configures (each with its own per-issue
// ack), HW dispatch completes, and the per-byte byte-equivalence check
// in test.cpp passes.
//
// Compute kernel: in-core bf16-by-bf16 copy from the acquired input
// slice to the acquired output slot.  Each iteration consumes one input
// slice and produces one output slot, so 4 input arrivals → 4 output
// dispatches → 4 output slots populated.
//
// Reference (test.cpp computes this):
//   output[i*64 + j] = input[j]  for i in [0,4), j in [0,64)
// All input values 0..63 are bf16-exact (integers fit in sign + 8 exp +
// 7 mantissa).  Output is 4 identical replicas of the input ramp.
//
// HW-shape note: slice = 64 bf16 (128 bytes) clears npu2 shim BD
// min-length comfortably.  N = 4 is well under any tile BD cap.
// Channel is NOT linked (no aie.objectfifo.link), so the link refusal
// doesn't fire — chainHasAwait is the sole gate exercised on the
// gets side.

module {
  aie.device(NPUDEVICE) {
    %shim_0   = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @in  (%shim_0,   {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @out (%tile_0_2, {%shim_0},   2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

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

    // Runtime sequence:
    //   ---- 4x identical puts on @in with FREE ONLY (no await) ----
    //   Chain shape `[false]` after dma-task-to-conduit → canon
    //   collapses them to 1 put + dma_repeat = 3 (LEGITIMATE collapse).
    //
    //   ---- 4x sliding-offset gets on @out with ISSUE_TOKEN + AWAIT + FREE ----
    //   Chain shape `[true, false]` after dma-task-to-conduit → canon
    //   refuses to collapse via chainHasAwait.
    aie.runtime_sequence @canon_no_collapse_on_gets_with_await(
        %input  : memref<64xbf16>,
        %output : memref<256xbf16>) {
      // ---- 4x identical MM2S puts on @in, free-only (NO await) ----
      %ti0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ti0)
      aiex.dma_free_task(%ti0)

      %ti1 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ti1)
      aiex.dma_free_task(%ti1)

      %ti2 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ti2)
      aiex.dma_free_task(%ti2)

      %ti3 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ti3)
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
