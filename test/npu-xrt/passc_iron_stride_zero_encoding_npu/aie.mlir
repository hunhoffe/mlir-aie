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
// HW smoke for the IRON `dma_repeat`-as-stride-0-outer-dim encoding case
// of `--conduit-to-dma` (Pass C v4 case (d)).  Lit-only counterpart pin:
//   test/Dialect/Conduit/passc_effective_repeat_iron_stride_zero_encoding.mlir
//
// Anchors firmware ground truth for the v4 stride-discriminator design.
// Pass C v4 (Sprint N+2) discriminates BD outer.STRIDE to choose between
// two emit cases on shim MM2S:
//   * outer.stride > 0 (BD walks N times, real iteration)   → push-queue=1
//   * outer.stride == 0 (BD does NOT walk; IRON dma_repeat
//     encoding per AIEDmaToNpu.cpp:385-388)                  → push-queue=N
// The stride==0 limb has heretofore rested on inferred firmware semantics
// + a lit-only pin (the named lit fixture above).  This e2e smoke is the
// missing third leg: a HW dispatch that PASSES iff
// `repeat_count = outer.size` is firmware-correct for the stride==0
// encoding.  Without this, v4 ships with one of three Pass C emit cases
// (cases b / c / d in v4) lacking ground-truth NPU verification.
//
// Shape (deliberately simpler than the lit pin's mismatched-stress form):
//   * Single shim MM2S channel @in_chan
//   * ONE configure_task with BD outermost dim <size = 4, stride = 0>
//     (IRON dma_repeat encoding — same 64-byte block sent four times,
//     BD does NOT advance per iteration) AND IRON-explicit
//     {repeat_count = 4 : i32} matched to outer.size (production form;
//     `--dma-task-to-conduit` lifts these so they agree by construction).
//   * Per-dispatch shim MM2S transfer = 4 × 64 bf16 = 256 bf16 bytes
//     pushed into the L1 fifo.
//
// Single column, npu2_1col target.  One compute tile at (0,2) does an
// identity copy from @in_chan to @out_chan (no fp_to_bf16 — purely
// memref load/store of bf16, so chess could in principle compile this
// kernel, but we route through Peano per sibling convention and to keep
// the REQUIRES line uniform).  Per dispatch the core loops 4 iterations
// (consume 64 bf16, produce 64 bf16) — driven entirely by data arrival
// from the IRON-encoded shim MM2S.  Output shim S2MM @out_chan is a
// single 256-bf16 transfer per dispatch (receives the 4 × 64-byte
// chunks back to back; no special outer encoding on the S2MM side, to
// keep the test focused on the input-side stride==0 encoding).
//
// Reference (test.cpp computes this):
//   in[j]  = (j % 16),  j ∈ [0, 64)
//   out[i] = in[i % 64], i ∈ [0, 256)   ⇒ four copies of `in` concatenated.
// All values are bf16-exact (integers in [0, 256) are exact in bf16).
//
// PASS expectation:
//   * Every dispatch (NUM_INVOCATIONS=4) completes without timeout.
//   * Output bytes match the four-copy reference exactly.
//
// Failure semantics (ground-truth diagnostic):
//   * UNDER-FIRE (e.g. only 1 of 4 transfers, output[0:64] correct,
//     output[64:256] zero) ⇒ firmware push_queue counter is 0-indexed
//     for this path (so emitting `repeat_count = N` actually fires N+1
//     transfers — counterintuitive but a real concern documented in
//     other Pass C emit sites).  Implication: v4's case-(d) emit
//     `repeat_count = outerSize` would need a -1 adjustment, and the
//     v4 design needs a bigger redesign than just "discriminate on
//     outer.STRIDE".  The lit pin would still PASS textually but the
//     emit value would be off-by-one in HW.
//   * OVER-FIRE (output[0:256] correct + extra bytes spilled past 256
//     OR kernel hang at second invocation due to push_queue overflow)
//     ⇒ BD walks N times AND push-queue fires N more times; both
//     mechanisms multiplicative.  Implication: v4's case-(d) limb is
//     wrong, push-queue should be 1 for stride==0 too (collapsing to
//     case-(c) / no-discriminator behavior — the pre-revise gate-2a
//     fix was right and v4's stride-discriminator hypothesis was wrong).
//   * WRONG-BYTES MID-BUFFER (correct count, scrambled values)
//     ⇒ shim S2MM L1↔L3 chunk-aggregation race (unrelated to the
//     stride==0 encoding under test); rerun with depth bump on
//     @out_chan to eliminate.
//
// Sibling references (read in this order to understand context):
//   1. test/Dialect/Conduit/passc_effective_repeat_iron_stride_zero_encoding.mlir
//      (lit-only pin — IR-level discrimination; this file is the e2e leg)
//   2. test/Dialect/Conduit/passc_effective_repeat_no_double_count.mlir
//      (lit-only pin for the OPPOSITE case: stride > 0 → push-queue=1)
//   3. test/npu-xrt/fuse_operators_convergent_npu/{aie.mlir,test.cpp,conduit.lit}
//      (canonical 3-file lit-style NPU smoke template this file mirrors)
//   4. test/npu-xrt/fuse_core_bodies_npu/{aie.mlir,test.cpp,conduit.lit}
//      (sibling lit-NPU smoke; per-invocation memset + verify pattern)

module {
  aie.device(NPUDEVICE) {

    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // Input channel: shim MM2S → compute consume.  This is the channel
    // whose runtime-sequence configure carries the IRON stride==0 outer
    // dim under test.
    aie.objectfifo @in_chan (%shim, {%tile}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Output channel: compute produce → shim S2MM.  Plain BD on the
    // shim side (no special encoding) — receives 4 × 64-bf16 chunks
    // back to back per dispatch.
    aie.objectfifo @out_chan (%tile, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    %core = aie.core(%tile) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 0xFFFFFE : index
      scf.for %niter = %c0 to %cmax step %c1 {
        %sv_in = aie.objectfifo.acquire @in_chan (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_in = aie.objectfifo.subview.access %sv_in[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_out = aie.objectfifo.acquire @out_chan (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_out = aie.objectfifo.subview.access %sv_out[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        // Identity copy — bf16 load/store, no float ops.  Each loop iter
        // moves one 64-bf16 chunk from L1 input slot to L1 output slot.
        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_in[%j]  : memref<64xbf16>
          memref.store %v, %elem_out[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @out_chan (Produce, 1)
        aie.objectfifo.release @in_chan  (Consume, 1)
      }
      aie.end
    }

    // Per-dispatch runtime sequence:
    //   * @in_chan configure carries the IRON stride==0 encoding.  Outer
    //     dim <size = 4, stride = 0> means the shim MM2S BD does NOT
    //     advance per iteration (sends the same 64-byte block four
    //     times).  Lower three placeholder slots <1,0> + inner
    //     <64,1>; lower-3 product (1×1×64) == BD len (64) ✓.
    //     `repeat_count = 4` matches outer.size (production form;
    //     post-`--dma-task-to-conduit` these always agree).  No
    //     issue_token — MM2S does not consume tokens.
    //   * @out_chan configure: plain 256-byte BD (no outer encoding) —
    //     shim S2MM single-shot transfer that receives the 4 × 64-byte
    //     chunks the core produces.  issue_token = true so the host can
    //     await completion before reading the BO.
    aie.runtime_sequence(%in : memref<64xbf16>, %out : memref<256xbf16>) {
      %t_in = aiex.dma_configure_task_for @in_chan {
        aie.dma_bd(%in : memref<64xbf16>, 0, 64,
          [<size = 4, stride = 0>,
           <size = 1, stride = 0>,
           <size = 1, stride = 0>,
           <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 4 : i32}
      aiex.dma_start_task(%t_in)

      %t_out = aiex.dma_configure_task_for @out_chan {
        aie.dma_bd(%out : memref<256xbf16>, 0, 256) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_in)
    }
  }
}
