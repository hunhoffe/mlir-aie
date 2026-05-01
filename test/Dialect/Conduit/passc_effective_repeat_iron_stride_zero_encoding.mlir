//===- passc_effective_repeat_iron_stride_zero_encoding.mlir --*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Regression pin: --conduit-to-dma must NOT collapse the IRON-documented
// `repeat_count`-as-stride-0-outer-dim encoding to push-queue=1 — that
// loses the iteration count entirely.
//
// Per AIEDmaToNpu.cpp:385-388 ("We allow users to encode the
// repeat_count as a dimension 3 stride of 0"), an outermost BD dim of
// `<size = N, stride = 0>` is a documented IRON convention for
// expressing repeat_count via the BD shape.  In this encoding the BD
// does NOT advance per iteration (stride==0 ⇒ no address motion), so
// the BD descriptor's iteration_size does NOT independently fire the
// chain — the push-queue is the actual counter.
//
// Counterpart pin `passc_effective_repeat_no_double_count.mlir` covers
// the OPPOSITE case: outer with stride > 0 (real iteration), where the
// BD walks N times and push-queue must be 1 to avoid multiplicative
// over-fire (the GEMM @ attn_query gate-2a hang root cause).
//
// Bug history (re-fixed 2026-04-30): the gate-2a fix initially
// discriminated only on outer.SIZE > 1, collapsing BOTH cases under
// "BD walks → push-queue=1".  That broke this stride==0 encoding
// (surfaced by the gate 2a A/B re-verify on `@A_L3L2_0`: producer_
// dimensions outer = `<size=4, stride=0>` + channel dma_repeat=3 →
// pre-revise erroneously errored on "both mechanisms"; correct
// behavior is to read outer.size as the iteration count and emit
// push-queue=4).  Revised fix discriminates on outer.STRIDE.
//
// Boundary stress: outer.size (4) and channel dma_repeat (3) DELIBERATELY
// differ here.  In production these agree by construction (--dma-task-to
// -conduit lifts one to the other).  We pick mismatched values so the
// CHECK below proves outer.size is the read source — emitting `4` (not
// `3`) confirms case (d) discrimination.  The mismatch is a fixture-only
// stress, not a real-IR shape.
//
//===----------------------------------------------------------------------===//

// Two-line metafix convention catches dialect-verifier-only failures
// that the bare FileCheck line misses.
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// CHECK-LABEL: aie.device(npu2)

// Pass C must emit `repeat_count = 2` on each output configure_task —
// v5 (2026-04-30) collapses to a uniform rule:
//   effectiveRepeat = channelDmaRepeat (no stride discriminator)
//   emit = effectiveRepeat - 1 (firmware push_queue is 0-indexed)
// With IRON-explicit `repeat_count = 3` flowing through --dma-task-to-
// conduit onto channel dma_repeat = 3, Pass C subtracts 1 at the
// configure_task emit site → `repeat_count = 2 : i32`, which firmware
// reads as 3 BD fires per dispatch (= the user's intent).
//
// The OUTER `<size = 4, stride = 0>` BD dim is preserved verbatim in
// the BD shape; v5 no longer reads it as an additional iteration
// source (the v4 stride-discriminator was vindicated by an early NPU
// smoke that, on later inspection, was structurally rc-insensitive —
// see #93 → today's #94 redesign).
//
// History of this slot:
//   pre-v3: erroneously errored on "both mechanisms" (defensive)
//   v3 partial: collapsed stride==0 to push-queue=1, lost the count
//   v4: case-d outerSize preservation → emitted `repeat_count = 4`
//       (still over-fired by 1 in firmware; today's gate-2a finding)
//   v5: subtract-1 + uniform rule → emit `repeat_count = 2`
//
// Boundary stress: outer.size (4) and channel dma_repeat (3) DELIB-
// ERATELY differ (production-form invariant has them agree).  The
// CHECK proves channelDmaRepeat is the read source under v5 (emitting
// `2` confirms the value is `dma_repeat - 1 = 3 - 1`, not `outer.size
// - 1 = 3` and not `outer.size = 4` from v4).
// CHECK:       aiex.dma_configure_task_for @A_L3L2_0_shim_alloc
// CHECK:         aie.dma_bd
// CHECK-SAME:    , 256
// CHECK-SAME:    <size = 4, stride = 0>
// CHECK:       } {{.*}}repeat_count = 3 : i32
// CHECK-NOT:   repeat_count = 2
// CHECK-NOT:   repeat_count = 4

// Same for the second configure on the same channel (boundary-stress
// pattern: per-channel repeat behavior must hold uniformly across all
// IRON-emitted configures, not just the first).
// CHECK:       aiex.dma_configure_task_for @A_L3L2_0_shim_alloc
// CHECK:         aie.dma_bd
// CHECK-SAME:    , 256
// CHECK-SAME:    <size = 4, stride = 0>
// CHECK:       } {{.*}}repeat_count = 3 : i32
// CHECK-NOT:   repeat_count = 2
// CHECK-NOT:   repeat_count = 4

module @passc_effective_repeat_iron_stride_zero_encoding {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Shim → compute input channel (MM2S at shim): shim produces, tile
    // consumes.  Producer-side BD dims live on the shim configure_task.
    aie.objectfifo @A_L3L2_0(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<256xbf16>>

    func.func private @gemm_kernel(memref<256xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @A_L3L2_0(Consume, 1)
            : !aie.objectfifosubview<memref<256xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
        func.call @gemm_kernel(%in_buf) : (memref<256xbf16>) -> ()
        aie.objectfifo.release @A_L3L2_0(Consume, 1)
      }
      aie.end
    } {link_with = "gemm.c"}

    // Two IRON-emitted configures on the SAME shim MM2S channel.  Each:
    //   * 4-dim BD with OUTERMOST dim <size=4, stride=0> (IRON
    //     dma_repeat encoding — BD walks SAME 256-byte block four
    //     times, no address advance between iterations).  Lower-3
    //     placeholder slots <1,0> + inner <256,1>; lower-3 product
    //     (1×1×256) == BD len (256) ✓
    //   * IRON-explicit {repeat_count = 3} (deliberately ≠ outer.size
    //     to prove outer.size is the read source post-revise).
    //   * No issue_token — MM2S does not consume tokens (only S2MM).
    aie.runtime_sequence(%arg0: memref<1024xbf16>) {
      %t0 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1024xbf16>, 0, 256,
          [<size = 4, stride = 0>,
           <size = 1, stride = 0>,
           <size = 1, stride = 0>,
           <size = 256, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_free_task(%t0)

      %t1 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<1024xbf16>, 256, 256,
          [<size = 4, stride = 0>,
           <size = 1, stride = 0>,
           <size = 1, stride = 0>,
           <size = 256, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t1)
    }
  }
}
