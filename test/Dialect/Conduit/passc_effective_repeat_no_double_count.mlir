//===- passc_effective_repeat_no_double_count.mlir -----------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Regression pin: --conduit-to-dma must NOT double-count repeat_count when
// IRON-emitted shim configures carry BOTH:
//   * an explicit `{repeat_count = K : i32}` attribute (surfaced onto
//     conduit.create's `dma_repeat = K` by --dma-task-to-conduit), AND
//   * a 4-dim BD whose OUTERMOST dim has `<size = N, stride > 0>` (a real
//     iteration, propagated onto consumer_dimensions for S2MM by
//     --dma-task-to-conduit).
//
// v5 fix (2026-04-30, ConduitToDMALower.cpp): collapse the
// effectiveRepeat computation to the uniform rule
//   effectiveRepeat = channelDmaRepeat
//   emit = effectiveRepeat - 1 (firmware push_queue is 0-indexed)
// regardless of outer-dim shape.  When the BD's outer dim has stride>0,
// the BD already walks the outer N times per fire; push-queue carries
// only the channel's intended fire count (1 for IRON `repeat_count = 1`),
// emit is `1 - 1 = 0`, firmware fires the BD once, total walks =
// outerSize × 1 = outerSize.  Pre-v5 history:
//   pre-fix: emit = max(K, N) → BD walked outerSize × max(K,N) =
//            multiplicative over-fire.  GEMM @ attn_query gate-2a
//            runtime hang root cause.
//   v4: stride-discriminator zeroed outerRepeat for stride>0 case
//            (push-queue = K).  But still emitted K verbatim, which
//            firmware read as K+1 fires.
//   v5: drop discriminator entirely + subtract-1 at emit.  Empirical
//       proof: #89 captured-IR rc=2→1 patch on this exact shape
//       byte-equivalent to stateful (0/2097152 differing bytes).
//
//===----------------------------------------------------------------------===//

// Two-line metafix convention catches dialect-verifier-only failures
// that the bare FileCheck line misses.
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// CHECK-LABEL: aie.device(npu2)

// Pass C consumes conduit.create + lowers to aie.shim_dma_allocation +
// aiex.dma_configure_task_for; the dma_repeat=1 on the conduit.create
// (set by --dma-task-to-conduit from IRON-explicit repeat_count=1) is
// the state being tested, but its observable effect lives entirely in
// the runtime configure's repeat_count field below.

// Pass C must emit effective repeat_count = 0 on each output
// configure_task (v5: subtract-1 at emit; user's IRON
// `repeat_count = 1` → 1 fire on hardware).  The MLIR printer elides
// default-zero integer attrs, so `repeat_count` does not appear in the
// textual output — verified absence is the correct pin.  The BD's
// outer wrap (size=2, stride=128) walks 2 iterations per fire, so
// total walks = 2 × 1 = 2 = the user's intent.  Pre-v5 this slot
// carried `repeat_count = 2` (= max(1, 2)), causing 4 walks per
// configure instead of the intended 2.
// CHECK:       aiex.dma_configure_task_for @C_L2L3_0_shim_alloc
// CHECK:         aie.dma_bd
// CHECK-SAME:    , 256
// CHECK-SAME:    <size = 2, stride = 128>
// CHECK:       } {issue_token = true, repeat_count = 1 : i32}

// Same for the second configure on the same channel (boundary-stress
// pattern: per-channel repeat behavior must hold uniformly across all
// IRON-emitted configures, not just the first).
// CHECK:       aiex.dma_configure_task_for @C_L2L3_0_shim_alloc
// CHECK:         aie.dma_bd
// CHECK-SAME:    , 256
// CHECK-SAME:    <size = 2, stride = 128>
// CHECK:       } {issue_token = true, repeat_count = 1 : i32}

module @passc_effective_repeat_no_double_count {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Compute → shim output channel (S2MM): tile produces, shim consumes.
    aie.objectfifo @C_L2L3_0(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<256xbf16>>

    func.func private @gemm_kernel(memref<256xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %out = aie.objectfifo.acquire @C_L2L3_0(Produce, 1)
            : !aie.objectfifosubview<memref<256xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
        func.call @gemm_kernel(%out_buf) : (memref<256xbf16>) -> ()
        aie.objectfifo.release @C_L2L3_0(Produce, 1)
      }
      aie.end
    } {link_with = "gemm.c"}

    // Two IRON-emitted configures on the SAME shim S2MM channel.  Each:
    //   * 4-dim BD with OUTERMOST dim <size=2, stride=128> (real iteration)
    //     plus two <1,0> placeholder slots and an inner <256,1> walk.
    //     Lower-3 product (1×1×256) == BD len (256) ✓
    //   * IRON-explicit {issue_token=true, repeat_count=1} (the canonical
    //     IRON GEMM-output shape).
    aie.runtime_sequence(%arg0: memref<1024xbf16>) {
      %t0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg0 : memref<1024xbf16>, 0, 256,
          [<size = 2, stride = 128>,
           <size = 1, stride = 0>,
           <size = 1, stride = 0>,
           <size = 256, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_free_task(%t0)

      %t1 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg0 : memref<1024xbf16>, 256, 256,
          [<size = 2, stride = 128>,
           <size = 1, stride = 0>,
           <size = 1, stride = 0>,
           <size = 256, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t1)
    }
  }
}
