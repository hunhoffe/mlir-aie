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
// NPU end-to-end regression for Pass C `effectiveRepeat` double-counting
// (companion to test/Dialect/Conduit/passc_effective_repeat_no_double_count.mlir).
//
// Bug shape (fixed in ConduitToDMALower.cpp at the
// `effectiveRepeat = std::max(channelDmaRepeat, outerRepeat)` site):
// when an IRON-emitted shim S2MM configure carries BOTH
//   * an explicit `{repeat_count = 1 : i32}` attribute on the
//     `aiex.dma_configure_task_for` (surfaced onto the Conduit channel as
//     `dma_repeat = 1` by --dma-task-to-conduit), AND
//   * a 4-dim BD whose OUTERMOST dim has `<size = N, stride > 0>` (a real
//     iteration, propagated onto consumer_dimensions for S2MM by
//     --dma-task-to-conduit),
// the pre-fix `std::max()` over-fired the post-Pass-C output:
// `repeat_count = max(1, N) = N` on the push-queue PLUS the BD's own outer
// dim N — the two mechanisms MULTIPLY in firmware, so the BD walked
// `N * N = N²` times instead of the intended `N`.  At N=2 (the shape
// below) the consumer drained 4 producer slots per configure, the BD
// re-walked offsets [0, 1024 bytes] twice, and the second walk
// OVERWROTE the first walk's data.
//
// This fixture pins post-fix behavior end-to-end by:
//   * compute(0,2) producing a deterministic monotonic-iter pattern
//     (each acquire/release writes `i32(iter)` to all 256 entries of the
//     slot; iter starts at 0 fresh per PDI load and advances by 1 per
//     produce);
//   * shim(0,0) S2MM draining via TWO IRON-emitted configure_task ops on
//     the same channel, each with the bug-triggering shape (4-dim BD
//     outer `<size=2, stride=256>` + IRON-explicit `repeat_count=1`),
//     base offsets 0 and 512 (covering the full 1024 i32 host buffer
//     non-overlapping).
//
// Post-fix expected output (deterministic, derived from first principles):
//   * Configure 1 fires twice, walks output[0..256] and output[256..512],
//     drains producer iters 0,1 → output[0..256]=0, output[256..512]=1.
//   * Configure 2 fires twice, walks output[512..768] and output[768..1024],
//     drains producer iters 2,3 → output[512..768]=2, output[768..1024]=3.
// Pre-fix output (what test.cpp catches):
//   * Configure 1 fires FOUR times (push-queue=2 × BD-outer=2), drains
//     producer iters 0,1,2,3, second BD walk overwrites first →
//     output[0..256]=2, output[256..512]=3.
//   * Configure 2 fires FOUR times, drains iters 4,5,6,7 →
//     output[512..768]=6, output[768..1024]=7.
//
// Producer-pre-cycling robustness: the FIFO queue is FIFO-ordered, so
// the consumer always drains the OLDEST producer entry first regardless
// of how many slots the producer pre-filled before the host fired the
// runtime sequence.  PDI load resets the core (iter starts at 0), so
// the absolute iter values seen by the host are deterministic.

module {
  aie.device(NPUDEVICE) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)

    aie.objectfifo @fifo_out (%t02, {%t00}, 2 : i32)
        : !aie.objectfifo<memref<256xi32>>

    aie.core(%t02) {
      %c0    = arith.constant 0 : index
      %c1    = arith.constant 1 : index
      %c256  = arith.constant 256 : index
      %cmax  = arith.constant 0xFFFFFE : index

      scf.for %niter = %c0 to %cmax step %c1 {
        %sv = aie.objectfifo.acquire @fifo_out (Produce, 1)
            : !aie.objectfifosubview<memref<256xi32>>
        %elem = aie.objectfifo.subview.access %sv[0]
            : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
        // Per-iter pattern: broadcast i32(iter) across the 256-elem slot.
        // Monotonic + deterministic-on-PDI-reset → host can pin the
        // exact post-fix expected values without ambiguity.
        %iter_i32 = arith.index_cast %niter : index to i32
        scf.for %i = %c0 to %c256 step %c1 {
          memref.store %iter_i32, %elem[%i] : memref<256xi32>
        }
        aie.objectfifo.release @fifo_out (Produce, 1)
      }
      aie.end
    }

    // Two IRON-emitted shim S2MM configures, identical bug shape:
    //   * 4-dim BD with OUTER <size = 2, stride = 256 elements>
    //     (= 1024 bytes per walk for int32) + 2 placeholder <1,0> slots
    //     + inner contiguous <256, 1> walk.  Lower-3 product == BD len. ✓
    //   * IRON-explicit {issue_token = true, repeat_count = 1} (the
    //     canonical IRON GEMM-output shape that --dma-task-to-conduit
    //     surfaces onto the Conduit channel as dma_repeat = 1).
    aie.runtime_sequence @drain(%output : memref<1024xi32>) {
      %t0 = aiex.dma_configure_task_for @fifo_out {
        aie.dma_bd(%output : memref<1024xi32>, 0, 256,
          [<size = 2, stride = 256>,
           <size = 1, stride = 0>,
           <size = 1, stride = 0>,
           <size = 256, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_free_task(%t0)

      %t1 = aiex.dma_configure_task_for @fifo_out {
        aie.dma_bd(%output : memref<1024xi32>, 512, 256,
          [<size = 2, stride = 256>,
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
