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
// ArithProgressionPattern HW smoke for --conduit-canonicalize-channel-puts
// (Task #21).
//
// Pattern:
//   Runtime sequence issues N=4 same-channel MM2S dma_configure_task_for ops
//   on @fifo_in with sliding offsets [0, 64, 128, 192] (1D arith progression,
//   slice-stride = 64 bf16).  Each task: configure {issue_token = true} +
//   start + await + free.  This mirrors IRON's `for batch in range(4)`
//   loop-unroll over a strided ramp into a single channel — the exact host
//   pattern that, pre-canon, exploded the compute-tile aie.mem chain at
//   Llama op7_GEMV / op11_GEMV (CLAUDE.md "Active Open Bugs" HIGH row).
//
// Mirrors the canon's lit pins under
//   test/Dialect/Conduit/canonicalize_channel_puts/<arith-progression>.mlir.
// Stateful (no --use-conduit) lowers each dma_configure_task_for as 4 BDs.
// Conduit (--use-conduit) round-trips into 4 conduit.put_memref_async ops,
// canon detects the linear arith progression and collapses to 1 put with
// dma_repeat = 4 + a stride attribute on @fifo_in.  Pass C emits a shim BD
// with repeat_count = 4 and the appropriate iter dim, keeping the
// compute-tile chain under the 16-block cap.
//
// Output side: 4 S2MM gets at offsets [0, 64, 128, 192] of output<256xbf16>.
//
// HW-shape note: spec called for offsets [0, 8, 16, 24] (slice = 8 bf16) as
// a "small-shape mock".  Scaled up to slice-stride = 64 bf16 (128 bytes)
// for npu2 shim BD-min-length headroom.  Canon match condition keys on
// structural arith progression (sizes/strides identical, offsets linear in
// step), NOT on absolute element counts; so this scaling does not change
// what the pattern detects.
//
// Reference (test.cpp computes this):
//   output[i*64 + j] = input[i*64 + j]  for i in [0,4), j in [0,64)
// (i.e. output is a byte-identical copy of the 256-bf16 input ramp).
// All values 0..255 are bf16-exact (integers in [0, 256) fit in sign + 8
// exp + 7 mant).

module {
  aie.device(NPUDEVICE) {
    %shim_0   = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo_in (%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @fifo_out (%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Compute core: infinite copy loop.  One acquire-in + one acquire-out +
    // bf16-elementwise copy + release both.  Host runtime sequence drives
    // dispatch count (= 4 for this test).
    %core_0_2 = aie.core(%tile_0_2) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 0xFFFFFE : index
      scf.for %niter = %c0 to %cmax step %c1 {
        %sv_in = aie.objectfifo.acquire @fifo_in (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_in = aie.objectfifo.subview.access %sv_in[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_out = aie.objectfifo.acquire @fifo_out (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_out = aie.objectfifo.subview.access %sv_out[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_in[%j]  : memref<64xbf16>
          memref.store %v, %elem_out[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @fifo_in  (Consume, 1)
        aie.objectfifo.release @fifo_out (Produce, 1)
      }
      aie.end
    }

    // 4 same-channel MM2S puts at sliding offsets [0, 64, 128, 192].
    aie.runtime_sequence @arith_progression_1d(%input  : memref<256xbf16>,
                                               %output : memref<256xbf16>) {
      // ---- Input puts: 4x sliding by 64 across input<256xbf16> ----
      %ti0 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<256xbf16>, 0,   64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti0)
      aiex.dma_await_task(%ti0)
      aiex.dma_free_task(%ti0)

      %ti1 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<256xbf16>, 64,  64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti1)
      aiex.dma_await_task(%ti1)
      aiex.dma_free_task(%ti1)

      %ti2 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<256xbf16>, 128, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti2)
      aiex.dma_await_task(%ti2)
      aiex.dma_free_task(%ti2)

      %ti3 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<256xbf16>, 192, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti3)
      aiex.dma_await_task(%ti3)
      aiex.dma_free_task(%ti3)

      // ---- Output gets: 4x sliding by 64 across output<256xbf16> ----
      %to0 = aiex.dma_configure_task_for @fifo_out {
        aie.dma_bd(%output : memref<256xbf16>, 0,   64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to0)
      aiex.dma_await_task(%to0)
      aiex.dma_free_task(%to0)

      %to1 = aiex.dma_configure_task_for @fifo_out {
        aie.dma_bd(%output : memref<256xbf16>, 64,  64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to1)
      aiex.dma_await_task(%to1)
      aiex.dma_free_task(%to1)

      %to2 = aiex.dma_configure_task_for @fifo_out {
        aie.dma_bd(%output : memref<256xbf16>, 128, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to2)
      aiex.dma_await_task(%to2)
      aiex.dma_free_task(%to2)

      %to3 = aiex.dma_configure_task_for @fifo_out {
        aie.dma_bd(%output : memref<256xbf16>, 192, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to3)
      aiex.dma_await_task(%to3)
      aiex.dma_free_task(%to3)
    }
  }
}
