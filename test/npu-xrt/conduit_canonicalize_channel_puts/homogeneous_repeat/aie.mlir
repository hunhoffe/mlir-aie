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
// HomogeneousRepeatPattern HW smoke for --conduit-canonicalize-channel-puts
// (Task #21; e2e companion to lit-only pins under
//   test/Dialect/Conduit/canonicalize_channel_puts/homogeneous_repeat_*.mlir).
//
// Pattern:
//   Runtime sequence issues N=4 same-channel MM2S dma_configure_task_for ops
//   on @fifo_in, ALL at offset 0 of the input buffer (offsets = [0, 0, 0, 0],
//   len = 64 bf16 each).  Each task: configure {issue_token = true} + start +
//   await + free.  This is the "pure repeat" shape (input window re-read 4x).
//
// Mirrors the canon's lit pin
//   homogeneous_repeat_collapse.mlir (4 structurally-identical puts on @chan).
// Stateful (no --use-conduit) lowers each dma_configure_task_for individually
// (4 BDs).  Conduit (--use-conduit) round-trips into 4 conduit.put_memref_async
// ops, then --conduit-canonicalize-channel-puts collapses to 1 put with
// dma_repeat = 4 on @fifo_in.  Pass C emits a single shim BD with
// repeat_count = 4, and the compute-tile S2MM consumer chain stays under
// the 16-block aie.mem cap.
//
// Output side: 4 S2MM dma_configure_task_for ops on @fifo_out at offsets
// [0, 64, 128, 192] of the 256-bf16 output, len = 64 each.  Output side is
// arith-progression-shaped; if AcquireReleasePattern symmetry is wired to
// get_memref_async it canonicalizes too — either way the byte-equivalence
// check below is the contract.
//
// HW-shape note: slice = 64 bf16 (128 bytes) is the smallest size that
// comfortably clears npu2 shim BD min-length.  The spec offset progression
// [0, 8, 16, 24] (slice = 8 bf16) was scaled up to slice-stride = 64 bf16
// for HW headroom; canon's match condition keys on structural identity of
// offsets/sizes/strides, NOT on absolute element counts.
//
// Reference (test.cpp computes this):
//   output[i*64 + j] = input[j]  for i in [0,4), j in [0,64)
// All values 0..63 are bf16-exact (integers fit in sign + 8 exp + 7 mant).

module {
  aie.device(NPUDEVICE) {
    %shim_0   = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @fifo_in (%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @fifo_out (%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Compute core: infinite copy loop.  Acquires one input slice and one
    // output slot, copies bf16-by-bf16, releases both.  The host runtime
    // sequence drives how many times the shim feeds it (= 4 for this test).
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

    // 4 same-channel MM2S puts, ALL at offset 0 (homogeneous repeat shape).
    // Output: 4 S2MM gets at offsets [0, 64, 128, 192] of output<256xbf16>.
    aie.runtime_sequence @homogeneous_repeat(%input  : memref<64xbf16>,
                                             %output : memref<256xbf16>) {
      // ---- Input puts: 4x identical at offset 0 ----
      %ti0 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti0)
      aiex.dma_await_task(%ti0)
      aiex.dma_free_task(%ti0)

      %ti1 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti1)
      aiex.dma_await_task(%ti1)
      aiex.dma_free_task(%ti1)

      %ti2 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ti2)
      aiex.dma_await_task(%ti2)
      aiex.dma_free_task(%ti2)

      %ti3 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
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
