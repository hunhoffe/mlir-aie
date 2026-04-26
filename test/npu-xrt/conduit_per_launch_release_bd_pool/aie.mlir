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
// Llama LM-head BD-pool exhaustion pattern, minimized to a single shim
// channel (Task #15 npuxrt-extractor design; e2e companion to the
// lit-only pin in
//   test/Dialect/Conduit/conduit_to_dma_allocator_exhaustion_per_channel_pool.mlir).
//
// Runtime sequence: 4 launches, each with 5 same-channel MM2S
// dma_configure_task_for ops on @fifo_in followed by 1 S2MM
// dma_configure_task_for op on @fifo_out.  Within a launch the 5 input
// configures are issued + started first; the 5th is awaited
// (issue_token=true on the configure → MM2S+token is firmware-safe per
// the existing real-silicon CI in test/npu-xrt/sync_task_complete_token/);
// the 4 prior input BDs are then released via batched dma_free_task ops.
// The output drain is configure (issue_token=true) + start + await + free.
//
// Per-shim live BD intervals on the MM2S input channel:
//   - Stateful (no --use-conduit): per-launch awaits/frees release each
//     MM2S BD by end-of-launch, so at most 5 MM2S BDs are concurrently
//     live.  5 <= 16 (per-channel pool on npu2 shim) → the allocator
//     assigns IDs and the design runs end-to-end on hardware.
//   - Conduit Path B (--use-conduit, fix landed at e2ea0ab450):
//     --dma-task-to-conduit ERASES the inline await/free ops
//     (ConduitDmaTaskToConduit.cpp:48,403-415); --conduit-to-dma
//     re-emits ALL releases at end-of-rtSeq (Step 8g, "Path B"), so
//     20 same-channel MM2S BDs are concurrently live.  20 > 16 →
//     AIEAssignRuntimeSequenceBDIDs emits the per-channel pool
//     exhaustion error
//     (lib/Dialect/AIEX/Transforms/AIEAssignRuntimeSequenceBDIDs.cpp:106-113)
//     and the compile fails before any binary is produced.
//
// stateful.lit consumes this file with `aiecc.py` (no --use-conduit) and
// runs the resulting xclbin/insts on hardware.  conduit_path_b_xfail.lit
// consumes the same file with `aiecc.py --use-conduit` and FileCheck-pins
// the verbatim allocator diagnostic; once Path C lands (preserves
// inline releases via conduit.wait_all{token = bool}) the conduit lit
// will be rewritten to mirror stateful.lit's HW-execute structure.

module {
  aie.device(NPUDEVICE) {

    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)

    aie.objectfifo @fifo_in (%t00, {%t02}, 2 : i32)
        : !aie.objectfifo<memref<256xi32>>
    aie.objectfifo @fifo_out (%t02, {%t00}, 2 : i32)
        : !aie.objectfifo<memref<256xi32>>

    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c5 = arith.constant 5 : index
      %c256 = arith.constant 256 : index
      %c0_i32 = arith.constant 0 : i32
      %cmax = arith.constant 0xFFFFFE : index

      scf.for %niter = %c0 to %cmax step %c1 {
        %sv_out = aie.objectfifo.acquire @fifo_out (Produce, 1)
            : !aie.objectfifosubview<memref<256xi32>>
        %elem_out = aie.objectfifo.subview.access %sv_out[0]
            : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
        // Zero-init the output tile.
        scf.for %i = %c0 to %c256 step %c1 {
          memref.store %c0_i32, %elem_out[%i] : memref<256xi32>
        }
        // Sum 5 input tiles into the output tile.
        scf.for %k = %c0 to %c5 step %c1 {
          %sv_in = aie.objectfifo.acquire @fifo_in (Consume, 1)
              : !aie.objectfifosubview<memref<256xi32>>
          %elem_in = aie.objectfifo.subview.access %sv_in[0]
              : !aie.objectfifosubview<memref<256xi32>> -> memref<256xi32>
          scf.for %i = %c0 to %c256 step %c1 {
            %a = memref.load %elem_out[%i] : memref<256xi32>
            %b = memref.load %elem_in[%i] : memref<256xi32>
            %s = arith.addi %a, %b : i32
            memref.store %s, %elem_out[%i] : memref<256xi32>
          }
          aie.objectfifo.release @fifo_in (Consume, 1)
        }
        aie.objectfifo.release @fifo_out (Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @reduce(%input : memref<5120xi32>, %output : memref<1024xi32>) {

      // ============================== Launch 0 ==============================
      // Input slices 0..4 (offsets 0, 256, 512, 768, 1024) → output slice 0.
      %t0_0 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 0,    256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0_0)
      %t0_1 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 256,  256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0_1)
      %t0_2 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 512,  256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0_2)
      %t0_3 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 768,  256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0_3)
      %t0_4 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 1024, 256) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0_4)
      aiex.dma_await_task(%t0_4)
      aiex.dma_free_task(%t0_0)
      aiex.dma_free_task(%t0_1)
      aiex.dma_free_task(%t0_2)
      aiex.dma_free_task(%t0_3)

      // Launch-0 output drain.
      %r0 = aiex.dma_configure_task_for @fifo_out {
        aie.dma_bd(%output : memref<1024xi32>, 0, 256) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%r0)
      aiex.dma_await_task(%r0)
      aiex.dma_free_task(%r0)

      // ============================== Launch 1 ==============================
      // Input slices 5..9 (offsets 1280, 1536, 1792, 2048, 2304) → output slice 1.
      %t1_0 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 1280, 256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1_0)
      %t1_1 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 1536, 256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1_1)
      %t1_2 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 1792, 256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1_2)
      %t1_3 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 2048, 256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1_3)
      %t1_4 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 2304, 256) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1_4)
      aiex.dma_await_task(%t1_4)
      aiex.dma_free_task(%t1_0)
      aiex.dma_free_task(%t1_1)
      aiex.dma_free_task(%t1_2)
      aiex.dma_free_task(%t1_3)

      %r1 = aiex.dma_configure_task_for @fifo_out {
        aie.dma_bd(%output : memref<1024xi32>, 256, 256) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%r1)
      aiex.dma_await_task(%r1)
      aiex.dma_free_task(%r1)

      // ============================== Launch 2 ==============================
      // Input slices 10..14 (offsets 2560, 2816, 3072, 3328, 3584) → output slice 2.
      %t2_0 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 2560, 256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2_0)
      %t2_1 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 2816, 256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2_1)
      %t2_2 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 3072, 256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2_2)
      %t2_3 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 3328, 256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2_3)
      %t2_4 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 3584, 256) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2_4)
      aiex.dma_await_task(%t2_4)
      aiex.dma_free_task(%t2_0)
      aiex.dma_free_task(%t2_1)
      aiex.dma_free_task(%t2_2)
      aiex.dma_free_task(%t2_3)

      %r2 = aiex.dma_configure_task_for @fifo_out {
        aie.dma_bd(%output : memref<1024xi32>, 512, 256) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%r2)
      aiex.dma_await_task(%r2)
      aiex.dma_free_task(%r2)

      // ============================== Launch 3 ==============================
      // Input slices 15..19 (offsets 3840, 4096, 4352, 4608, 4864) → output slice 3.
      %t3_0 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 3840, 256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3_0)
      %t3_1 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 4096, 256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3_1)
      %t3_2 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 4352, 256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3_2)
      %t3_3 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 4608, 256) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3_3)
      %t3_4 = aiex.dma_configure_task_for @fifo_in {
        aie.dma_bd(%input : memref<5120xi32>, 4864, 256) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t3_4)
      aiex.dma_await_task(%t3_4)
      aiex.dma_free_task(%t3_0)
      aiex.dma_free_task(%t3_1)
      aiex.dma_free_task(%t3_2)
      aiex.dma_free_task(%t3_3)

      %r3 = aiex.dma_configure_task_for @fifo_out {
        aie.dma_bd(%output : memref<1024xi32>, 768, 256) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%r3)
      aiex.dma_await_task(%r3)
      aiex.dma_free_task(%r3)
    }
  }
}
