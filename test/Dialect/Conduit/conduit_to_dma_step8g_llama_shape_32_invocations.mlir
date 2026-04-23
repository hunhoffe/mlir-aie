//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// Llama-shape regression for FS7 followup (Task #62, Direction 1a).
//
// Mirrors the IRON emission pattern for @op7_A_L3L1_0 in the Llama
// fused_op runlist: 32 MM2S invocations of one shim channel, each
// reading num_elems=16384 (bf16) from a 4194304-element host argument
// at stride-131072 byte offsets (= 16384 elems × 8 bytes per elem-in-bf16
// when collapsed; the offsets here are in *element* units to match
// objectfifo conventions).
//
// 32 > 16-BD shim pool, so the OLD batched-at-end Step 8g shape would
// fail at --aie-assign-runtime-sequence-bd-ids with "Allocator exhausted
// available buffer descriptor IDs".  Direction 1a's per-channel
// configure/free interleaving keeps each per-channel BD interval
// trivially short, so the allocator picks the lowest free id (0) for
// every invocation.

// CHECK-LABEL: module @step8g_llama_shape_32_invocations

// All 32 BDs reuse bd_id = 0 — per-channel intervals are disjoint, so
// the lowest free id is always 0.
// CHECK-COUNT-32: aie.dma_bd(%{{.*}}, 16384) {bd_id = 0 : i32, burst_length = 0 : i32}
// (CHECK-NOT intentionally omitted: core-side mem region also emits
//  aie.dma_bd ops on the consumer-side double-buffer; those are
//  unrelated to the runtime-sequence BD-pool recycling under test.)

module @step8g_llama_shape_32_invocations {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    aie.objectfifo @op7_A_L3L1_0(%shim, {%tile}, 2 : i32)
        : !aie.objectfifo<memref<16384xbf16>>

    func.func private @kernel(memref<16384xbf16>)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @op7_A_L3L1_0(Consume, 1)
            : !aie.objectfifosubview<memref<16384xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<16384xbf16>> -> memref<16384xbf16>
        func.call @kernel(%in_buf) : (memref<16384xbf16>) -> ()
        aie.objectfifo.release @op7_A_L3L1_0(Consume, 1)
      }
      aie.end
    } {link_with = "kernel.a"}

    aie.runtime_sequence(%arg0: memref<4194304xbf16>) {
      %t0 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 0, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 131072, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 262144, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 393216, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3)
      %t4 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 524288, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t4)
      %t5 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 655360, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t5)
      %t6 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 786432, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t6)
      %t7 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 917504, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t7)
      %t8 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 1048576, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t8)
      %t9 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 1179648, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t9)
      %t10 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 1310720, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t10)
      %t11 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 1441792, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t11)
      %t12 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 1572864, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t12)
      %t13 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 1703936, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t13)
      %t14 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 1835008, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t14)
      %t15 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 1966080, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t15)
      %t16 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 2097152, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t16)
      %t17 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 2228224, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t17)
      %t18 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 2359296, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t18)
      %t19 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 2490368, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t19)
      %t20 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 2621440, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t20)
      %t21 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 2752512, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t21)
      %t22 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 2883584, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t22)
      %t23 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 3014656, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t23)
      %t24 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 3145728, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t24)
      %t25 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 3276800, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t25)
      %t26 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 3407872, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t26)
      %t27 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 3538944, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t27)
      %t28 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 3670016, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t28)
      %t29 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 3801088, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t29)
      %t30 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 3932160, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t30)
      %t31 = aiex.dma_configure_task_for @op7_A_L3L1_0 {
        aie.dma_bd(%arg0 : memref<4194304xbf16>, 4063232, 16384) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t31)
    }
  }
}
