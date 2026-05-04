//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma %s | FileCheck %s
// Metafix Candidate 1 (basic Path C): also smoke through downstream
// shim-allocation-substitution + BD-ID assignment passes.
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// Basic Path C (2026-04-25, Task #18) — multi-channel variant.  IRON
// emits per-iteration dma_await_task on S2MM (out_data) and dma_free_task
// on MM2S (in_data); both survive through --dma-task-to-conduit as
// conduit.wait_all{token = true|false} and are lowered by Step 8g back to
// inline aiex.dma_await_task / aiex.dma_free_task at the wait_all source
// locations — preserving IRON's per-iteration release shape.
//
// Source order: in_a, out_a, await_out_a, free_in_a,
//               in_b, out_b, await_out_b, free_in_b,
//               in_c, out_c, await_out_c, free_in_c.
//
// Expected output shape:
//   configure_in_a, start
//   configure_out_a {issue_token = true}, start
//   await_out_a            (inline, from wait_all{token=true})
//   free_in_a              (inline, from wait_all{token=false}; MM2S
//                           configure NOT stamped with issue_token because
//                           wait_all{token=false} does not require it)
//   ... iteration b
//   ... iteration c
//   (no trailing releases — every task in releasedTasks)

// CHECK-LABEL: module @step8g_multi_channel_interleaved
// CHECK:       aie.runtime_sequence

// Iteration 1.
// CHECK:           [[INA:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @in_data
// CHECK:           aiex.dma_start_task([[INA]])
// CHECK:           [[OUTA:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @out_data
// CHECK:           } {issue_token = true
// CHECK:           aiex.dma_start_task([[OUTA]])
// CHECK:           aiex.dma_await_task([[OUTA]])
// CHECK-NEXT:      aiex.dma_free_task([[INA]])

// Iteration 2.
// CHECK:           [[INB:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @in_data
// CHECK:           aiex.dma_start_task([[INB]])
// CHECK:           [[OUTB:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @out_data
// CHECK:           } {issue_token = true
// CHECK:           aiex.dma_start_task([[OUTB]])
// CHECK:           aiex.dma_await_task([[OUTB]])
// CHECK-NEXT:      aiex.dma_free_task([[INB]])

// Iteration 3.
// CHECK:           [[INC:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @in_data
// CHECK:           aiex.dma_start_task([[INC]])
// CHECK:           [[OUTC:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @out_data
// CHECK:           } {issue_token = true
// CHECK:           aiex.dma_start_task([[OUTC]])
// CHECK:           aiex.dma_await_task([[OUTC]])
// CHECK-NEXT:      aiex.dma_free_task([[INC]])

// No trailing-release block at end-of-rtSeq — every configured task is
// released by an explicit wait_all consumer.
// CHECK-NOT:       aiex.dma_await_task
// CHECK-NOT:       aiex.dma_free_task

module @step8g_multi_channel_interleaved {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    aie.objectfifo @in_data(%shim, {%tile}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @out_data(%tile, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @in_data(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @out_data(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @out_data(Produce, 1)
        aie.objectfifo.release @in_data(Consume, 1)
      }
      aie.end
    } {link_with = "kernel.a"}

    aie.runtime_sequence(%arg0: memref<384xbf16>, %arg1: memref<384xbf16>) {
      // Iteration 1
      %in0 = aiex.dma_configure_task_for @in_data {
        aie.dma_bd(%arg0 : memref<384xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%in0)
      %out0 = aiex.dma_configure_task_for @out_data {
        aie.dma_bd(%arg1 : memref<384xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out0)
      aiex.dma_await_task(%out0)
      aiex.dma_free_task(%in0)

      // Iteration 2
      %in1 = aiex.dma_configure_task_for @in_data {
        aie.dma_bd(%arg0 : memref<384xbf16>, 128, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%in1)
      %out1 = aiex.dma_configure_task_for @out_data {
        aie.dma_bd(%arg1 : memref<384xbf16>, 128, 128) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out1)
      aiex.dma_await_task(%out1)
      aiex.dma_free_task(%in1)

      // Iteration 3
      %in2 = aiex.dma_configure_task_for @in_data {
        aie.dma_bd(%arg0 : memref<384xbf16>, 256, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%in2)
      %out2 = aiex.dma_configure_task_for @out_data {
        aie.dma_bd(%arg1 : memref<384xbf16>, 256, 128) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out2)
      aiex.dma_await_task(%out2)
      aiex.dma_free_task(%in2)
    }
  }
}
