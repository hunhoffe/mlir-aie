//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma %s | FileCheck %s

// Direction 1a (FS7 followup, Task #62) multi-channel interleaved variant:
// per-op release on channel X is independent of channel Y.  Source order:
//   in_1, out_1, in_2, out_2, in_3, out_3
// Step 8g per-channel state should produce:
//   configure_in_a, start
//   configure_out_a, start
//   free_in_a,  configure_in_b, start
//   await_out_a, configure_out_b, start
//   free_in_b,  configure_in_c, start
//   await_out_b, configure_out_c, start
//   trailing: free_in_c, await_out_c
// MM2S releases via dma_free_task; S2MM (issue_token=true) releases via
// dma_await_task — matching the established convention from
// conduit_to_dma_await_all_outputs.mlir and IRON's emission shape.

// CHECK-LABEL: module @step8g_multi_channel_interleaved
// CHECK:       aie.runtime_sequence

// Per-channel + per-op interleaving asserted by the ordered CHECK
// block below.  Total expected ops (asserted indirectly via the
// ordered shape): 6 configures (3 per channel), 3 frees (in_data
// MM2S releases), 3 awaits (out_data S2MM releases).

// First in/out pair: configure+start, no preceding release on either channel.
// CHECK:           [[IN0:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @in_data
// CHECK:           aiex.dma_start_task([[IN0]])
// CHECK-NEXT:      [[OUT0:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @out_data
// CHECK:           aiex.dma_start_task([[OUT0]])

// Second in: free of [[IN0]] precedes new configure (channel-local
// release).  No await of [[OUT0]] yet (cross-channel independence).
// CHECK-NEXT:      aiex.dma_free_task([[IN0]])
// CHECK-NEXT:      [[IN1:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @in_data
// CHECK:           aiex.dma_start_task([[IN1]])

// Second out: await of [[OUT0]] precedes new configure (S2MM release).
// CHECK-NEXT:      aiex.dma_await_task([[OUT0]])
// CHECK-NEXT:      [[OUT1:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @out_data
// CHECK:           aiex.dma_start_task([[OUT1]])

// Third in: free of [[IN1]] precedes new configure.
// CHECK-NEXT:      aiex.dma_free_task([[IN1]])
// CHECK-NEXT:      [[IN2:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @in_data
// CHECK:           aiex.dma_start_task([[IN2]])

// Third out: await of [[OUT1]] precedes new configure.
// CHECK-NEXT:      aiex.dma_await_task([[OUT1]])
// CHECK-NEXT:      [[OUT2:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @out_data
// CHECK:           aiex.dma_start_task([[OUT2]])

// Trailing release: one per channel, in first-seen order (in_data then
// out_data).
// CHECK:           aiex.dma_free_task([[IN2]])
// CHECK-NEXT:      aiex.dma_await_task([[OUT2]])

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
