//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma %s | FileCheck %s

// Path B regression (2026-04-25, supersedes FS7/Task #62 inline-release
// design): --conduit-to-dma Step 8g must NOT emit any inline release
// (aiex.dma_free_task / aiex.dma_await_task) between same-channel
// configures.  All releases are batched at end-of-runtime_sequence,
// matching stateful's emission pattern.
//
// The prior eager-release design recycled BD IDs while still-firing
// shim BDs' queued repeat_count > 0 fires drained, leading to
// mid-flight register overwrite (bug class 3, N-stripe rollover at
// GEMM @ attn_query — fixed by this change).
//
// This single-channel test validates the trailing-release shape on the
// simplest case: 4 MM2S invocations of one objectfifo.  Expected output
// shape:
//   configure_1, start_1,
//   configure_2, start_2,
//   configure_3, start_3,
//   configure_4, start_4,
//   free_1, free_2, free_3, free_4   (all trailing, source order)
// i.e. 4 configures interleaved with NO releases, then 4 frees at end
// in source order.  AIEAssignRuntimeSequenceBDIDs will assign 4 distinct
// BD IDs from the per-channel pool (16 BDs on shim, easily fits 4).

// CHECK-LABEL: module @step8g_single_channel
// CHECK:       aie.runtime_sequence

// 4 configures + 4 starts in source order, no inline releases.
// CHECK:           [[T0:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           aiex.dma_start_task([[T0]])
// CHECK-NEXT:      [[T1:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           aiex.dma_start_task([[T1]])
// CHECK-NEXT:      [[T2:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           aiex.dma_start_task([[T2]])
// CHECK-NEXT:      [[T3:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           aiex.dma_start_task([[T3]])

// Trailing releases: 4 frees in source order (channel ext_in is MM2S,
// so dma_free_task; only one channel, so emission order matches source
// configure order).
// CHECK-NEXT:      aiex.dma_free_task([[T0]])
// CHECK-NEXT:      aiex.dma_free_task([[T1]])
// CHECK-NEXT:      aiex.dma_free_task([[T2]])
// CHECK-NEXT:      aiex.dma_free_task([[T3]])

module @step8g_single_channel {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    aie.objectfifo @ext_in(%shim, {%tile}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @kernel(memref<128xbf16>)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @kernel(%in_buf) : (memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_in(Consume, 1)
      }
      aie.end
    } {link_with = "kernel.a"}

    // Four invocations of the same channel.  IRON's pre-conduit
    // emission interleaves awaits/frees per iteration;
    // --dma-task-to-conduit drops the original sync ops, so Step 8g
    // reconstructs exactly one trailing release per configured task.
    aie.runtime_sequence(%arg0: memref<512xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<512xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<512xbf16>, 128, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<512xbf16>, 256, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<512xbf16>, 384, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3)
      aiex.dma_await_task(%t0)
      aiex.dma_await_task(%t1)
      aiex.dma_await_task(%t2)
      aiex.dma_await_task(%t3)
    }
  }
}
