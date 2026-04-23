//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// End-to-end regression: Direction 1a's per-op release in --conduit-to-dma
// Step 8g (FS7 followup, Task #62) makes per-channel BD-ID intervals
// trivially short ([configure, free] = adjacent positions), so the
// --aie-assign-runtime-sequence-bd-ids interval allocator (Task #55
// Option B) recycles a single BD ID across N invocations of one
// channel.
//
// Without Direction 1a, all N free_tasks were batched at end-of-rtSeq;
// the allocator saw N intervals all spanning the entire rtSeq body and
// would either (a) demand N distinct BD IDs (exhausting the 16-BD shim
// pool for N > 16) or (b) silently corrupt by reusing an in-flight ID.
//
// This test fires 20 invocations of one MM2S channel on a single shim
// tile.  20 > 16, so the OLD batched-at-end shape would FAIL with
// "Allocator exhausted available buffer descriptor IDs".  The new shape
// must succeed with bd_id = 0 reused across all 20 invocations.

// CHECK-LABEL: module @step8g_bd_pool_recycling

// All 20 BDs reuse the same id (0): per-channel intervals are disjoint,
// so the allocator picks the lowest free id (0) every time.
// CHECK-COUNT-20: aie.dma_bd(%{{.*}}, 128) {bd_id = 0 : i32, burst_length = 0 : i32}
// (CHECK-NOT intentionally omitted: core-side mem region also emits
//  aie.dma_bd ops on the consumer-side double-buffer; those are
//  unrelated to the runtime-sequence BD-pool recycling under test.)

module @step8g_bd_pool_recycling {
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

    aie.runtime_sequence(%arg0: memref<2560xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 128, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 256, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 384, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3)
      %t4 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 512, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t4)
      %t5 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 640, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t5)
      %t6 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 768, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t6)
      %t7 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 896, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t7)
      %t8 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 1024, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t8)
      %t9 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 1152, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t9)
      %t10 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 1280, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t10)
      %t11 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 1408, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t11)
      %t12 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 1536, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t12)
      %t13 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 1664, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t13)
      %t14 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 1792, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t14)
      %t15 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 1920, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t15)
      %t16 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 2048, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t16)
      %t17 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 2176, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t17)
      %t18 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 2304, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t18)
      %t19 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2560xbf16>, 2432, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t19)
    }
  }
}
