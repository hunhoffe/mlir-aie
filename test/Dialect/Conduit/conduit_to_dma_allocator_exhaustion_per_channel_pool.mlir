//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --verify-diagnostics --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// Path B end-to-end allocator-exhaustion regression (2026-04-25):
// --conduit-to-dma Step 8g (Path B, supersedes FS7) emits NO inline
// release between same-channel configures, so per-channel BD-ID live
// intervals span [configure ... trailing-release] with all configures
// preceding all releases.  When same-channel configures exceed the
// per-channel BD pool (16 BDs on a shim DMA channel for npu2), the
// AIEAssignRuntimeSequenceBDIDs allocator emits the diagnostic at
// AIEAssignRuntimeSequenceBDIDs.cpp:106-113 ("Allocator exhausted
// available buffer descriptor IDs for channel ... Live BD intervals
// exceed the per-channel pool capacity; interleave aiex.dma_await_task
// / aiex.dma_free_task with configures to recycle IDs.").
//
// This test pins that error-on-exhaustion behavior at the smallest
// triggering size: 17 same-channel MM2S invocations through Pass C.
// The 17th configure (after Pass C lowering preserves source
// locations) is the one the allocator can no longer accommodate.
//
// This pin is the structural counterpart to the rewritten
// step8g_single_channel_per_op_release.mlir and
// step8g_multi_channel_interleaved.mlir tests: those pin the no-inline-
// release shape; this pins the allocator's correct response when that
// shape exceeds pool capacity.

module @allocator_exhaustion_per_channel_pool {
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

    aie.runtime_sequence(%arg0: memref<2176xbf16>) {
      // 17 invocations of the same MM2S channel.  16 BDs on shim
      // channel — the 17th must trigger the allocator's exhaustion
      // error.  Source locations are preserved through
      // --dma-task-to-conduit and --conduit-to-dma, so the
      // expected-error annotation on the 17th input configure matches
      // the diagnostic emitted on the lowered op of the same loc.
      %t0 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 128, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 256, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 384, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3)
      %t4 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 512, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t4)
      %t5 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 640, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t5)
      %t6 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 768, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t6)
      %t7 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 896, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t7)
      %t8 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1024, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t8)
      %t9 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1152, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t9)
      %t10 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1280, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t10)
      %t11 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1408, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t11)
      %t12 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1536, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t12)
      %t13 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1664, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t13)
      %t14 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1792, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t14)
      %t15 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1920, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t15)
      // expected-error@+1 {{Allocator exhausted available buffer descriptor IDs for channel}}
      %t16 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 2048, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t16)
    }
  }
}
