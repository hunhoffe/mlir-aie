//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma %s | FileCheck %s
// Metafix Candidate 1 (basic Path C): also smoke through the downstream
// shim-allocation-substitution + BD-ID assignment passes so dma_await_task
// legalization (issue_token = true on MM2S configures) is exercised here,
// not just at full-aiecc time.
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// Basic Path C (2026-04-25, Task #18): IRON's per-launch
// aiex.dma_await_task / aiex.dma_free_task in the source rtSeq survive
// through --dma-task-to-conduit as conduit.wait_all{token=true|false}, and
// --conduit-to-dma Step 8g lowers each wait_all back to an INLINE
// aiex.dma_await_task / aiex.dma_free_task at the wait_all's source
// location — preserving IRON's BD-release boundaries so the
// AIEAssignRuntimeSequenceBDIDs allocator's per-channel live intervals
// remain narrow ([configure ... explicit-release]).  Tasks released
// inline are skipped by the trailing-release loop to avoid double-release.
//
// For wait_all{token = true} consumers on MM2S configures, Step 8g also
// stamps `issue_token = true` on the configure_task — required by aiecc
// legalization (and firmware-safe per Task #5 investigation).
//
// This test pins the 4-invocation single-channel MM2S shape with explicit
// IRON awaits.  Expected output:
//   configure_1 {issue_token = true}, start_1
//   configure_2 {issue_token = true}, start_2
//   configure_3 {issue_token = true}, start_3
//   configure_4 {issue_token = true}, start_4
//   await_1, await_2, await_3, await_4   (inline at the original IRON
//                                         await positions; trailing-release
//                                         loop emits nothing because all
//                                         tasks are in releasedTasks)

// CHECK-LABEL: module @step8g_single_channel
// CHECK:       aie.runtime_sequence

// 4 configures + 4 starts in source order, with `issue_token = true`
// stamped on each MM2S configure (driven by the wait_all{token = true}
// consumers).
// CHECK:           [[T0:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           } {issue_token = true
// CHECK:           aiex.dma_start_task([[T0]])
// CHECK:           [[T1:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           } {issue_token = true
// CHECK:           aiex.dma_start_task([[T1]])
// CHECK:           [[T2:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           } {issue_token = true
// CHECK:           aiex.dma_start_task([[T2]])
// CHECK:           [[T3:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           } {issue_token = true
// CHECK:           aiex.dma_start_task([[T3]])

// 4 inline awaits at the original IRON dma_await_task source positions
// (end-of-rtSeq in source order).  No trailing free emissions — every
// task was released by an explicit wait_all consumer.
// CHECK:           aiex.dma_await_task([[T0]])
// CHECK-NEXT:      aiex.dma_await_task([[T1]])
// CHECK-NEXT:      aiex.dma_await_task([[T2]])
// CHECK-NEXT:      aiex.dma_await_task([[T3]])
// CHECK-NOT:       aiex.dma_free_task

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

    // Four MM2S invocations of the same channel, with IRON's explicit
    // dma_await_task per invocation.  Basic Path C preserves these as
    // wait_all{token = true} → inline aiex.dma_await_task in the lowered IR.
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
