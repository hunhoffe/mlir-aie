//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma %s | FileCheck %s

// Direction 1a regression (FS7 followup, Task #62): --conduit-to-dma Step
// 8g must emit aiex.dma_free_task / aiex.dma_await_task at the matching
// per-op release boundary, NOT batched at the end of the runtime_sequence
// body.  When a new put/get_memref on channel X arrives, the previous
// task's BD on X must be released BEFORE the new configure — otherwise
// the BD-ID allocator (--aie-assign-runtime-sequence-bd-ids) sees every
// interval span the whole rtSeq body and cannot recycle, exhausting the
// per-tile pool.
//
// This single-channel test validates the per-op release on the simplest
// shape: 4 MM2S invocations of one objectfifo.  The expected output IR
// shape is:
//   configure_1, start_1,
//   await_1, configure_2, start_2,
//   await_2, configure_3, start_3,
//   await_3, configure_4, start_4,
//   free_4   (trailing release at end-of-body)
// i.e. 4 configures, 3 in-loop awaits (await guarantees prior queued
// repeat_count fires drain before next configure overwrites the BD
// register), and 1 trailing free for the last invocation.

// CHECK-LABEL: module @step8g_single_channel
// CHECK:       aie.runtime_sequence

// Per-op interleaving asserted by the ordered CHECK / CHECK-NEXT block
// below.  Each of the 4 invocations emits a (configure, start) tuple in
// source order, with an await on the prior task before the next
// configure (channel-local).  The trailing release for the last op is a
// dma_free_task emitted at end-of-rtSeq.  First configure has no
// preceding release (no previous task on this channel).
// CHECK:           [[T0:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           aiex.dma_start_task([[T0]])
// CHECK-NEXT:      aiex.dma_await_task([[T0]])
// CHECK-NEXT:      [[T1:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           aiex.dma_start_task([[T1]])
// CHECK-NEXT:      aiex.dma_await_task([[T1]])
// CHECK-NEXT:      [[T2:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           aiex.dma_start_task([[T2]])
// CHECK-NEXT:      aiex.dma_await_task([[T2]])
// CHECK-NEXT:      [[T3:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           aiex.dma_start_task([[T3]])
// CHECK:           aiex.dma_free_task([[T3]])

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

    // Four invocations of the same channel.  IRON's pre-conduit emission
    // would interleave the awaits/frees per iteration; --dma-task-to-conduit
    // drops the original sync ops, so Step 8g must reconstruct the
    // per-op release.
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
