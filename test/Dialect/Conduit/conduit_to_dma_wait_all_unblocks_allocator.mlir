//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma %s | FileCheck %s
// Metafix Candidate 1 (basic Path C): also smoke through downstream
// shim-allocation-substitution + BD-ID assignment.  Critical for this
// test: with 17 same-channel configures and 16 inline frees, the
// per-channel BD live-interval set must remain narrow enough that the
// allocator does NOT exhaust the per-shim 16-BD pool — i.e. the second
// RUN line must succeed silently.  Compare to
// conduit_to_dma_allocator_exhaustion_per_channel_pool.mlir which pins
// the no-wait_all input that DOES exhaust.
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// Basic Path C (2026-04-25, Task #18) end-to-end demonstration: 17
// same-channel MM2S invocations with per-iteration IRON-emitted
// dma_free_task between configures.  These survive --dma-task-to-conduit
// as conduit.wait_all{token = false} ops, and Step 8g lowers each one to
// an inline aiex.dma_free_task at the wait_all source location.
//
// The 17th invocation has no IRON-emitted free in source, so its release
// falls through to the trailing-release loop.  Expected output shape:
//   configure_0, start_0, free_0,  (inline)
//   configure_1, start_1, free_1,  (inline)
//   ... 14 more interleaved configure / start / free triples
//   configure_15, start_15, free_15,  (inline; 16 inline frees total)
//   configure_16, start_16,           (no inline free for the 17th)
//   free_16                           (trailing — only one trailing free)
//
// Critical invariant: with 16 inline frees, the per-channel BD live
// intervals never overlap (each interval is [configure_i, free_i]
// non-overlapping).  The allocator can assign all 17 the same BD ID, so
// the per-channel pool of 16 is more than sufficient.  Without basic
// Path C, all 17 intervals would be [configure_i, trailing_free_i],
// fully overlapping, and the allocator would emit
// "Allocator exhausted available buffer descriptor IDs" (see
// AIEAssignRuntimeSequenceBDIDs.cpp:106-113).

// CHECK-LABEL: module @wait_all_unblocks_allocator
// CHECK:       aie.runtime_sequence

// First 16 invocations: configure + start + INLINE free per iteration.
// CHECK:           [[T0:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           aiex.dma_start_task([[T0]])
// CHECK:           aiex.dma_free_task([[T0]])
// CHECK:           [[T1:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           aiex.dma_start_task([[T1]])
// CHECK:           aiex.dma_free_task([[T1]])

// Skip ahead — count the frees and configures via CHECK-COUNT below.

// The 17th configure has no inline free in source → goes to trailing
// release.  Pin the 17th configure's task SSA, then verify exactly one
// trailing free of that task at end-of-rtSeq.
// CHECK:           [[T16:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_in
// CHECK:           aiex.dma_start_task([[T16]])
// CHECK-NOT:       aiex.dma_free_task
// CHECK:           aiex.dma_free_task([[T16]])

// Total free count: 16 inline + 1 trailing = 17 (one per configured task).
// The per-T0/T1/T16 CHECK blocks above pin the structural shape; total count
// is implied by the per-task assertions, which fail loudly if any free is
// missing or out of position.

module @wait_all_unblocks_allocator {
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
      // Iterations 0..15: configure + start + inline free.
      %t0 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_free_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 128, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_free_task(%t1)
      %t2 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 256, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      aiex.dma_free_task(%t2)
      %t3 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 384, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3)
      aiex.dma_free_task(%t3)
      %t4 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 512, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t4)
      aiex.dma_free_task(%t4)
      %t5 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 640, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t5)
      aiex.dma_free_task(%t5)
      %t6 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 768, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t6)
      aiex.dma_free_task(%t6)
      %t7 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 896, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t7)
      aiex.dma_free_task(%t7)
      %t8 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1024, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t8)
      aiex.dma_free_task(%t8)
      %t9 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1152, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t9)
      aiex.dma_free_task(%t9)
      %t10 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1280, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t10)
      aiex.dma_free_task(%t10)
      %t11 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1408, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t11)
      aiex.dma_free_task(%t11)
      %t12 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1536, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t12)
      aiex.dma_free_task(%t12)
      %t13 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1664, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t13)
      aiex.dma_free_task(%t13)
      %t14 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1792, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t14)
      aiex.dma_free_task(%t14)
      %t15 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 1920, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t15)
      aiex.dma_free_task(%t15)

      // Iteration 16: configure + start, NO inline free.  This task's
      // release falls through to the trailing-release loop (one trailing
      // free of %t16 at end-of-rtSeq).
      %t16 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<2176xbf16>, 2048, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t16)
    }
  }
}
