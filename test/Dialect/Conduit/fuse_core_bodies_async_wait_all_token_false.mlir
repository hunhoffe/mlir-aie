//===- fuse_core_bodies_async_wait_all_token_false.mlir ----*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Task #75 — Dialect-level regression pin for the
// `--conduit-fuse-core-bodies` Step 0 (`mergeAndUnifyDevices`)
// async-token-aware erase fix (Task #74).
//
// Bug shape (pre-fix): when convergent fusion collapses a paired
// fusion_group consumer channel and devB's runtime_sequence carries a
// `conduit.put_memref_async` / `get_memref_async` referencing that
// consumer channel, the Step 0 erase loop raw-erased the async op while
// a downstream `conduit.wait_all{token=false}` (the `dma_free_task`
// release marker) still held its result token as an SSA operand →
// `LLVM ERROR: operation destroyed but still has uses`.
//
// Fix: split the consumer-channel rt-seq op collection — async ops route
// through `safeEraseAsyncOp` (token-aware: prunes dead operands from
// downstream wait_all / wait_all_async, erasing the wait if its operand
// list becomes empty); blocking variants (`put_memref` / `get_memref`)
// keep raw-erase. Mirrors the `cleanUpDeadOps` pattern already used
// post-fusion at line 1217+.
//
// This Dialect-level pin complements the HW smoke at
// `test/npu-xrt/fuse_hybrid_swiglu_npu/` by making the regression
// catchable WITHOUT HW dispatch — fast lit-only signal so future
// refactors that re-introduce raw-erase fail at `ninja check` time
// rather than first-NPU-contact time.
//
// Sibling pattern reference:
//   `path_c_async_fuse_corebody_blocks_at_wait_all.mlir` — covers the
//   simpler single-device case (no `mergeAndUnifyDevices` invocation;
//   pinned the rt-seq passthrough invariant).  This fixture extends to
//   the cross-device case where Step 0 actually erases the async op.
//   `fuse_operators_convergent_basic.mlir` — convergent fusion_group
//   shape borrowed for the producer-pair structure.
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit \
// RUN:   --conduit-fuse-core-bodies %s | FileCheck %s

// CHECK-LABEL: module @fuse_core_bodies_async_wait_all_token_false

// Step 0 collapses the consumer-side `@inter_b` (devB) into the producer-
// side `@inter_a` (devA) via the matched fusion_group pair.  The
// consumer-side conduit.create must NOT survive Step 0.
// CHECK-NOT:   conduit.create @inter_b

// The producer-side @inter_a survives Step 0 (it is the rename target,
// not a deletion target).  Step 1+ findFusableCorePairs is intentionally
// not exercised by this fixture — the two cores both PRODUCE to their
// inter channel (post-rename: both produce to @inter_a) so no
// producer→consumer core pair forms and the @inter_a Create persists.
// This narrows the regression target strictly to Step 0's safe-erase.
// CHECK-DAG:   conduit.create @inter_a

// External I/O channels survive (only the fusion_group-matched pair is
// touched by Step 0).
// CHECK-DAG:   conduit.create @ext_in_a
// CHECK-DAG:   conduit.create @ext_in_b

// Critical regression invariant: NO put_memref_async / get_memref_async
// referencing the collapsed consumer channel @inter_b survives Step 0.
// (Pre-fix: the walk added the async op to a raw-erase list, then a
// downstream wait_all{token=false} still referenced its result → crash
// before this CHECK could even run.  Post-fix: safeEraseAsyncOp removes
// the async op AFTER pruning the wait_all operand.)
// CHECK-NOT:   {{(put|get)}}_memref_async {{.*}}name = @inter_b

// Critical regression invariant: the wait_all{token=false} that consumed
// the now-dead async-op's token must have been rebuilt OR erased.  In
// this fixture devB's rt-seq has exactly ONE `dma_free_task` on
// @inter_b, whose release-marker semantic lowers to a single-operand
// wait_all{token=false} on @inter_b's get_memref_async token.  After
// safeEraseAsyncOp the operand list becomes empty → removeTokenFromWaitOp
// erases the wait_all entirely.  Other wait_all{token=false} ops (for
// @ext_in_a / @ext_in_b dma_free_tasks) survive untouched — so the
// stricter pin is the operand-name CHECK-NOT above.  As a positive pin,
// the surviving rt-seq still emits at least one wait_all.
// CHECK:       conduit.wait_all

module @fuse_core_bodies_async_wait_all_token_false {

  // Producer device: pre-fuse @inter_a is the producer-side intermediate
  // tagged with fusion_group "fg0" (fusion_index = 0).
  aie.device(npu2) @devA {
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)

    aie.objectfifo @ext_in_a(%shim_a, {%tile_a}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    aie.objectfifo @inter_a(%tile_a, {%shim_a}, 2 : i32)
        {fusion_group = "fg0", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @kernel_a(memref<128xbf16>, memref<128xbf16>)

    %core_a = aie.core(%tile_a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cmax = arith.constant 9223372036854775807 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_a(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_a(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @kernel_a(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @inter_a(Produce, 1)
        aie.objectfifo.release @ext_in_a(Consume, 1)
      }
      aie.end
    } {link_with = "kernel_a.o"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_a {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_free_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_a {
        aie.dma_bd(%arg1 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
    }
  }

  // Consumer device: pre-fuse @inter_b is the consumer-side intermediate
  // tagged with the same fusion_group "fg0" (fusion_index = 1) so
  // mergeAndUnifyDevices selects it as the consumer channel to collapse.
  //
  // The runtime_sequence's `dma_configure_task_for @inter_b` + `dma_start`
  // + `dma_free_task` lowers via --dma-task-to-conduit to:
  //     %t = conduit.get_memref_async {name = @inter_b, ...} -> token
  //     conduit.wait_all %t {token = false}
  // i.e. the EXACT shape that pre-fix raw-erase blew up on.
  aie.device(npu2) @devB {
    %shim_b = aie.tile(0, 0)
    %tile_b = aie.tile(0, 2)

    aie.objectfifo @ext_in_b(%shim_b, {%tile_b}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    aie.objectfifo @inter_b(%tile_b, {%shim_b}, 2 : i32)
        {fusion_group = "fg0", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @kernel_b(memref<128xbf16>, memref<128xbf16>)

    %core_b = aie.core(%tile_b) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cmax = arith.constant 9223372036854775807 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_b(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_b(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @kernel_b(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @inter_b(Produce, 1)
        aie.objectfifo.release @ext_in_b(Consume, 1)
      }
      aie.end
    } {link_with = "kernel_b.o"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_b {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_free_task(%t0)
      // THIS is the bug-trigger: dma_configure_task_for on the
      // consumer-side fusion_group channel @inter_b, with a dma_free_task
      // (which lowers to wait_all{token=false} on the async op's token).
      // mergeAndUnifyDevices Step 0 collapses @inter_b → @inter_a; the
      // pre-fix raw-erase of the get_memref_async left the wait_all with
      // a dangling SSA operand → crash.  Post-fix safeEraseAsyncOp prunes
      // the wait_all operand list before erasing the async op.
      %t1 = aiex.dma_configure_task_for @inter_b {
        aie.dma_bd(%arg1 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_free_task(%t1)
    }
  }
}
