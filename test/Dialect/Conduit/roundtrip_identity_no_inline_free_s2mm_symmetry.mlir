//===- roundtrip_identity_no_inline_free_s2mm_symmetry.mlir ---*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Roundtrip-identity invariant — S2MM symmetry mirror of
// `roundtrip_identity_no_inline_free_mm2s_invariant.mlir`.  See that file
// for the full rationale on bug class 3 ("Eager dma_free_task overwrites
// pending BD") and Option A's structural fix.
//
// Why this file exists alongside the MM2S form: pre-Option-A the inline
// release for S2MM was already `dma_await_task` (because S2MM with
// `issue_token=true` requires await for completion signaling).  So bug
// class 3 never manifested on S2MM — but a FUTURE regression that
// "unifies" the two release codepaths in the wrong direction (e.g.,
// "S2MM should use free for symmetry with MM2S" — exactly the inverse
// of Option A) would silently re-introduce the bug class on S2MM.
// This test pins the await-not-free invariant on S2MM permanently.
//
// Pattern: 3 same-channel S2MM (compute → shim) invocations of one
// objectfifo with `issue_token = true`.  After Pass C the only
// `dma_free_task` in the runtime_sequence body should be... none.
// (S2MM trailing release is `dma_await_task`, not `dma_free_task`.)
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence

// Step 1: in-loop release between configure_1 and configure_2 is await.
// CHECK:           [[T0:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ch_out
// CHECK:           aiex.dma_start_task([[T0]])
// CHECK-NEXT:      aiex.dma_await_task([[T0]])
// CHECK-NEXT:      [[T1:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ch_out

// Step 2: same for configure_2 → configure_3.
// CHECK:           aiex.dma_start_task([[T1]])
// CHECK-NEXT:      aiex.dma_await_task([[T1]])
// CHECK-NEXT:      [[T2:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ch_out

// Step 3: trailing release for the last invocation is also `dma_await_task`
// (S2MM with issue_token=true → completion-signaling await).
// CHECK:           aiex.dma_start_task([[T2]])
// CHECK:           aiex.dma_await_task([[T2]])

// Step 4 (NEGATIVE invariant pin): ZERO `aiex.dma_free_task` ops anywhere
// in the runtime_sequence for this S2MM-only pattern.  Any future change
// that introduces an inline OR trailing free on S2MM trips this guard.
// CHECK-NOT:       aiex.dma_free_task

module @roundtrip_identity_no_inline_free_s2mm_symmetry {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    aie.objectfifo @ch_out(%tile, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @kernel(memref<128xbf16>)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %out = aie.objectfifo.acquire @ch_out(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @kernel(%out_buf) : (memref<128xbf16>) -> ()
        aie.objectfifo.release @ch_out(Produce, 1)
      }
      aie.end
    } {link_with = "kernel.a"}

    aie.runtime_sequence(%arg0: memref<384xbf16>) {
      %t0 = aiex.dma_configure_task_for @ch_out {
        aie.dma_bd(%arg0 : memref<384xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ch_out {
        aie.dma_bd(%arg0 : memref<384xbf16>, 128, 128) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @ch_out {
        aie.dma_bd(%arg0 : memref<384xbf16>, 256, 128) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t0)
      aiex.dma_await_task(%t1)
      aiex.dma_await_task(%t2)
    }
  }
}
