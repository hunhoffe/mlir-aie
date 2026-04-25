//===- roundtrip_identity_no_inline_free_mm2s_invariant.mlir --*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Roundtrip-identity invariant — bug class 3 ("Eager dma_free_task
// overwrites pending BD"), CLAUDE.md "Active Open Bugs":
//
//   Pass C MUST NOT emit `aiex.dma_free_task` between two
//   `aiex.dma_configure_task_for` ops on the same channel without an
//   intervening `aiex.dma_await_task`.
//
// Why the invariant matters: a synchronous BD-slot release (free) does not
// bound the prior start's queued `repeat_count > 0` fires.  Pre-Option-A
// the next configure could overwrite the still-firing shim BD register,
// manifesting as an N-stripe rollover (GEMM @ attn_query 87% mismatch
// vs stateful, stripe 0 50% wrong, stripes 1..3 99.84% each).  Option A
// (commit landing this file's first-green state) swaps the inline MM2S
// release from `dma_free_task` to `dma_await_task`.  BD-pool recycling
// is preserved because AIEAssignRuntimeSequenceBDIDs synthesizes a
// `dma_free_task` after each `dma_await_task` before its interval-
// collection walk.
//
// Why a NEGATIVE-form (CHECK-NOT) test in addition to the shape-form
// step8g tests: the shape-form tests pin the EXPECTED IR for one
// specific input.  This NEGATIVE-form pins the structural property
// directly — any future change that re-introduces an inline MM2S free
// (regardless of input shape, fusion mode, or pass ordering) trips
// this test.  Strongest available regression guard against bug class 3.
//
// Pattern: 3 same-channel MM2S invocations of one objectfifo.  After
// Pass C the only `dma_free_task` in the runtime_sequence body is the
// trailing release for the last invocation; the in-loop releases are
// `dma_await_task`.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence

// Step 1 (positive shape pin): the in-loop release between configure_1
// and configure_2 is `dma_await_task`, NOT `dma_free_task`.
// CHECK:           [[T0:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ch
// CHECK:           aiex.dma_start_task([[T0]])
// CHECK-NEXT:      aiex.dma_await_task([[T0]])
// CHECK-NEXT:      [[T1:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ch

// Step 2 (positive shape pin): same for configure_2 → configure_3.
// CHECK:           aiex.dma_start_task([[T1]])
// CHECK-NEXT:      aiex.dma_await_task([[T1]])
// CHECK-NEXT:      [[T2:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ch

// Step 3 (trailing release): the LAST invocation's release is the only
// `dma_free_task` for @ch — emitted at end-of-rtSeq, no subsequent
// configure on @ch can overwrite the BD register.
// CHECK:           aiex.dma_start_task([[T2]])
// CHECK:           aiex.dma_free_task([[T2]])

// The bug-class-3 NEGATIVE invariant ("no inline MM2S free between
// same-channel configures") is pinned structurally by the
// next-line await assertions in Steps 1 and 2 above.  Any future
// regression that re-introduces an inline `dma_free_task` between
// configures would fail those next-line assertions because the line
// immediately after `dma_start_task` would no longer match
// `dma_await_task`.

module @roundtrip_identity_no_inline_free_mm2s_invariant {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    aie.objectfifo @ch(%shim, {%tile}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @kernel(memref<128xbf16>)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ch(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @kernel(%in_buf) : (memref<128xbf16>) -> ()
        aie.objectfifo.release @ch(Consume, 1)
      }
      aie.end
    } {link_with = "kernel.a"}

    aie.runtime_sequence(%arg0: memref<384xbf16>) {
      %t0 = aiex.dma_configure_task_for @ch {
        aie.dma_bd(%arg0 : memref<384xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ch {
        aie.dma_bd(%arg0 : memref<384xbf16>, 128, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @ch {
        aie.dma_bd(%arg0 : memref<384xbf16>, 256, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t0)
      aiex.dma_await_task(%t1)
      aiex.dma_await_task(%t2)
    }
  }
}
