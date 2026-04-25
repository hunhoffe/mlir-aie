//===- roundtrip_identity_no_inline_free_with_dma_repeat.mlir -*- MLIR -*-===//
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
// overwrites pending BD") under the RUNTIME-TRIGGER condition.  See
// `roundtrip_identity_no_inline_free_mm2s_invariant.mlir` for the full
// rationale on bug class 3 and Option A's structural fix.
//
// Why this file exists alongside the MM2S form: bug class 3 only
// MANIFESTS at runtime when `dma_repeat > 0` on the channel — that is
// when the prior start has queued fires that have not yet drained when
// the synchronous free executes.  The structural lit invariant pinned
// in the MM2S file holds independent of `dma_repeat` (it is an IR-level
// property, not a runtime property).  This file pins the COMBINED
// invariant: with `dma_repeat = N > 0` AND multi-invocation on the same
// channel, the inline release is `dma_await_task` AND the surfaced
// `repeat_count = N` is preserved on each output configure_task.
//
// Why the combination matters as a permanent guard: a future change
// might preserve the await-not-free invariant under the simple shape
// (no dma_repeat) but accidentally break it under the dma_repeat path
// (e.g., a release-codepath fork keyed on dma_repeat).  This test
// catches that regression class.
//
// Pattern: 2 same-channel MM2S invocations with `repeat_count = 1 : i32`
// on each input configure_task (= 2 fires per call per IRON convention,
// surfacing as `dma_repeat = 1` on the conduit.create per
// `dma_task_to_conduit_propagates_repeat_count.mlir`).
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence

// First configure: surfaces `repeat_count = 1` on the output (Pass C
// `> 0` guard from commit 928e29adfe).  No preceding release.
// CHECK:           [[T0:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ch
// CHECK:           aie.dma_bd
// CHECK:           } {repeat_count = 1 : i32}
// CHECK:           aiex.dma_start_task([[T0]])

// Inline release: must be `dma_await_task`, NOT `dma_free_task`.  This
// is the bug-class-3 invariant under the runtime-trigger condition.
// CHECK-NEXT:      aiex.dma_await_task([[T0]])

// Second configure: also surfaces `repeat_count = 1` (channel-level
// dma_repeat surfaces verbatim per invocation).
// CHECK-NEXT:      [[T1:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ch
// CHECK:           aie.dma_bd
// CHECK:           } {repeat_count = 1 : i32}
// CHECK:           aiex.dma_start_task([[T1]])

// Trailing release: the only `dma_free_task` for this MM2S-only pattern.
// CHECK:           aiex.dma_free_task([[T1]])

// The bug-class-3 NEGATIVE invariant ("no inline MM2S free between
// same-channel configures, even with dma_repeat > 0") is pinned
// structurally by the next-line await assertion on `[[T0]]` above —
// a future regression that re-introduced an inline `dma_free_task`
// between configures (whether or not gated on the dma_repeat path)
// would fail that next-line assertion.

module @roundtrip_identity_no_inline_free_with_dma_repeat {
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

    aie.runtime_sequence(%arg0: memref<256xbf16>) {
      %t0 = aiex.dma_configure_task_for @ch {
        aie.dma_bd(%arg0 : memref<256xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ch {
        aie.dma_bd(%arg0 : memref<256xbf16>, 128, 128) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t0)
      aiex.dma_await_task(%t1)
    }
  }
}
