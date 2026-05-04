//===- passC_surfaces_dma_repeat_1_on_output_configure_task.mlir *-MLIR-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Regression test: --conduit-to-dma surfaces a properly subtract-1
// adjusted `repeat_count = 0` onto the OUTPUT (S2MM) shim
// `aiex.dma_configure_task_for` when the source `conduit.create`
// carries `dma_repeat = 1`.
//
// v5 convention (locked 2026-04-30): firmware push_queue `repeat_count`
// is 0-indexed — emit value N causes the BD to fire N+1 times.  The
// CHANNEL-side `dma_repeat = N` represents the user's intended fire
// count (N total fires, not N+1).  Pass C subtracts 1 at the
// configure_task emit site to translate user-intent → firmware-encoding.
// So `dma_repeat = 1` (user wants 1 fire) → `repeat_count = 0` on the
// configure_task → firmware fires once.  Empirical proof that the
// subtract-1 is correct: #89 captured-IR rc=4→3 patch + iron_stride_zero
// NPU smoke patched rc=4→3 → byte-equivalent / PASS, both 2026-04-30.
//
// Pipeline check:
//   1. Pass A (--objectfifo-to-conduit): SHIM channel sees emit.count = 1
//      and skips stamping `dma_repeat` per dc792ebbd5.  ✓
//   2. --dma-task-to-conduit: surfaces IRON's literal `repeat_count = 1`
//      onto the conduit.create as `dma_repeat = 1`.  ✓
//   3. Pass C (--conduit-to-dma): ConduitToDMALower.cpp v5 emit gate
//      `effectiveRepeat >= 1` surfaces `effectiveRepeat - 1` onto the
//      output configure_task.  ✓
//
// History: this test was originally the BUG isolation for the GEMM
// @ attn_query 1.95M-wrong-rows (50%-zero) numerical bug.  The
// historical "fix" surfaced repeat_count verbatim (`= 1`) — that
// over-fired by 1 in firmware; the v5 fix corrects the semantic
// translation.  Pinned originally as
// `passC_drops_dma_repeat_1_on_output_configure_task_BUG.mlir` at
// `a8c83780dd`; flipped to its prior verbatim-surface form, then
// re-pinned at v5 (subtract-1) on 2026-04-30.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: aie.device(npu2)

// The output configure_task emits effective repeat_count = 0 (v5:
// subtract-1 at emit; user's IRON `repeat_count = 1` → 1 fire on
// hardware = firmware default).  The MLIR printer elides default-zero
// integer attrs, so `repeat_count` does not appear in the textual
// output — verified absence is the correct pin.  Pre-v5 this slot
// carried `repeat_count = 1 : i32` (verbatim surface, over-fired by 1
// in firmware).
// CHECK:       aiex.dma_configure_task_for @C_shim_alloc
// CHECK:         aie.dma_bd
// CHECK:       } {issue_token = true, repeat_count = 1 : i32}

module @passC_surfaces_dma_repeat_1_on_output_configure_task {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Compute → shim output channel: tile produces, shim consumes (S2MM).
    aie.objectfifo @C(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<4096xbf16>>

    func.func private @gemm_kernel(memref<4096xbf16>)

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %sub = aie.objectfifo.acquire @C(Produce, 1)
            : !aie.objectfifosubview<memref<4096xbf16>>
        %buf = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
        func.call @gemm_kernel(%buf) : (memref<4096xbf16>) -> ()
        aie.objectfifo.release @C(Produce, 1)
      }
      aie.end
    } {link_with = "gemm.c"}

    // IRON emits an OUTPUT-side configure_task with
    //   {issue_token = true, repeat_count = 1 : i32}
    // = 2 fires per call.  --dma-task-to-conduit surfaces the
    // `repeat_count = 1` onto the conduit.create as `dma_repeat = 1`
    // (verified by the existing s2mm propagation test).  Pass C should
    // re-emit `repeat_count = 1` on the output configure_task to match
    // stateful — but currently SKIPs it (the BUG this lit pins).
    aie.runtime_sequence(%arg0: memref<8192xbf16>) {
      %t = aiex.dma_configure_task_for @C {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 0, 4096) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}
