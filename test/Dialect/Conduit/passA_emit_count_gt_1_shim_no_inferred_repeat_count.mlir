//===- passA_emit_count_gt_1_shim_off_by_one_overfire_BUG.mlir -*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// HISTORY — this fixture originally pinned the LATENT over-fire bug (Pass A
// `emit.count > 1` × Pass C shim surfacing producing an off-by-one over-fire
// at runtime).  Its CHECKs expected `dma_repeat = 4` on the conduit.create
// and `repeat_count = 4` on the Pass C shim configure_task.  The fixture's
// own "Forward-flip" block predicted those CHECKs would flip to 3/3 once
// the bug was fixed via a convention swap (additional-firings).
//
// Task #40 / Task #42 (2026-04-28) rooted the bug differently.  The actual
// fix is NOT a convention swap — it's REMOVING the inference path for
// shim-bearing channels entirely.  The host-side `num_invocations` is
// invisible to the IR; Pass A's three-factor formula under-divides by it
// and inflates `dma_repeat` by exactly that factor (Llama op7_GEMV /
// op11_GEMV emitted `repeat_count = 16` where stateful emits 0 — see
// `.claude/plans/repeat-count-overfire-rootcause.md`).  Post-fix, Pass A
// no longer infers `dma_repeat` for ANY shim-bearing channel
// (`emit.count >= 1`) — the value defers to the runtime / IRON-explicit
// stamp.
//
// POST-FIX SEMANTICS pinned by this fixture:
//   * Pass A: NO `dma_repeat` on `conduit.create @chan` (was: `dma_repeat = 4`).
//   * Pass C shim: NO `repeat_count` on the per-emission configure_task
//     (was: `repeat_count = 4` verbatim → firmware over-fire).
//   * Each per-batch `aiex.dma_configure_task_for` fires exactly once per
//     host dispatch; the host's `num_invocations` `run()` loop drives the
//     replay (matches stateful's behaviour byte-for-byte).
//
// IRON-explicit path is unchanged (still correct):
// `passC_shim_bd_dma_repeat_uses_configure_task_repeat.mlir` exercises
// the IRON-explicit `repeat_count = 4` source, which Pass C surfaces
// verbatim post-fix.
//
// ---------------------------------------------------------------------------
// Geometry (smallest reproducer)
// ---------------------------------------------------------------------------
//   * fifo elem  = memref<4xbf16>
//   * core loop  = scf.for 0..8 step 1; one acquire/release per iter
//                  → total_core_acquires = 8
//   * emit.count = 2 (two identical aiex.dma_configure_task_for ops on
//                  the same channel — one per "batch" emission)
//   * BD len     = 4 elements; per_BD = 4 / 4 = 1
//   * (Pre-fix, Pass A would have inferred dma_repeat = (8/2)/1 = 4.)
//   * Post-fix: shim-bearing channel ⇒ skip ⇒ no inferred dma_repeat,
//     no surfaced repeat_count.
//
//===----------------------------------------------------------------------===//

// Pass A only: NO `dma_repeat` attribute on the conduit.create
// (shim-bearing channel ⇒ inference skipped post Task #42).
// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s --check-prefix=PASSA
// PASSA-LABEL: aie.device(npu1)
// PASSA:      conduit.create @chan
// PASSA-NOT:  dma_repeat

// Full pipeline: NO `repeat_count` surfaces onto the shim configure_task
// (matches stateful behaviour for the same input — the per-batch
// emissions each fire exactly once per host dispatch).
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s --check-prefix=PASSC
// PASSC-LABEL: aie.device(npu1)
// PASSC:      aiex.dma_configure_task_for @chan_shim_alloc
// PASSC:        aie.dma_bd
// PASSC-NOT:    repeat_count
// PASSC:      aiex.dma_start_task

module @passA_emit_count_gt_1_shim_off_by_one_overfire_BUG {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<4xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      // 8 acquires total; emit.count = 2 → dma_repeat = (8 / 2) / 1 = 4.
      scf.for %i = %c0 to %c8 step %c1 {
        %sub = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<4xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<4xbf16>> -> memref<4xbf16>
        aie.objectfifo.release @chan (Consume, 1)
      }
      aie.end
    }

    // Two identical emissions on the same channel — one per "batch" —
    // so emit.count = 2 and Pass A's inference fires (post-#74 the
    // emit.count == 1 shim path is SKIPped).  Both Pass C output
    // configure_tasks carry repeat_count = 4 (the CHECK matches the
    // first one).
    aie.runtime_sequence(%a0: memref<4xbf16>) {
      %t0 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%a0 : memref<4xbf16>, 0, 4,
            [<size = 1, stride = 0>,
             <size = 1, stride = 0>,
             <size = 1, stride = 0>,
             <size = 4, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_free_task(%t0)
      %t1 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%a0 : memref<4xbf16>, 0, 4,
            [<size = 1, stride = 0>,
             <size = 1, stride = 0>,
             <size = 1, stride = 0>,
             <size = 4, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t1)
    }
  }
}
