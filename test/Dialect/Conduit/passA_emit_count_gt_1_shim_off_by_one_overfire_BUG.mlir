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
// LATENT BUG ISOLATION — Pass A `emit.count > 1` × Pass C shim surfacing
// produces an off-by-one over-fire at runtime.
//
// Status: pinned per the user-locked "isolate bugs with lit BEFORE fixing"
// convention (CLAUDE.md, 2026-04-24).  This test pins the CURRENT WRONG
// behavior; the fix is deferred (see "Forward-flip" below) and is not
// sprint-critical (Llama does not exercise the offending path — see
// CLAUDE.md "Active Open Bugs" → "Convention-divergence" row, recorded
// commit `f035bd0`).
//
// ---------------------------------------------------------------------------
// Bug summary (split convention; off-by-one in shim only)
// ---------------------------------------------------------------------------
// Pass A's `inferDmaRepeatForChannel` stamps `dma_repeat = N` on the
// `conduit.create` intending N to be the TOTAL number of BD firings per
// host dispatch — formula:
//     dma_repeat = (total_core_acquires / rt_emissions_per_channel)
//                  / acquires_per_BD
// Source:
//   mlir-aie/lib/Dialect/Conduit/Transforms/ObjectFifoToConduit.cpp:495
//     (`inferDmaRepeatForChannel`, formula at :596,
//      emit.count == 1 SHIM skip at :609-647)
//
// Pass C's MEMTILE surfacing already accounts for libxaie's "0 = once,
// N-1 convention" by stamping `DMAStartOp.repeat_count = dma_repeat - 1`.
// Source:
//   mlir-aie/lib/Dialect/Conduit/Transforms/ConduitToDMALink.cpp:1804-1810
// This is internally consistent with Pass A's TOTAL convention on the
// memtile path.
//
// Pass C's SHIM surfacing (`aiex.dma_configure_task_for.repeat_count`)
// emits the value VERBATIM — no `-1` correction.
// Source:
//   mlir-aie/lib/Dialect/Conduit/Transforms/ConduitToDMALower.cpp:1209-1248
//     (channelDmaRepeat plumbed straight into repeat_count attr at :1245-1248)
//
// However, firmware reads SHIM `repeat_count = N` as **N+1 fires** (the
// libxaie additional-firings convention also applies on the shim NPU
// command-word path).  This was proven by NPU byte-patch evidence in
// Task #73 (npu-hypothesis-verify): patching the produced
// `repeat_count: 4 → 0` shifted observed runtime behavior from "5 fires
// (over-fire)" to "1 fire", confirming the shim path consumes the field
// in firmware-additional convention.
//
// Combined effect: `emit.count > 1` shim channels over-fire by one per
// dispatch.  This is NOT exercised by Llama (the dominant `emit.count = 1`
// path is dc792ebbd5-skipped — no Pass A stamp on that path).  It bites
// the gemv-style multi-emission shape (one `rt.fill` per batch via
// Python-side `for batch in range(N)`).
//
// The IRON-explicit path (commit `7f1af434bd` dma-task-fix,
// dma-task-to-conduit) is CORRECT because IRON's user-supplied
// `repeat_count = R` already uses the firmware-additional convention, so
// verbatim copy through Pass C lands on the right firmware count.  Only
// Pass A's *inferred* dma_repeat (TOTAL convention) collides with the
// shim's verbatim/additional surfacing.
//
// Documentation drift: Conduit.td:462-463 states
//   "DMAStartOp.repeat_count = dma_repeat - 1"
// — accurate for the memtile path; misleading for the shim path under
// the current code.
//
// ---------------------------------------------------------------------------
// Chosen fix convention (B) additional-firings — landed as separate task
// ---------------------------------------------------------------------------
// `dma_repeat = N` will mean "fire N+1 times" (firmware/libxaie native).
// Under that convention the fix is:
//   * Pass A `inferDmaRepeatForChannel`: subtract 1 from the computed
//     total before stamping; skip stamp when the result would be 0.
//   * Pass C shim surfacing (`ConduitToDMALower.cpp`): keep verbatim
//     (already correct under chosen convention).
//   * Pass C memtile surfacing (`ConduitToDMALink.cpp:1810`): drop the
//     `- 1` (already in additional-firings convention from Pass A).
//   * Conduit.td:462-463 docstring: rewrite to describe additional-
//     firings convention.
//
// ---------------------------------------------------------------------------
// Geometry (smallest reproducer derived from
// passC_shim_bd_dma_repeat_uses_configure_task_repeat.mlir)
// ---------------------------------------------------------------------------
//   * fifo elem  = memref<4xbf16>
//   * core loop  = scf.for 0..8 step 1; one acquire/release per iter
//                  → total_core_acquires = 8
//   * emit.count = 2 (two identical aiex.dma_configure_task_for ops on
//                  the same channel — one per "batch" emission)
//   * BD len     = 4 elements; per_BD = 4 / 4 = 1
//   * Pass A inferred dma_repeat = (8 / 2) / 1 = 4  (TOTAL convention)
//   * Pass C shim surface          repeat_count    = 4  (verbatim)
//   * Firmware fires                              = 5  (additional → over-fire)
//
// ---------------------------------------------------------------------------
// Forward-flip (when the chosen-convention fix lands)
// ---------------------------------------------------------------------------
//   * This test's CHECKs become `dma_repeat = 3` and `repeat_count = 3`
//     (= "fire 4 times per dispatch" under additional convention,
//     matching the core's per-dispatch acquire count of 4).
//   * Drop the `_BUG` suffix from the filename.
//   * The two existing tests must flip together in the same fix commit:
//       - test/Dialect/Conduit/passC_shim_bd_dma_repeat_uses_configure_task_repeat.mlir
//         CHECKs `dma_repeat = 4` / `repeat_count = 4` → flip to 3 / 3.
//       - test/Dialect/Conduit/infer_iter_count_multi_emission_gemv_pattern.mlir
//         CHECK `dma_repeat = 2` → flip to `dma_repeat = 1` (= "fire 2 times").
//
//===----------------------------------------------------------------------===//

// Pass A only: pin Pass A's `dma_repeat = 4` stamp on the conduit.create
// (TOTAL convention; intended meaning: "fire 4 times per dispatch").
// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s --check-prefix=PASSA
// PASSA-LABEL: aie.device(npu1)
// PASSA:      conduit.create @chan
// PASSA-SAME: dma_repeat = 4

// Full pipeline: pin Pass C shim surfacing — repeat_count = 4 emitted
// VERBATIM into aiex.dma_configure_task_for.  This is the bug: firmware
// reads N as N+1 fires, so the BD will fire 5 times per dispatch.
// (Under the chosen-convention fix, this CHECK becomes `repeat_count = 3`,
// meaning "fire 4 times" per firmware additional convention.)
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s --check-prefix=PASSC
// PASSC-LABEL: aie.device(npu1)
// PASSC:      aiex.dma_configure_task_for @chan_shim_alloc
// PASSC:        aie.dma_bd
// PASSC:      {{.*}}repeat_count = 4
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
