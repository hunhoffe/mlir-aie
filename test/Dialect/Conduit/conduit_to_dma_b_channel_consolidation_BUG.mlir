//===- conduit_to_dma_b_channel_consolidation_BUG.mlir ------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// BUG pin (Sprint N+4, 2026-05-03): --conduit-to-dma over-collapses
// structurally-identical IRON puts on a shim-MM2S → memtile linked
// channel into a SINGLE output configure_task with `repeat_count = N-1`
// (= N firmware fires).  This wedges hardware on the Llama prefill
// `attn_scores` GEMM (M=2048 K=2048 N=512 num_invocations=16) — the
// B-channel weight MM2S path hangs at the first dispatch with XRT
// timeout.
//
// Empirical proof of root cause (2026-05-03, gemm-handpatch-verify):
//   * Captured failing conduit post-Pass-C IR has 8 such consolidated
//     B-channel configures (one per shim col 0-7), each with
//     `{repeat_count = 7 : i32}` (= 8 firmware fires consolidated from
//     8 IRON puts paced 2/round × 4 rounds).
//   * Hand-patched the runtime_sequence to replace the 8×repeat=7
//     consolidation with stateful's 64×repeat=1 paced shape (per-col
//     pattern `(C, A, B, A, B)` × 4 rounds).
//   * aiecc compiled the patched IR; NPU dispatch completed in <1ms,
//     output matched numpy bf16 matmul reference (max_abs_diff =
//     0.003261, well within bf16 K=2048 accumulation tolerance).
// Consolidation is unambiguously the cause.
//
// Root-cause area (per CLAUDE.md "Active Open Bugs" → row
// "ConduitToDMALink.cpp Case B / join MM2S residual"):
//   "the chain shape is still always-circular (% caseBEffectiveBDs)
//   and inflates by nConsumerBuffers() * bd_repeat instead of
//   collapsing via dma_repeat".  For shim-MM2S → memtile, the
//   nConsumerBuffers()=putCount override fires (Common.h:315-326), the
//   8 puts collapse into a single configure with repeat_count=7, and
//   firmware front-loads all 8 transfers at start of round 1 instead
//   of pacing them across 4 rounds interleaved with sibling A and C
//   channels.
//
// Fixture geometry (minimum to trigger the consolidation):
//   * 3 tiles: shim(0,0) → memtile(0,1) → compute(0,2).
//   * @B_L3L2: shim-producer → memtile-consumer, depth=2.
//   * @B_L2L1: memtile-producer → compute-consumer, depth=2.
//   * aie.objectfifo.link [@B_L3L2] -> [@B_L2L1]([] [0]) — the LINK is
//     what makes Pass C take Case B (shim-MM2S → memtile route through
//     a memtile relay) instead of the direct-shim-MM2S → compute path.
//   * Compute core: scf.for trip=∞ acquire/release(Consume, 1) on
//     @B_L2L1 — same shape as the GEMM kernel core.
//   * 8 IRON-emitted `aiex.dma_configure_task_for @B_L3L2` ops in the
//     runtime_sequence, all STRUCTURALLY IDENTICAL: same %arg1, offset
//     0, BD len 4096, no producer dims, no `repeat_count` attribute.
//     This matches the GEMM B_L3L2_0 shape (lines 1460/1470/1620/
//     1630/1844/1854/2036/2046 of the IRON-emitted pre-Pass-C IR;
//     2 puts/round × 4 rounds = 8 total per col).
//
// CHECK lines below pin WRONG-CURRENT behavior (per CLAUDE.md
// USER-LOCKED 2026-04-24 "Isolate bugs with a minimal lit test BEFORE
// fixing"): exactly ONE shim-side B configure is emitted, carrying
// `repeat_count = 7 : i32`.  Post-fix, the CHECK lines flip to pin the
// CORRECT shape (8 separate configures with no/elided repeat_count, BD
// len 4096 each, byte-identical-pacing to stateful) and the file is
// renamed by dropping the `_BUG` suffix.
//
// Fix design target (per CLAUDE.md USER-LOCKED 2026-04-30 "Capture
// stateful's actual emit BEFORE designing any Pass C emit-rule
// change"): emit one configure per IRON put (the pre-collapse shape),
// preserving the round-pacing the host runtime sequence expressed.
// Stateful's reference shape for the same input was captured at
// `/tmp/stateful-gemm-capture/input.mlir.prj/input_with_addresses.mlir`.
//
//===----------------------------------------------------------------------===//

// Pipeline mirrors the bare `--use-conduit` aiecc pipeline (aiecc.cpp:1492-
// 1508): objectfifo-to-conduit → dma-task-to-conduit → canon-channel-puts
// → depth-promote → conduit-to-dma.  Canon-puts is the ACTUAL collapsing
// pass; without it Pass C alone leaves the 8 puts as 8 separate configures.
// Two-line metafix convention catches dialect-verifier-only failures the
// bare FileCheck line misses (per CLAUDE.md locked design rule).
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-canonicalize-channel-puts --conduit-depth-promote --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-canonicalize-channel-puts --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// CHECK-LABEL: aie.device(npu2)

// BUG PIN: exactly ONE shim configure for @B_L3L2_shim_alloc inside the
// runtime_sequence, carrying `repeat_count = 7 : i32` (= 8 firmware
// fires consolidated from the 8 IRON puts).  The `dma_bd` line and the
// `repeat_count = 7` attribute live on the same op as adjacent CHECK
// lines.  Post-fix: the count-1 directive flips to count-8, the
// `repeat_count` CHECK is dropped, and the file is renamed (drop the
// `_BUG` suffix).
// CHECK:         aie.runtime_sequence
// CHECK-COUNT-1: aiex.dma_configure_task_for @B_L3L2_shim_alloc
// CHECK:           aie.dma_bd(%{{.*}}, 0, 4096)
// CHECK:         } {repeat_count = 7 : i32}
// CHECK-NOT:     aiex.dma_configure_task_for @B_L3L2_shim_alloc

// The shim_dma_allocation for the B-channel must be present and routed
// MM2S (shim is producer).  Anchored AFTER the runtime_sequence per
// the actual emit ordering.
// CHECK:       aie.shim_dma_allocation @B_L3L2_shim_alloc(%{{.*}}, MM2S, {{[0-9]+}}) {conduit_channel = @B_L3L2}

module @b_channel_consolidation_BUG {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %mem  = aie.tile(0, 1)
    %comp = aie.tile(0, 2)

    // Shim → memtile (depth=2).  Linked downstream to memtile → compute.
    aie.objectfifo @B_L3L2(%shim, {%mem}, 2 : i32)
        : !aie.objectfifo<memref<4096xbf16>>
    aie.objectfifo @B_L2L1(%mem, {%comp}, 2 : i32)
        : !aie.objectfifo<memref<4096xbf16>>
    aie.objectfifo.link [@B_L3L2] -> [@B_L2L1]([] [0])

    func.func private @kernel(memref<4096xbf16>)

    aie.core(%comp) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %sub = aie.objectfifo.acquire @B_L2L1(Consume, 1)
            : !aie.objectfifosubview<memref<4096xbf16>>
        %buf = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
        func.call @kernel(%buf) : (memref<4096xbf16>) -> ()
        aie.objectfifo.release @B_L2L1(Consume, 1)
      }
      aie.end
    } {link_with = "kernel.a"}

    // 8 IRON-emitted MM2S puts on @B_L3L2.  All STRUCTURALLY IDENTICAL
    // (offset = 0, BD len = 4096, no producer dims, no repeat_count
    // attr) — matches the GEMM @B_L3L2_0 per-col shape (2 puts/round
    // × 4 rounds).  Pass C currently consolidates these into a single
    // configure with repeat_count = 7 (the BUG pinned by the CHECK
    // lines above).
    aie.runtime_sequence(%arg0: memref<32768xbf16>) {
      %t0 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg0 : memref<32768xbf16>, 0, 4096) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg0 : memref<32768xbf16>, 0, 4096) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg0 : memref<32768xbf16>, 0, 4096) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg0 : memref<32768xbf16>, 0, 4096) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3)
      %t4 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg0 : memref<32768xbf16>, 0, 4096) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t4)
      %t5 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg0 : memref<32768xbf16>, 0, 4096) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t5)
      %t6 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg0 : memref<32768xbf16>, 0, 4096) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t6)
      %t7 = aiex.dma_configure_task_for @B_L3L2 {
        aie.dma_bd(%arg0 : memref<32768xbf16>, 0, 4096) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t7)
      // Per-put dma_free_task ops — `--dma-task-to-conduit` lowers each
      // into `conduit.wait_all{token=false}` consuming the put's token.
      // Canon-puts requires each put to have its own sync chain (see
      // `collectSyncChain` in HomogeneousRepeatPattern.cpp:88-93).  IRON's
      // GEMM B-channel emits exactly this shape (free, no await, since
      // MM2S issues no token).
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
      aiex.dma_free_task(%t2)
      aiex.dma_free_task(%t3)
      aiex.dma_free_task(%t4)
      aiex.dma_free_task(%t5)
      aiex.dma_free_task(%t6)
      aiex.dma_free_task(%t7)
    }
  }
}
