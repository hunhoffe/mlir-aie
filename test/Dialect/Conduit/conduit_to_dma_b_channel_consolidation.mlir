//===- conduit_to_dma_b_channel_consolidation.mlir ----------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Pin (Sprint N+4, 2026-05-03 → flipped to correct-pin once canon
// refuse-to-collapse-on-link landed): canon
// (--conduit-canonicalize-channel-puts) MUST NOT collapse 8 structurally-
// identical IRON puts on a shim-MM2S → memtile LINKED channel into a
// single configure_task with `repeat_count = 7`.  Pre-fix, canon's
// HomogeneousRepeatPattern over-collapsed and Pass C surfaced a single
// `{repeat_count = 7 : i32}` configure → wedged HW on the Llama prefill
// `attn_scores` GEMM (M=2048 K=2048 N=512 num_invocations=16): B-channel
// weight MM2S path hung at first dispatch with XRT timeout.
//
// Empirical proof of root cause (2026-05-03, gemm-handpatch-verify):
//   * Captured failing conduit post-Pass-C IR had 8 consolidated
//     B-channel configures (one per shim col 0-7), each with
//     `{repeat_count = 7 : i32}` (= 8 firmware fires consolidated from
//     8 IRON puts paced 2/round × 4 rounds).
//   * Hand-patched the runtime_sequence to replace the 8×repeat=7
//     consolidation with stateful's 64×repeat=1 paced shape (per-col
//     pattern `(C, A, B, A, B)` × 4 rounds).
//   * aiecc compiled the patched IR; NPU dispatch completed in <1ms,
//     output matched numpy bf16 matmul reference (max_abs_diff =
//     0.003261, well within bf16 K=2048 accumulation tolerance).
// Canon's homogeneous-repeat collapse on a LINKED channel is the
// unambiguous root cause.
//
// Root-cause area: canon, NOT Pass C.
// `lib/Dialect/Conduit/Transforms/patterns/HomogeneousRepeatPattern.cpp::
// tryCollapsePuts` was over-eager — it folded the 8 structurally-identical
// puts into 1 put + `dma_repeat = 7`, then Pass C surfaced that verbatim
// onto `configure_task.repeat_count`.  Pass C is faithful; canon was the
// over-collapse site.  The CLAUDE.md "Active Open Bugs" Case B residual
// row in ConduitToDMALink.cpp is adjacent but DIFFERENT — that row is the
// always-circular chain shape; this case is the homogeneous-collapse
// pattern firing on a shim-MM2S → memtile **linked** path where stateful's
// correct shape is N separate paced configures.
//
// Fix shape (landed in this commit): canon's HomogeneousRepeatPattern
// (and ArithProgressionPattern, symmetric) refuses to collapse when the
// channel participates in any aie.objectfifo.link (lowered to
// conduit.scatter / gather / transpose by Pass A).  Refusal happens
// AFTER cheap structural matches BEFORE the dma_repeat / cap checks.
// New helper: `xilinx::conduit::detail::isLinkedChannel(scope, chanName)`
// in CanonicalizeChannelPutsUtils.{h,cpp}.  History: the wrong-current
// pin (count-1 directive on the consolidated configure with a
// repeat_count attr of 7) landed in acdb1d7415 and was flipped to the
// correct-pin shape below in the same commit that landed the source fix.
//
// Fixture geometry (minimum to exercise the canon refusal):
//   * 3 tiles: shim(0,0) → memtile(0,1) → compute(0,2).
//   * @B_L3L2: shim-producer → memtile-consumer, depth=2.
//   * @B_L2L1: memtile-producer → compute-consumer, depth=2.
//   * aie.objectfifo.link [@B_L3L2] -> [@B_L2L1]([] [0]) — Pass A lowers
//     this to conduit.scatter; isLinkedChannel sees @B_L3L2 in the
//     scatter's `srcs` array → returns true → canon refuses to collapse.
//   * Compute core: scf.for trip=∞ acquire/release(Consume, 1) on
//     @B_L2L1 — same shape as the GEMM kernel core.
//   * 8 IRON-emitted `aiex.dma_configure_task_for @B_L3L2` ops in the
//     runtime_sequence, all STRUCTURALLY IDENTICAL: same %arg1, offset
//     0, BD len 4096, no producer dims, no `repeat_count` attribute.
//
//===----------------------------------------------------------------------===//

// Pipeline mirrors the bare `--use-conduit` aiecc pipeline (aiecc.cpp:1492-
// 1508).  Two-line metafix convention catches dialect-verifier-only
// failures the bare FileCheck line misses (per CLAUDE.md locked design
// rule).
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-canonicalize-channel-puts --conduit-depth-promote --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-canonicalize-channel-puts --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// CHECK-LABEL: aie.device(npu2)

// CORRECT PIN (post-fix): exactly EIGHT shim configures for
// @B_L3L2_shim_alloc inside the runtime_sequence, each carrying
// `aie.dma_bd(%{{.*}}, 0, 4096)` and NO `repeat_count` attribute.
// Canon refused to collapse because @B_L3L2 participates in
// aie.objectfifo.link; Pass C therefore emits one configure per IRON
// put (byte-pacing-equivalent to stateful).
// CHECK:         aie.runtime_sequence
// CHECK-COUNT-8: aiex.dma_configure_task_for @B_L3L2_shim_alloc
// CHECK-NOT:       repeat_count
// CHECK-NOT:     aiex.dma_configure_task_for @B_L3L2_shim_alloc

// The shim_dma_allocation for the B-channel must be present and routed
// MM2S (shim is producer).  Anchored AFTER the runtime_sequence per
// the actual emit ordering.
// CHECK:       aie.shim_dma_allocation @B_L3L2_shim_alloc(%{{.*}}, MM2S, {{[0-9]+}}) {conduit_channel = @B_L3L2}

module @b_channel_consolidation {
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
    // × 4 rounds).  Post-fix, canon's HomogeneousRepeatPattern refuses
    // to collapse because @B_L3L2 participates in aie.objectfifo.link;
    // Pass C therefore emits 8 separate configures (the CORRECT shape
    // pinned by the CHECK lines above).
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
