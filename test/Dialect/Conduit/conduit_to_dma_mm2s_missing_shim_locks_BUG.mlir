//===- conduit_to_dma_mm2s_missing_shim_locks_BUG.mlir --------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// BUG ISOLATION — Pass C drops shim-side locks for MM2S input channels that
// are distribute-link sources (shim → memtile → multiple compute consumers).
//
// Status: pinned per the user-locked "isolate bugs with lit BEFORE fixing"
// convention (CLAUDE.md, 2026-04-24). This test pins the CURRENT WRONG
// behavior (CHECK-NOT lines on lines 99-104). Forward-flip to positive
// CHECKs once the source fix lands.
//
// ---------------------------------------------------------------------------
// Bug summary
// ---------------------------------------------------------------------------
// For shim-producer conduits that ALSO appear in `state.linkSrcNamesEarly`
// (i.e. are the source of a distribute objectfifo.link), Pass C's Phase 4a
// (`ConduitToDMARoute.cpp:408-409`) takes the early-skip branch:
//
//   bool isDistributeLinkSrc = state.linkSrcNamesEarly.count(name) > 0;
//   if (isAIE2 && !info.noLocks && !isDistributeLinkSrc) {
//     // ... allocate <name>_prod_lock_0 + <name>_cons_lock_0 on shim tile
//   }
//
// The skip was introduced by commit `a50b27ced7`
// ("fix(Pass C): allocate shim->MemTile relay locks on MemTile not shim",
// 2026-03-24) on the assumption that shim DMA for distribute sources is
// "fire-and-forget" (host runtime via `aiex.npu.dma_memcpy_nd` manages all
// synchronization, so shim-side locks are dead resources).
//
// That assumption is contradicted by:
//   (a) byte-diff evidence (Task #94, 2026-04-24): conduit's Llama prefill
//       artifact emits 328 aie.lock ops vs stateful's 352 — exactly 24
//       missing locks ≡ 12 distribute-link-source MM2S input channels
//       (4 A_L3L2_ + 8 B_L3L2_) × 2 shim-side locks each;
//   (b) failure mode: K=2048 GEMM (`repeat_count = 3` → 4 fires per
//       dispatch) → 50%-zero output rows; K=64 (smaller window) → ~6%
//       wrong rows; Llama prefill TIMEOUTs at attn_key once lock starvation
//       cascades through the runlist;
//   (c) reconciliation: at num_invocations=1 (e.g. bug_c shape) there is
//       no replay → missing locks degenerate → conduit ≡ stateful
//       byte-equivalent (Task #78 still holds, no contradiction).
//
// Mechanism: the memtile-side `_link_prod_lock_<N>` / `_link_cons_lock_<N>`
// (init = depth × bd_repeat / 0) throttle the memtile MM2S → compute hop
// per slice, but they do NOT back-pressure the shim S2MM → memtile hop.
// In stateful, the shim-side `_prod_lock_0` / `_cons_lock_0` (both init=0)
// are referenced by the host runtime's `aiex.npu.dma_memcpy_nd` token
// signaling — completion of one shim BD fire is gated by the memtile
// draining the prior fire's data. Without these shim locks (conduit), all
// `repeat_count + 1` BDs fire back-to-back into the memtile S2MM port; the
// memtile sees the LAST iteration's data N times → compute reads stale /
// wrong A and B tiles.
//
// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------
// Skip site (the bug):
//   mlir-aie/lib/Dialect/Conduit/Transforms/ConduitToDMARoute.cpp:408-409
// Symmetric S2MM path (NOT skipped — emits shim locks for shim-consumer):
//   mlir-aie/lib/Dialect/Conduit/Transforms/ConduitToDMARoute.cpp:562-581
// Memtile-side slice locks (correct, untouched by this bug):
//   mlir-aie/lib/Dialect/Conduit/Transforms/ConduitToDMALink.cpp:489-557
// Smoking-gun commit:
//   `a50b27ced7` (2026-03-24) — also added an enshrining regression test
//   `mlir-aie/test/Dialect/Conduit/dma/conduit_shim_to_memtile_relay_locks.mlir`
//   whose CHECK-NOT lines pin the buggy state and will need to flip too.
//
// ---------------------------------------------------------------------------
// Forward-flip (when the fix lands)
// ---------------------------------------------------------------------------
// The CHECK-NOT lines below MUST flip to positive CHECK lines that pin:
//
//   aie.lock(%shim_noc_tile_2_0, {{[0-9]+}})
//     {init = 0 : i32, sym_name = "src_prod_lock_0"}
//   aie.lock(%shim_noc_tile_2_0, {{[0-9]+}})
//     {init = 0 : i32, sym_name = "src_cons_lock_0"}
//
// ---------------------------------------------------------------------------
// Topology (matches Llama A_L3L2_ / B_L3L2_ distribute pattern)
// ---------------------------------------------------------------------------
//   shim(2,0) --[src, depth=2]--> memtile(2,1) --[d1, d2, d3, depth=2]--> {
//       compute(2,2), compute(2,3), compute(3,3)
//   }
//   aie.objectfifo.link [@src] -> [@d1, @d2, @d3]      // distribute link
//
// `src` ends up in `linkSrcNamesEarly` → bug-skip branch fires.
//
// ---------------------------------------------------------------------------

// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: aie.device

// Shim DMA allocation IS emitted (proves the skip is in shim-side LOCK
// allocation only — the flow + alloc still exist, but the locks that
// would back-pressure them do not).
// CHECK: aie.shim_dma_allocation @src_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
// CHECK: aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)

// Memtile-side slice locks ARE emitted (one pair per distribute slice).
// CHECK: aie.lock(%mem_tile_2_1, {{[0-9]+}}) {init = 2 : i32, sym_name = "src_link_prod_lock_0"
// CHECK: aie.lock(%mem_tile_2_1, {{[0-9]+}}) {init = 0 : i32, sym_name = "src_link_cons_lock_0"
// CHECK: aie.lock(%mem_tile_2_1, {{[0-9]+}}) {init = 2 : i32, sym_name = "src_link_prod_lock_1"
// CHECK: aie.lock(%mem_tile_2_1, {{[0-9]+}}) {init = 0 : i32, sym_name = "src_link_cons_lock_1"
// CHECK: aie.lock(%mem_tile_2_1, {{[0-9]+}}) {init = 2 : i32, sym_name = "src_link_prod_lock_2"
// CHECK: aie.lock(%mem_tile_2_1, {{[0-9]+}}) {init = 0 : i32, sym_name = "src_link_cons_lock_2"

// BUG: Pass C does NOT emit shim-side _prod_lock_0 / _cons_lock_0 for the
// distribute-link source. Stateful emits both (both on shim tile, both
// init=0). When the fix lands, flip these CHECK-NOTs to positive CHECKs
// per the "Forward-flip" section above.
// CHECK-NOT: sym_name = "src_prod_lock_0"
// CHECK-NOT: sym_name = "src_cons_lock_0"

module @mm2s_missing_shim_locks_BUG {
  aie.device(xcve2302) {
    %shim = aie.tile(2, 0)
    %mem  = aie.tile(2, 1)
    %t22  = aie.tile(2, 2)
    %t23  = aie.tile(2, 3)
    %t33  = aie.tile(3, 3)

    // Source: shim → MemTile, depth=2, 48 elements (Llama-shape: large tile).
    aie.objectfifo @src (%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    // 3 destinations: MemTile → compute tiles, depth=2 each (mirrors Llama
    // A_L3L2_ which feeds 4 compute consumers via distribute link).
    aie.objectfifo @d1 (%mem, {%t22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d2 (%mem, {%t23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @d3 (%mem, {%t33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

    // Distribute link: marks @src as `linkSrcNamesEarly` → triggers the
    // bug-skip branch in ConduitToDMARoute.cpp:408-409.
    aie.objectfifo.link [@src] -> [@d1, @d2, @d3] ([][0, 16, 36])
  }
}
