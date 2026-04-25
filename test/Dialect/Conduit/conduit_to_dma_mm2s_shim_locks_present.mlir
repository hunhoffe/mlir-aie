//===- conduit_to_dma_mm2s_shim_locks_present.mlir ------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Regression test (#96): Pass C emits shim-side _prod_lock_0 / _cons_lock_0
// for MM2S input channels that are distribute-link sources (shim → memtile
// → multiple compute consumers).
//
// ---------------------------------------------------------------------------
// Why both shim AND memtile locks are required
// ---------------------------------------------------------------------------
// The memtile-side `_link_prod_lock_<N>` / `_link_cons_lock_<N>`
// (init = depth × bd_repeat / 0, allocated by ConduitToDMALink.cpp:489-557)
// throttle the memtile MM2S → compute hop per slice, but they do NOT
// back-pressure the shim S2MM → memtile hop. The shim-side `_prod_lock_0`
// / `_cons_lock_0` (both init=0, allocated by Phase 4a in
// ConduitToDMARoute.cpp) are referenced by the host runtime's
// `aiex.npu.dma_memcpy_nd` token signaling — completion of one shim BD
// fire is gated by the memtile draining the prior fire's data. Without
// these shim locks, all `repeat_count + 1` BDs fire back-to-back into the
// memtile S2MM port; the memtile sees the LAST iteration's data N times
// and compute reads stale / wrong tiles.
//
// Failure mode this test guards against (pre-#96): K=2048 GEMM
// (`repeat_count = 3` → 4 fires per dispatch) → 50%-zero output rows;
// K=64 → ~6% wrong rows; Llama prefill TIMEOUTs at attn_key once lock
// starvation cascades through the runlist. Byte-diff evidence (Task #94):
// 12 distribute-link-source MM2S input channels (4 A_L3L2_ + 8 B_L3L2_)
// × 2 shim-side locks each = exactly 24 missing lock ops vs stateful
// before #96.
//
// ---------------------------------------------------------------------------
// Topology (matches Llama A_L3L2_ / B_L3L2_ distribute pattern)
// ---------------------------------------------------------------------------
//   shim(2,0) --[src, depth=2]--> memtile(2,1) --[d1, d2, d3, depth=2]--> {
//       compute(2,2), compute(2,3), compute(3,3)
//   }
//   aie.objectfifo.link [@src] -> [@d1, @d2, @d3]      // distribute link
//
// `src` ends up in `linkSrcNamesEarly`; Phase 4a allocates shim-side
// locks unconditionally for AIE2 regardless.
//
// ---------------------------------------------------------------------------

// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: aie.device

// Shim tile (2,0) prod/cons locks for the distribute-link source: init=0
// for both, programmed by host runtime aiex.npu.dma_memcpy_nd token signaling.
// CHECK: aie.lock(%shim_noc_tile_2_0, {{[0-9]+}}) {init = 0 : i32, sym_name = "src_prod_lock_0"
// CHECK: aie.lock(%shim_noc_tile_2_0, {{[0-9]+}}) {init = 0 : i32, sym_name = "src_cons_lock_0"

// Shim DMA allocation + flow shim→MemTile (Phase 4a).
// CHECK: aie.shim_dma_allocation @src_shim_alloc(%shim_noc_tile_2_0, MM2S, 0)
// CHECK: aie.flow(%shim_noc_tile_2_0, DMA : 0, %mem_tile_2_1, DMA : 0)

// Memtile-side slice locks (one pair per distribute slice) emitted by linkPhase.
// CHECK: aie.lock(%mem_tile_2_1, {{[0-9]+}}) {init = 2 : i32, sym_name = "src_link_prod_lock_0"
// CHECK: aie.lock(%mem_tile_2_1, {{[0-9]+}}) {init = 0 : i32, sym_name = "src_link_cons_lock_0"
// CHECK: aie.lock(%mem_tile_2_1, {{[0-9]+}}) {init = 2 : i32, sym_name = "src_link_prod_lock_1"
// CHECK: aie.lock(%mem_tile_2_1, {{[0-9]+}}) {init = 0 : i32, sym_name = "src_link_cons_lock_1"
// CHECK: aie.lock(%mem_tile_2_1, {{[0-9]+}}) {init = 2 : i32, sym_name = "src_link_prod_lock_2"
// CHECK: aie.lock(%mem_tile_2_1, {{[0-9]+}}) {init = 0 : i32, sym_name = "src_link_cons_lock_2"

module @mm2s_shim_locks_present {
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

    // Distribute link: marks @src as `linkSrcNamesEarly`. Pre-#96 this
    // triggered an early-skip in Phase 4a; post-#96 shim locks are
    // allocated unconditionally for AIE2.
    aie.objectfifo.link [@src] -> [@d1, @d2, @d3] ([][0, 16, 36])
  }
}
