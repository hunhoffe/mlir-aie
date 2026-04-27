//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// Pass C linkPhase device-qualified-key WRITE-side regression pin.
//
// Companion to conduit_to_dma_link_lookup_unqualified_key_multi_device.mlir
// (which pins the READ-side fix for a 1->1 link). That fixture exercises
// only the read sites at L280 / L682 / L700 / L723 / L750 of
// ConduitToDMALink.cpp and does NOT touch any of the WRITE sites that the
// multi-device key bug audit identified (L394 / L743 / L799 / L848 / L889 /
// L922 / L967 / L1000). A regression at any of those write sites would
// therefore not be caught by the read-side fixture.
//
// This fixture exercises L967: the JOIN source-MM2S writeback.
//   `state.conduitMM2SChannel[srcKey] = srcMM2SCh;`
// (post-fix; pre-fix used `srcName` directly — unqualified.) It also
// exercises L743 (joinS2MMChannels writeback) as a side-effect of the
// JOIN path. Both are post-fix written under the device-qualified key
// `srcName__d<idx>` produced by `state.makeConduitKey(srcName, linkOp.op)`.
//
// ----------------------------------------------------------------------------
// Bug class
// ----------------------------------------------------------------------------
// Phase 5.5a (Link.cpp:1411-1461) iterates `state.conduitMap` whose keys
// are device-qualified (e.g. `j_a__d0`). It looks up the previously
// allocated MM2S channel via `state.conduitMM2SChannel.find(name)` where
// `name` is the QUALIFIED iteration key. With the pre-fix unqualified
// write at L967 the lookup MISSES, falls through to the `preUsedMM2SChannels`
// dynamic allocation path, and bumps the channel up by one — ending up at
// MM2S channel 1 (since channel 0 was already inserted into preUsed by L970).
// The Phase-5 flow (emitted at L971-L974) used the original srcMM2SCh = 0,
// so the resulting `aie.mem` BD chain uses MM2S channel 1 while the
// `aie.flow` declares DMA : 0 — channel mismatch, BD chain blocks forever.
//
// Multi-device-only: in single-device mode `makeConduitKey` returns the
// name unchanged, so qualified == unqualified and the lookup hits.
//
// ----------------------------------------------------------------------------
// Topology (minimum reproducer of the JOIN compute-tile-source pattern)
// ----------------------------------------------------------------------------
//
//   compute(0,2) ─┐
//                 ├──> memtile(0,1) ──> shim(0,0)
//   compute(0,3) ─┘
//
// Two devices, each with the same JOIN topology and the same conduit
// names (j_a, j_b, j_d). The pre-fix bug triggers because both devices
// write into the same unqualified `conduitMM2SChannel["j_a"]` /
// `["j_b"]` slot — the second device's write overwrites the first, and
// subsequently every Phase 5.5a lookup MISSES the qualified key.
//
// ----------------------------------------------------------------------------
// CHECK discipline
// ----------------------------------------------------------------------------
// Pin the post-fix CORRECT behavior: every `aie.mem(<compute_tile>)` BD
// chain MUST `aie.dma_start(MM2S, 0, ...)` to match the `aie.flow(...,
// DMA : 0, ...)` emitted at L971. Pre-fix the BD chain would emit
// `aie.dma_start(MM2S, 1, ...)` — flow/BD-chain channel mismatch.

// CHECK-LABEL: module @link_join_unqualified_write_multi_device

// FIRST device.
// CHECK: aie.device(npu2) @first
// CHECK-DAG: aie.flow(%[[T02_FIRST:.*]], DMA : 0, %{{.*}}mem_tile_0_1{{.*}}, DMA : 0)
// CHECK-DAG: aie.flow(%[[T03_FIRST:.*]], DMA : 0, %{{.*}}mem_tile_0_1{{.*}}, DMA : 1)
// MM2S BD chain on each compute tile must use channel 0 (matching the
// flow). Pre-fix this is MM2S 1 (Phase 5.5a fallback allocates next-free).
// CHECK: aie.mem(%{{.*}}tile_0_2{{.*}})
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.mem(%{{.*}}tile_0_3{{.*}})
// CHECK: aie.dma_start(MM2S, 0

// SECOND device. Same pin. Pre-fix this is also MM2S 1 because the
// Phase 5.5a lookup misses for the `j_*__d1` qualified keys too.
// CHECK: aie.device(npu2) @second
// CHECK-DAG: aie.flow(%[[T02_SECOND:.*]], DMA : 0, %{{.*}}mem_tile_0_1{{.*}}, DMA : 0)
// CHECK-DAG: aie.flow(%[[T03_SECOND:.*]], DMA : 0, %{{.*}}mem_tile_0_1{{.*}}, DMA : 1)
// CHECK: aie.mem(%{{.*}}tile_0_2{{.*}})
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.mem(%{{.*}}tile_0_3{{.*}})
// CHECK: aie.dma_start(MM2S, 0

// Defense-in-depth: no compute-tile MM2S BD chain may be on channel 1
// (the buggy emit). aie.mem only contains compute-tile DMA; memtile
// uses aie.memtile_dma. So a `dma_start(MM2S, 1` inside any aie.mem
// region is the bug signature. A single CHECK-NOT after the entire
// module catches it independent of ordering.
// CHECK-NOT: aie.dma_start(MM2S, 1

module @link_join_unqualified_write_multi_device {
  aie.device(npu2) @first {
    %shim     = aie.tile(0, 0)
    %mem      = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @j_a (%tile_0_2, {%mem}, 2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @j_b (%tile_0_3, {%mem}, 2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @j_d (%mem, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // JOIN: 2 compute-tile sources -> 1 shim destination via memtile.
    // Per-source byte offsets [0, 32]; each source contributes 32 bf16
    // to the 64-bf16 dst.
    aie.objectfifo.link [@j_a, @j_b] -> [@j_d] ([0, 32][])

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @j_a(Produce, 1)
              : !aie.objectfifosubview<memref<32xbf16>>
      aie.objectfifo.release @j_a(Produce, 1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %0 = aie.objectfifo.acquire @j_b(Produce, 1)
              : !aie.objectfifosubview<memref<32xbf16>>
      aie.objectfifo.release @j_b(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<64xbf16>) {
      %t = aiex.dma_configure_task_for @j_d {
        aie.dma_bd(%arg0 : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }

  aie.device(npu2) @second {
    %shim     = aie.tile(0, 0)
    %mem      = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @j_a (%tile_0_2, {%mem}, 2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @j_b (%tile_0_3, {%mem}, 2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @j_d (%mem, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo.link [@j_a, @j_b] -> [@j_d] ([0, 32][])

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @j_a(Produce, 1)
              : !aie.objectfifosubview<memref<32xbf16>>
      aie.objectfifo.release @j_a(Produce, 1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %0 = aie.objectfifo.acquire @j_b(Produce, 1)
              : !aie.objectfifosubview<memref<32xbf16>>
      aie.objectfifo.release @j_b(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<64xbf16>) {
      %t = aiex.dma_configure_task_for @j_d {
        aie.dma_bd(%arg0 : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
