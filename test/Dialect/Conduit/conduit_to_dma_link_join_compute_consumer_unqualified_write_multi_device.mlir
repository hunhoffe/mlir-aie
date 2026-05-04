//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// Pass C linkPhase device-qualified-key WRITE-side regression pin —
// JOIN with COMPUTE-tile consumer destination (L982 path).
//
// Sister fixture to:
//   conduit_to_dma_link_join_unqualified_write_multi_device.mlir
//     (JOIN with SHIM consumer destination — exercises the Phase 4b
//      shim-dst path; this fixture exercises the in-linkPhase compute-dst
//      path that emits per-consumer flows directly at L983.)
//
// Why both shim + compute consumer variants of JOIN?
//   linkPhase L965-L992 has two paths for the JOIN destination:
//     (a) compute-tile consumer: emit memtile->compute flow inline +
//         allocate per-consumer S2MM channel via L975, then write the
//         channel via the qualified-key inserter at L982:
//           state.insertS2MMChannel(dstName, ci, consS2MM, linkOp.op)
//     (b) shim consumer: skip emission here — Phase 4b emits the shim
//         flow separately. The shim path does NOT exercise L982.
//   The existing join-shim fixture pins L949 (compute-source MM2S
//   writeback) for the SHIM-DST topology. This fixture pins the same
//   L949 surface but for the COMPUTE-DST topology — a regression that
//   only breaks the compute-dst branch (e.g. someone reverts L982 only
//   while leaving the shim-dst Phase 4b writes correct) gets caught here.
//
// Honest scope: L982 itself has NO direct IR-diff surface in symmetric
// lit topologies. The downstream read at ConduitToDMALink.cpp:2255
// (Phase 5.5 consumer-S2MM BD chain emission, direct unqualified `find`)
// would miss post-fix qualified writes — but the fallback allocator at
// L2259 (preUsedS2MMChannels[consTileVal] starting at 0) returns the
// SAME channel value as the writer when the consumer tile has no prior
// S2MM allocations (single-allocation collapse). Surfacing L982 directly
// would require a multi-flow consumer tile, which lives outside the
// minimum-reproducer discipline of this fixture.
//
// What this fixture DOES pin:
//
//   - L949 (state.insertMM2SChannel(sName, srcMM2SCh, linkOp.op) for the
//     JOIN compute-source MM2S writeback) via the COMPUTE-CONSUMER dst
//     branch. Observable as `aie.dma_start(MM2S, 0)` on each compute
//     source tile (matching the upstream `aie.flow(<src>, DMA : 0,
//     <memtile>, DMA : <ingest>)`). Pre-fix any unqualified-write
//     regression at L949 produces `aie.dma_start(MM2S, 1)` on the source
//     tile (Phase 5.5a fallback at L1432→L1437 bumps past the
//     pre-inserted preUsed slot).
//
// Topology (per device, both devices identical):
//
//   compute(0,2) ─┐
//                 ├──> memtile(0,1) ──> compute(0,4)
//   compute(0,3) ─┘
//
// JOIN with two compute sources merging into a single compute consumer
// (NOT a shim consumer — that path is covered by the sister fixture).

// CHECK-LABEL: module @link_join_compute_consumer_unqualified_write_multi_device

// -----------------------------------------------------------------
// FIRST device.
// -----------------------------------------------------------------
// CHECK: aie.device(npu2) @first
// Per-source flows (compute -> memtile, MM2S 0 / S2MM 0 and S2MM 1).
// CHECK-DAG: aie.flow(%[[T02_FIRST:.*]], DMA : 0, %{{.*}}mem_tile_0_1{{.*}}, DMA : 0)
// CHECK-DAG: aie.flow(%[[T03_FIRST:.*]], DMA : 0, %{{.*}}mem_tile_0_1{{.*}}, DMA : 1)
// Destination flow (memtile -> compute consumer).
// CHECK-DAG: aie.flow(%{{.*}}mem_tile_0_1{{.*}}, DMA : 0, %{{.*}}tile_0_4{{.*}}, DMA : 0)
// Compute-source MM2S BD chains MUST use channel 0 (matching upstream
// flow). Pre-fix the L949 regression emits `dma_start(MM2S, 1)` on each
// source tile due to Phase 5.5a fallback bumping past the preUsed slot.
// CHECK: aie.mem(%{{.*}}tile_0_2{{.*}})
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.mem(%{{.*}}tile_0_3{{.*}})
// CHECK: aie.dma_start(MM2S, 0
// Consumer S2MM BD chain on tile_0_4 must use channel 0 (sanity check;
// see header — not a direct L982 pin, but defense-in-depth catches any
// regression that does happen to surface as a consumer S2MM diff here).
// CHECK: aie.mem(%{{.*}}tile_0_4{{.*}})
// CHECK: aie.dma_start(S2MM, 0

// -----------------------------------------------------------------
// SECOND device.  Identical topology + identical conduit names.
// -----------------------------------------------------------------
// CHECK: aie.device(npu2) @second
// CHECK-DAG: aie.flow(%[[T02_SECOND:.*]], DMA : 0, %{{.*}}mem_tile_0_1{{.*}}, DMA : 0)
// CHECK-DAG: aie.flow(%[[T03_SECOND:.*]], DMA : 0, %{{.*}}mem_tile_0_1{{.*}}, DMA : 1)
// CHECK-DAG: aie.flow(%{{.*}}mem_tile_0_1{{.*}}, DMA : 0, %{{.*}}tile_0_4{{.*}}, DMA : 0)
// CHECK: aie.mem(%{{.*}}tile_0_2{{.*}})
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.mem(%{{.*}}tile_0_3{{.*}})
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.mem(%{{.*}}tile_0_4{{.*}})
// CHECK: aie.dma_start(S2MM, 0

// Bug-signature CHECK-NOTs — the actual pin.
// aie.mem holds compute-tile DMAs only (memtile uses aie.memtile_dma),
// so a `dma_start(MM2S, 1` anywhere in the module is the L949 bug
// signature.
// CHECK-NOT: aie.dma_start(MM2S, 1
// Defense-in-depth for any regression that surfaces as a non-zero
// compute-consumer S2MM channel (e.g. future L982-class bugs that
// happen to surface in larger fixtures).
// CHECK-NOT: aie.dma_start(S2MM, 1

module @link_join_compute_consumer_unqualified_write_multi_device {
  aie.device(npu2) @first {
    %shim     = aie.tile(0, 0)
    %mem      = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    aie.objectfifo @j_a (%tile_0_2, {%mem},      2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @j_b (%tile_0_3, {%mem},      2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @j_d (%mem,      {%tile_0_4}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // JOIN: 2 compute-tile sources -> 1 COMPUTE-tile destination via memtile.
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
    %core_0_4 = aie.core(%tile_0_4) {
      %0 = aie.objectfifo.acquire @j_d(Consume, 1)
              : !aie.objectfifosubview<memref<64xbf16>>
      aie.objectfifo.release @j_d(Consume, 1)
      aie.end
    }
  }

  aie.device(npu2) @second {
    %shim     = aie.tile(0, 0)
    %mem      = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    aie.objectfifo @j_a (%tile_0_2, {%mem},      2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @j_b (%tile_0_3, {%mem},      2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @j_d (%mem,      {%tile_0_4}, 2 : i32)
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
    %core_0_4 = aie.core(%tile_0_4) {
      %0 = aie.objectfifo.acquire @j_d(Consume, 1)
              : !aie.objectfifosubview<memref<64xbf16>>
      aie.objectfifo.release @j_d(Consume, 1)
      aie.end
    }
  }
}
