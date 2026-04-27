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
// memtile distribute with compute-tile producer + multi-consumer
// destinations.
//
// Companion to:
//   - conduit_to_dma_link_lookup_unqualified_key_multi_device.mlir
//     (1->1 READ-side pin)
//   - conduit_to_dma_link_join_unqualified_write_multi_device.mlir
//     (JOIN write-side pin: L743 + L967)
//
// This fixture targets the DISTRIBUTE-side memtile-relay write sites
// that the JOIN fixture does not exercise:
//   - L880  state.insertS2MMChannel(dstName, consIdx, s2mmCh, linkOp.op)
//           per-consumer S2MM allocation in the circuit-broadcast loop
//   - L910  state.insertMM2SChannel(srcName, srcPort, linkOp.op)
//           compute-producer MM2S allocation for the src->memtile flow
//
// Both writes were keyed unqualified pre-fix; multi-device modules that
// reuse the same conduit names across devices would have the second
// device's writes overwrite the first.  Post-fix the writes are keyed
// device-qualified via state.makeConduitKey, so each device's
// allocations live in a distinct conduitMap slot.
//
// Topology (per device, both devices identical):
//
//   compute(0,2) ── op_in ──> memtile(0,1)
//                            distribute ([0, 32]) ──> op_out_a ──> compute(0,3)
//                                                ──> op_out_b ──> compute(0,4)
//
// Pass A lowers the link to a ScatterOp{src=@op_in, dsts=[@op_out_a,
// @op_out_b]} on the memtile.  In linkPhase:
//   - L880 fires twice (once per dst in the circuit broadcast loop)
//   - L910 fires once (compute producer -> memtile MM2S flow)
//
// CHECK discipline: pin the post-fix CORRECT BD chain channels.  In
// each device, the compute(0,2) MM2S BD chain must match the producer
// flow's MM2S channel; the consumer S2MM BD chains on (0,3)/(0,4) must
// match the per-consumer flows.  Defense-in-depth CHECK-NOTs catch any
// flow/BD-chain channel drift that would surface from a regression at
// either write site.

// CHECK-LABEL: module @link_distribute_unqualified_write_multi_device

// -----------------------------------------------------------------
// FIRST device.
// -----------------------------------------------------------------
// CHECK: aie.device(npu2) @first
// Producer flow: compute(0,2) DMA -> memtile(0,1) DMA.
// CHECK-DAG: aie.flow(%{{.*}}tile_0_2{{.*}}, DMA : 0, %{{.*}}mem_tile_0_1{{.*}}, DMA : 0)
// Per-consumer broadcast flows from memtile MM2S to each consumer S2MM.
// CHECK-DAG: aie.flow(%{{.*}}mem_tile_0_1{{.*}}, DMA : 0, %{{.*}}tile_0_3{{.*}}, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}mem_tile_0_1{{.*}}, DMA : 1, %{{.*}}tile_0_4{{.*}}, DMA : 0)
// Compute-producer MM2S BD chain must use channel 0 (matches flow).
// Pre-fix L1432 read of conduitMM2SChannel via unqualified key would miss
// in the multi-device-overwrite scenario and fall through to the
// preUsedMM2SChannels allocator, which can return a different value.
// CHECK: aie.mem(%{{.*}}tile_0_2{{.*}})
// CHECK: aie.dma_start(MM2S, 0
// Each consumer S2MM BD chain must use channel 0.
// CHECK: aie.mem(%{{.*}}tile_0_3{{.*}})
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.mem(%{{.*}}tile_0_4{{.*}})
// CHECK: aie.dma_start(S2MM, 0

// -----------------------------------------------------------------
// SECOND device.  Identical topology + identical conduit names.
// -----------------------------------------------------------------
// CHECK: aie.device(npu2) @second
// CHECK-DAG: aie.flow(%{{.*}}tile_0_2{{.*}}, DMA : 0, %{{.*}}mem_tile_0_1{{.*}}, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}mem_tile_0_1{{.*}}, DMA : 0, %{{.*}}tile_0_3{{.*}}, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}mem_tile_0_1{{.*}}, DMA : 1, %{{.*}}tile_0_4{{.*}}, DMA : 0)
// CHECK: aie.mem(%{{.*}}tile_0_2{{.*}})
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.mem(%{{.*}}tile_0_3{{.*}})
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.mem(%{{.*}}tile_0_4{{.*}})
// CHECK: aie.dma_start(S2MM, 0

// Defense-in-depth: no compute-tile MM2S BD chain may be on channel 1
// (the buggy emit when L1432 falls through to the preUsedMM2SChannels
// allocator with a non-zero next-free slot).  aie.mem only contains
// compute-tile DMA chains; memtile uses aie.memtile_dma.
// CHECK-NOT: aie.dma_start(MM2S, 1

module @link_distribute_unqualified_write_multi_device {
  aie.device(npu2) @first {
    %shim     = aie.tile(0, 0)
    %mem      = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    aie.objectfifo @op_in    (%tile_0_2, {%mem},      2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @op_out_a (%mem,      {%tile_0_3}, 2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @op_out_b (%mem,      {%tile_0_4}, 2 : i32)
        : !aie.objectfifo<memref<32xbf16>>

    aie.objectfifo.link [@op_in] -> [@op_out_a, @op_out_b] ([] [0, 32])

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @op_in(Produce, 1)
              : !aie.objectfifosubview<memref<64xbf16>>
      aie.objectfifo.release @op_in(Produce, 1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %0 = aie.objectfifo.acquire @op_out_a(Consume, 1)
              : !aie.objectfifosubview<memref<32xbf16>>
      aie.objectfifo.release @op_out_a(Consume, 1)
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %0 = aie.objectfifo.acquire @op_out_b(Consume, 1)
              : !aie.objectfifosubview<memref<32xbf16>>
      aie.objectfifo.release @op_out_b(Consume, 1)
      aie.end
    }
  }

  aie.device(npu2) @second {
    %shim     = aie.tile(0, 0)
    %mem      = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    aie.objectfifo @op_in    (%tile_0_2, {%mem},      2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @op_out_a (%mem,      {%tile_0_3}, 2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @op_out_b (%mem,      {%tile_0_4}, 2 : i32)
        : !aie.objectfifo<memref<32xbf16>>

    aie.objectfifo.link [@op_in] -> [@op_out_a, @op_out_b] ([] [0, 32])

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @op_in(Produce, 1)
              : !aie.objectfifosubview<memref<64xbf16>>
      aie.objectfifo.release @op_in(Produce, 1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %0 = aie.objectfifo.acquire @op_out_a(Consume, 1)
              : !aie.objectfifosubview<memref<32xbf16>>
      aie.objectfifo.release @op_out_a(Consume, 1)
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %0 = aie.objectfifo.acquire @op_out_b(Consume, 1)
              : !aie.objectfifosubview<memref<32xbf16>>
      aie.objectfifo.release @op_out_b(Consume, 1)
      aie.end
    }
  }
}
