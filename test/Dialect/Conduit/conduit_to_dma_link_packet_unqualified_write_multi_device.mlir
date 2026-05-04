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
// PACKET-MODE distribute-broadcast variant.
//
// Sister fixture to:
//   conduit_to_dma_link_distribute_unqualified_write_multi_device.mlir
//     (CIRCUIT-MODE distribute, same topology, same L949 surface)
//
// Why both circuit + packet variants?
//   linkPhase has two top-level distribute branches: CIRCUIT (around L827)
//   and PACKET (L780+). The two branches share a producer-MM2S writeback
//   site (now `state.insertMM2SChannel(srcName, srcPort, linkOp.op)` at
//   ConduitToDMALink.cpp:910 for the packet branch, L949 in the JOIN/
//   distribute else-branch is the analog for CIRCUIT). This fixture pins
//   the PACKET branch's compute-source MM2S writeback so a regression
//   confined to the packet branch (e.g. someone reverts L910 only and
//   leaves L949 qualified) is caught independent of the CIRCUIT fixture.
//
// Honest scope (the team-lead Task #16 brief listed L799 + L848 as the
// nominal targets here; both have NO direct IR-diff surface in symmetric
// lit topologies — see "Surface analysis" below):
//
//   - L799  state.insertPacketID(dstName, *pktID, linkOp.op)
//     Per-dst packet ID write. Downstream reads at L1267 (qualified-then-
//     unqualified helper) happen IN THE SAME LINK BRANCH — write+read are
//     sequenced within one device, no cross-device dependency. The other
//     read site at L1746 only fires for compute-tile producers, but
//     packet-broadcast distribute has a MEMTILE producer, so L1746 does
//     not fire here. Result: the unqualified-vs-qualified key distinction
//     produces no observable IR diff in any small lit topology we can
//     construct from objectfifo IR.
//
//   - L848  state.insertS2MMChannel(dstName, consIdx, s2mmCh, linkOp.op)
//     Per-consumer S2MM write inside the packet broadcast loop. Downstream
//     read at L2255 (Phase 5.5, direct unqualified `find`) WOULD miss
//     post-fix qualified writes — but the fallback allocator at L2259
//     (preUsedS2MMChannels[consTileVal] starting at 0) produces the SAME
//     channel value as the writer in any topology where the consumer tile
//     has no prior S2MM allocations. In symmetric lit shapes, every
//     consumer tile sees exactly one S2MM allocation → writer + fallback
//     both return 0 → no IR diff. Surfacing requires multi-flow consumer
//     pollution, which lives outside this fixture's scope.
//
// What this fixture DOES pin:
//
//   - The packet-broadcast distribute branch's producer-MM2S writeback
//     surface, observable as `aie.dma_start(MM2S, 0)` on the compute
//     producer (tile_0_2). Pre-fix any unqualified-write regression that
//     pollutes the lookup at ConduitToDMALink.cpp:1432 (Phase 5.5a) for
//     this branch produces `aie.dma_start(MM2S, 1)` (fallback bumps past
//     the inserted preUsed slot) — a flow/BD-chain channel mismatch that
//     blocks the memtile DMA forever.
//
// Topology (per device, both devices identical — symmetric):
//
//   compute(0,2) ── op_in ──> memtile(0,1)
//                            distribute (packet) ──> op_out_a ──> compute(0,3)
//                                                ──> op_out_b ──> compute(0,4)
//
// op_out_a / op_out_b are tagged routing_mode = "packet" so the memtile
// distribute path takes the PACKET branch (L780+) instead of CIRCUIT.

// CHECK-LABEL: module @link_packet_unqualified_write_multi_device

// -----------------------------------------------------------------
// FIRST device.
// -----------------------------------------------------------------
// CHECK: aie.device(npu2) @first
// One packet_flow per packet-routed dst conduit (defense-in-depth: the
// PACKET branch is being taken; otherwise this fixture would not exercise
// the targeted code path at all).
// CHECK-COUNT-2: aie.packet_flow
// Compute producer tile_0_2 MM2S BD chain MUST be on channel 0 (matching
// the upstream flow). Pre-fix regression: `aie.dma_start(MM2S, 1` here.
// CHECK: aie.mem(%{{.*}}tile_0_2{{.*}})
// CHECK: aie.dma_start(MM2S, 0
// Per-consumer S2MM BD chains use channel 0 (single-allocation collapse;
// asserted as a sanity check, not as a direct L848 pin — see header).
// CHECK: aie.mem(%{{.*}}tile_0_3{{.*}})
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.mem(%{{.*}}tile_0_4{{.*}})
// CHECK: aie.dma_start(S2MM, 0

// -----------------------------------------------------------------
// SECOND device.  Identical topology + identical conduit names.
// -----------------------------------------------------------------
// CHECK: aie.device(npu2) @second
// CHECK-COUNT-2: aie.packet_flow
// CHECK: aie.mem(%{{.*}}tile_0_2{{.*}})
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.mem(%{{.*}}tile_0_3{{.*}})
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.mem(%{{.*}}tile_0_4{{.*}})
// CHECK: aie.dma_start(S2MM, 0

// Bug-signature CHECK-NOTs — these are the actual pin.
// aie.mem only contains compute-tile DMA chains (memtile uses
// aie.memtile_dma), so a `dma_start(MM2S, 1` inside the module would
// have to be on a compute tile — the bug signature.
// CHECK-NOT: aie.dma_start(MM2S, 1
// Defense-in-depth for any future consumer-side regression that does
// surface (e.g. if L848 is reverted AND the consumer is polluted by
// some other code path).
// CHECK-NOT: aie.dma_start(S2MM, 1

module @link_packet_unqualified_write_multi_device {
  aie.device(npu2) @first {
    %shim     = aie.tile(0, 0)
    %mem      = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    aie.objectfifo @op_in    (%tile_0_2, {%mem},      2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @op_out_a (%mem,      {%tile_0_3}, 2 : i32)
        {routing_mode = "packet"} : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @op_out_b (%mem,      {%tile_0_4}, 2 : i32)
        {routing_mode = "packet"} : !aie.objectfifo<memref<32xbf16>>

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
        {routing_mode = "packet"} : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @op_out_b (%mem,      {%tile_0_4}, 2 : i32)
        {routing_mode = "packet"} : !aie.objectfifo<memref<32xbf16>>

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
