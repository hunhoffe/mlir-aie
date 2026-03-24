// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test: B-2 — S2MM channel consistency when a MemTile is both a
// relay source (for a join) and a broadcast consumer.
//
// Bug: when the join link allocates a new S2MM channel for a non-MemTile relay
// source, the channel was NOT recorded in conduitConsS2MMChannel. A subsequent
// lookup (e.g., from a broadcast consumer path on the same conduit) would then
// allocate a DIFFERENT S2MM channel, causing:
//   1. The aie.flow to point to the wrong S2MM channel.
//   2. The DMAStartOp to use the wrong S2MM channel.
//   This mismatch causes a hardware deadlock.
//
// B-2 fix: after allocating a new join S2MM channel, record it in
// conduitConsS2MMChannel[{sName, 0u}] immediately, so any subsequent
// lookup (broadcast consumer path, forward link) finds the same channel.
//
// Topology:
//   tile(2,2) [join source A] → MemTile(2,1) [join hub] → shim(2,0)
//   tile(3,2) [join source B] → MemTile(2,1) [join hub]
//
// The join source S2MM channels allocated in Phase 5 (join path) must
// match the aie.flow ops emitted for the same conduits.
//
// CHECK-LABEL: module @conduit_relay_join_s2mm_consistent
// CHECK: aie.device

// Verify: exactly 2 source flows into the join MemTile,
// with matching S2MM channels in both flows and DMAStartOps.
// Source A flow: tile(2,2) → MemTile(2,1) S2MM 0
// CHECK: aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 0)
// Source B flow: tile(3,2) → MemTile(2,1) S2MM 1
// CHECK: aie.flow(%tile_3_2, DMA : 0, %mem_tile_2_1, DMA : 1)

// MemTile DMA: 2 S2MM channels (0 for src_a, 1 for src_b) + 1 MM2S
// CHECK: aie.memtile_dma(%mem_tile_2_1)
// CHECK:   aie.dma_start(S2MM, 0,
// CHECK:   aie.dma_start(S2MM, 1,
// CHECK:   aie.dma_start(MM2S, 0,
// CHECK:   aie.end

// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.link

module @conduit_relay_join_s2mm_consistent {
  aie.device(npu2) {
    %shim = aie.tile(2, 0)
    %mem_tile = aie.tile(2, 1)
    %tile_a = aie.tile(2, 2)
    %tile_b = aie.tile(3, 2)

    // Source A: tile(2,2) → MemTile(2,1)
    aie.objectfifo @src_a (%tile_a, {%mem_tile}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    // Source B: tile(3,2) → MemTile(2,1)
    aie.objectfifo @src_b (%tile_b, {%mem_tile}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    // Joined output: MemTile(2,1) → shim(2,0)
    aie.objectfifo @join_out (%mem_tile, {%shim}, 2 : i32) : !aie.objectfifo<memref<128xi8>>

    // Join link: combine src_a + src_b → join_out
    aie.objectfifo.link [@src_a, @src_b] -> [@join_out] ([0, 64][])

    %core_a = aie.core(%tile_a) {
      aie.end
    }
    %core_b = aie.core(%tile_b) {
      aie.end
    }
  }
}
