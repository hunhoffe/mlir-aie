// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test: MemTile relay channel feeding into a multi-source join.
//
// Tests the Phase 3j + Phase 5 fix in ConduitToDMALink.cpp for MemTile→MemTile
// relay channels consumed as join sources at the destination MemTile.
//
// Topology:
//   tile(3,2) → MemTile(3,1) ──relay──→ MemTile(5,1) ─┐
//   tile(5,2) → MemTile(5,1) ─────────────────────────────┼── join ──→ shim(5,0)
//
// Expected:
//   4 aie.flow: compute→memtile (×2), memtile→memtile (relay), memtile→shim
//   MemTile(3,1): 1 S2MM + 1 MM2S (relay forwarding)
//   MemTile(5,1): 2 S2MM + 1 MM2S (join + output)

// CHECK-LABEL: module @memtile_relay_join_source
// CHECK:   aie.device(npu2) {

// --- Shim DMA allocation for join output ---
// CHECK: aie.shim_dma_allocation @join_out_shim_alloc

// --- MemTile(5,1)→shim output flow ---
// CHECK: aie.flow(%mem_tile_5_1, DMA : 0, %shim_noc_tile_5_0, DMA : 0)

// --- Relay flow: MemTile(3,1)→MemTile(5,1) ---
// CHECK: aie.flow(%mem_tile_3_1, DMA : 0, %mem_tile_5_1, DMA : 0)

// --- Compute→MemTile flow for source A ---
// CHECK: aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 0)

// --- MemTile(3,1) relay DMA: 1 S2MM (ingest from tile_3_2) + 1 MM2S (relay out) ---
// CHECK:     aie.memtile_dma(%mem_tile_3_1) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.dma_start(MM2S, 0,
// CHECK:       aie.end
// CHECK:     }

// --- Compute→MemTile flow for source B ---
// CHECK: aie.flow(%tile_5_2, DMA : 0, %mem_tile_5_1, DMA : 1)

// --- MemTile(5,1) join DMA: 2 S2MM (relay + direct) + 1 MM2S (joined output) ---
// CHECK:     aie.memtile_dma(%mem_tile_5_1) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.dma_start(S2MM, 1,
// CHECK:       aie.dma_start(MM2S, 0,
// CHECK:       aie.end
// CHECK:     }

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.link

module @memtile_relay_join_source {
  aie.device(npu2) {
    %shim_5_0 = aie.tile(5, 0)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %tile_3_2 = aie.tile(3, 2)
    %tile_5_2 = aie.tile(5, 2)

    // Source A: compute tile(3,2) → local MemTile(3,1)
    aie.objectfifo @prod_a (%tile_3_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    // Relay: MemTile(3,1) → MemTile(5,1)
    aie.objectfifo @relay (%mem_tile_3_1, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    // Relay link at MemTile(3,1): forward prod_a → relay
    aie.objectfifo.link [@prod_a] -> [@relay] ([][])

    // Source B: compute tile(5,2) → local MemTile(5,1)
    aie.objectfifo @prod_b (%tile_5_2, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    // Joined output: MemTile(5,1) → shim(5,0)
    aie.objectfifo @join_out (%mem_tile_5_1, {%shim_5_0}, 2 : i32) : !aie.objectfifo<memref<128xi8>>

    // Join link at MemTile(5,1): combine relay + prod_b → join_out
    aie.objectfifo.link [@relay, @prod_b] -> [@join_out] ([0, 64][])
  }
}
