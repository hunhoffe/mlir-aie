// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// MemTile distribute to same-column compute tiles (pathfinder-safe pattern).
//
// This test verifies the safe pattern for MemTile→compute DMA distribution:
// all destination compute tiles are in the SAME column as the MemTile.
// Cross-column MemTile→compute flows trigger a pathfinder assertion crash
// (CRITICAL-1), so this pattern must always be used instead.
//
// Topology:
//   shim(2,0) → MemTile(2,1) → {tile(2,2), tile(2,3)}  [same column 2]
//
// Expected:
//   aie.flow(%shim..., DMA:0, %mem_tile_2_1, DMA:0)  [shim → MemTile]
//   aie.flow(%mem_tile_2_1, DMA:0, %tile_2_2, DMA:0)  [MemTile → compute, col 2]
//   aie.flow(%mem_tile_2_1, DMA:1, %tile_2_3, DMA:0)  [MemTile → compute, col 2]
//
// No cross-column MemTile flows should appear.

// CHECK-LABEL: module @memsame_column_relay
// CHECK:   aie.device(xcve2302) {

// --- MemTile distribute flows (all same column 2) ---
// CHECK: aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_2, DMA : 0)
// CHECK: aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_3, DMA : 0)

// --- No cross-column MemTile DMA flows ---
// CHECK-NOT: aie.flow(%mem_tile_2_1, DMA : {{[0-9]+}}, %tile_0_
// CHECK-NOT: aie.flow(%mem_tile_2_1, DMA : {{[0-9]+}}, %tile_1_
// CHECK-NOT: aie.flow(%mem_tile_2_1, DMA : {{[0-9]+}}, %tile_3_

// --- MemTile DMA block ---
// CHECK:     aie.memtile_dma(%mem_tile_2_1) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.dma_start(MM2S, 0,
// CHECK:       aie.dma_start(MM2S, 1,
// CHECK:       aie.end
// CHECK:     }

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.link

module @memsame_column_relay {
  aie.device(xcve2302) {
    %shim_2_0 = aie.tile(2, 0)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)

    // Shim → MemTile (ingest from host)
    aie.objectfifo @relay_in (%shim_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<512xi8>>

    // MemTile → compute tile 0 (first 256 bytes)
    aie.objectfifo @relay_dst0 (%mem_tile_2_1, {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<256xi8>>

    // MemTile → compute tile 1 (second 256 bytes)
    aie.objectfifo @relay_dst1 (%mem_tile_2_1, {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<256xi8>>

    // Distribute link: split 512B buffer into two 256B slices
    aie.objectfifo.link [@relay_in] -> [@relay_dst0, @relay_dst1] ([][0, 256])
  }
}
