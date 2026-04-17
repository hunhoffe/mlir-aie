// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// MemTile distribute to same-column compute tiles (pathfinder-safe pattern).
//
// Topology:
//   shim(2,0) → MemTile(2,1) → {tile(2,2), tile(2,3)}  [same column 2]

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
// CHECK-NOT: conduit.scatter

module @memsame_column_relay {
  aie.device(xcve2302) {
    %shim_2_0 = aie.tile(2, 0)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)

    // Ingress: shim → MemTile
    conduit.create @relay_in {element_type = memref<512xi8>, depth = 2 : i64}
    // Egress: MemTile → compute tiles
    conduit.create @relay_dst0 {element_type = memref<256xi8>, depth = 2 : i64}
    conduit.create @relay_dst1 {element_type = memref<256xi8>, depth = 2 : i64}

    // Distribute link: split 512B buffer into two 256B slices at MemTile(2,1)
    conduit.scatter{src = @relay_in, dsts = [@relay_dst0, @relay_dst1] {memtile = "tile(2,1)", offsets = array<i64: 0, 256>}}

    // Shim producer allocation for ingress channel.
    aie.shim_dma_allocation @relay_in_shim_alloc(%shim_2_0, MM2S, 0) {conduit_channel = @relay_in}

    // Consumer cores — structural info for tile inference.
    %core_2_2 = aie.core(%tile_2_2) {
      %0 = conduit.acquire {count = 1 : i64, name = @relay_dst0,
                            port = #conduit.port<Consume>} : <memref<256xi8>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Consume>} : <memref<256xi8>>
      aie.end
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %0 = conduit.acquire {count = 1 : i64, name = @relay_dst1,
                            port = #conduit.port<Consume>} : <memref<256xi8>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Consume>} : <memref<256xi8>>
      aie.end
    }
  }
}
