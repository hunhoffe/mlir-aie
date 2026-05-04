// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests a shim→MemTile→compute distribute pattern.
// NOTE: Sprint 6 gap — dimensionsToStream BD dims not yet threaded through
// the Conduit pipeline (tracked for a future pass update). This test verifies
// the basic structural lowering (flows, DMA starts, buffers).

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {

// Flows: shim→MemTile, MemTile→compute×2
// CHECK:     aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// CHECK:     aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// CHECK:     aie.flow(%{{.*}}, DMA : 1, %{{.*}}, DMA : 0)

// MemTile DMA: S2MM (link receive) + MM2S×2
// CHECK:     %{{.*}} = aie.memtile_dma(%{{.*}}) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_bd(%{{.*}}of0_cons_buff_0
// CHECK:       aie.dma_start(MM2S, 1
// CHECK:       aie.dma_bd(%{{.*}}of0_cons_buff_0

// Consumer 0 S2MM BD
// CHECK:     %{{.*}} = aie.mem(%{{.*}}) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(%{{.*}} : memref<128xi32>, 0, 128)

// Consumer 1 S2MM BD
// CHECK:     %{{.*}} = aie.mem(%{{.*}}) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(%{{.*}} : memref<128xi32>, 0, 128)

// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(xcve2302) {
    %tile_1_0 = aie.tile(1, 0)
    %tile_1_1 = aie.tile(1, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile_1_0, {%tile_1_1},
                         2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.objectfifo @of1 (%tile_1_1 dimensionsToStream [<size = 4, stride = 1>],
                        {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<128xi32>>

    aie.objectfifo @of2 (%tile_1_1 dimensionsToStream [<size = 4, stride = 1>],
                        {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<128xi32>>

    aie.objectfifo.link [ @of0 ] -> [ @of1, @of2 ] ([][0, 512])

    %core_2_2 = aie.core(%tile_2_2) {
      %sv = aie.objectfifo.acquire @of1(Consume, 1) : !aie.objectfifosubview<memref<128xi32>>
      aie.objectfifo.release @of1(Consume, 1)
      aie.end
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %sv = aie.objectfifo.acquire @of2(Consume, 1) : !aie.objectfifosubview<memref<128xi32>>
      aie.objectfifo.release @of2(Consume, 1)
      aie.end
    }
  }
}
