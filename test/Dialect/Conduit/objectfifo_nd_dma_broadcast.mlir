// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests dimensionsFromStream with 2 broadcast consumers, each with different
// dims.  Consumer 0 (tile_2_3) gets [<size=8, stride=1>]; consumer 1 (tile_3_3)
// gets [<size=4, stride=2>].  The producer MM2S BD should carry NO dims.

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// Two flows (one per consumer)
// CHECK:     aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// CHECK:     aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// Producer MM2S BD: no dims (just buffer, offset, length)
// CHECK:     aie.mem({{.*}}) {
// CHECK:       aie.dma_start(MM2S
// CHECK:       aie.dma_bd(%{{.*}} : memref<16xi32>, 0, 16)
// Consumer 0 S2MM BD: carries [<size = 8, stride = 1>]
// CHECK:     aie.mem({{.*}}) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(%{{.*}} : memref<16xi32>, 0, 16, [<size = 8, stride = 1>])
// Consumer 1 S2MM BD: carries [<size = 4, stride = 2>]
// CHECK:     aie.mem({{.*}}) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(%{{.*}} : memref<16xi32>, 0, 16, [<size = 4, stride = 2>])
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(xcve2302) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)

    aie.objectfifo @of (%tile_0_2,
                        {%tile_2_3 dimensionsFromStream [<size = 8, stride = 1>],
                         %tile_3_3 dimensionsFromStream [<size = 4, stride = 2>]},
                        1 : i32)
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Produce, 1)
      aie.end
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %0 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %0 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    }
  }
}
