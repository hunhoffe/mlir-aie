// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests dimensionsToStream: the producer MM2S BD should carry the
// BDDimLayout attribute from the objectfifo's dimensionsToStream.
//
// Uses non-adjacent tiles to force DMA path.

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// Producer tile MM2S BD must carry the BDDimLayout dimensions.
// CHECK:     aie.mem({{.*}}) {
// CHECK:       aie.dma_start(MM2S
// CHECK:       aie.dma_bd({{.*}} [<size = 16, stride = 1>

module {
  aie.device(xcve2302) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_2_3 = aie.tile(2, 3)

    aie.objectfifo @of (%tile_0_2 dimensionsToStream [<size = 16, stride = 1>],
                        {%tile_2_3}, 1 : i32)
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
  }
}
