// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests dimensionsFromStream on the consumer side only.
// The consumer S2MM BD should carry [<size = 8, stride = 1>];
// the producer MM2S BD should have NO dims.
//
// Uses non-adjacent tiles to force DMA path.

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// CHECK:     aie.flow
// Producer MM2S BD: no dims (just buffer, offset, length)
// CHECK:     aie.mem({{.*}}) {
// CHECK:       aie.dma_start(MM2S
// CHECK:       aie.dma_bd(%{{.*}} : memref<16xi32>, 0, 16)
// Consumer S2MM BD: carries the dimensionsFromStream dims
// CHECK:     aie.mem({{.*}}) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(%{{.*}} : memref<16xi32>, 0, 16, [<size = 8, stride = 1>])
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(xcve2302) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_2_3 = aie.tile(2, 3)

    aie.objectfifo @of (%tile_0_2,
                        {%tile_2_3 dimensionsFromStream [<size = 8, stride = 1>]},
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
  }
}
