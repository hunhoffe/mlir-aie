// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests via_DMA=true: even for adjacent tiles (that would normally use shared
// memory), a DMA flow must be emitted.
//
// Tiles (0,2) and (0,3) are adjacent. Without via_DMA, Conduit would use
// shared memory (no aie.flow). With via_DMA=true, an aie.flow must appear.

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// CHECK:     aie.flow
// CHECK:     aie.mem

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @of(%tile_0_2, {%tile_0_3}, 1 : i32) {via_DMA = true}
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Produce, 1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %0 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    }
  }
}
