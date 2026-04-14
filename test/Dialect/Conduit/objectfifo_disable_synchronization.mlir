// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests disable_synchronization=true on a non-adjacent DMA path: no aie.lock
// or aie.use_lock should appear.  DMA BD chains must still be emitted for
// the producer MM2S and consumer S2MM paths.

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// CHECK:     aie.flow
// CHECK:     aie.mem
// CHECK:       aie.dma_start
// CHECK:       aie.dma_bd
// CHECK:     aie.mem
// CHECK:       aie.dma_start
// CHECK:       aie.dma_bd
// No locks anywhere in the output (disable_synchronization removes all sync).
// CHECK-NOT: aie.lock
// CHECK-NOT: aie.use_lock

module {
  aie.device(xcve2302) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_3 = aie.tile(1, 3)

    aie.objectfifo @of(%tile_0_2, {%tile_1_3}, 2 : i32) { disable_synchronization = true }
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Produce, 1)
      aie.end
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %0 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    }
  }
}
