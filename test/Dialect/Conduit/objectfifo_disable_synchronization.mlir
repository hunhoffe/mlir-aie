// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests disable_synchronization=true: no aie.lock or aie.use_lock should
// appear in the lowered output.  DMA BD chains must still be emitted for
// the producer MM2S and consumer S2MM paths.

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// CHECK-NOT: aie.lock
// CHECK-NOT: aie.use_lock
// CHECK:     aie.flow
// CHECK:     aie.mem
// CHECK:       aie.dma_start
// CHECK:       aie.dma_bd
// CHECK:     aie.memtile_dma
// CHECK:       aie.dma_start
// CHECK:       aie.dma_bd

module {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of(%tile_0_2, {%tile_0_1}, 2 : i32) { disable_synchronization = true }
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Produce, 1)
      aie.end
    }
  }
}
