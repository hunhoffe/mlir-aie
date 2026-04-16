// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests iter_count=K: DMAStartOp.repeat_count should be K-1, and the BD chain
// loops back (last BD → first BD). repeat_count controls termination.
//
// For iter_count = 5 on a depth-2 shim→compute fifo, DMAStartOp repeat_count = 4.

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// CHECK:     aie.mem
// CHECK:       aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 4)
// BD chain is circular: last BD loops back to ^bb1 (repeat_count controls termination)
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd ^bb2
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd ^bb1

module {
  aie.device(xcve2302) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @in(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) {iter_count = 5 : i32}
        : !aie.objectfifo<memref<1024xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
      aie.objectfifo.release @in(Consume, 1)
      aie.end
    }
  }
}
