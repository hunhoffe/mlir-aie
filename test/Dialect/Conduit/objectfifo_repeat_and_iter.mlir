// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests repeat_count=3 + iter_count=5 combined on a non-adjacent compute path.
//
// Expected: DMAStartOp repeat_count = iter_count-1 = 4.
// BD chain has 3 blocks (one per repeat), last BD loops back to first BD.
// Producer lock init = depth * repeat_count = 1 * 3 = 3.
// Consumer lock init = depth = 1.

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// Producer lock init = depth * repeat_count = 1 * 3 = 3
// CHECK:     aie.lock({{.*}}) {init = 3 : i32
// Consumer tile lock init = depth = 1 (repeat_count does not multiply here;
// the consumer FIFO has only depth slots, independent of repeat_count)
// CHECK:     aie.lock({{.*}}) {init = 1 : i32
// CHECK:     aie.flow
// Producer DMA: repeat_count = iter_count - 1 = 4
// CHECK:     aie.mem
// CHECK:       aie.dma_start(MM2S, 0, {{.*}}, {{.*}}, repeat_count = 4)
// 3 BD blocks (one per repeat_count)
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// Last BD loops back to first BD (repeat_count controls termination)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb4:
// CHECK:       aie.end
// Consumer DMA also gets repeat_count = 4
// CHECK:     aie.mem
// CHECK:       aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 4)
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(xcve2302) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_3 = aie.tile(1, 3)

    aie.objectfifo @of(%tile_0_2, {%tile_1_3}, 1 : i32) {repeat_count = 3 : i32, iter_count = 5 : i32}
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
