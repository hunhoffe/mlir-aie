// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests repeat_count=3 + iter_count=5 combined on a MemTile producer.
//
// Expected: DMAStartOp repeat_count = iter_count-1 = 4.
// BD chain has 3 blocks (one per repeat), last BD → end block (non-circular).
// Producer lock init = depth * repeat_count = 1 * 3 = 3.

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// Producer lock init = 3
// CHECK:     aie.lock({{.*}}) {init = 3 : i32
// Consumer lock init = 3
// CHECK:     aie.lock({{.*}}) {init = 3 : i32
// CHECK:     aie.flow
// Producer MemTile DMA: repeat_count = iter_count - 1 = 4
// CHECK:     aie.memtile_dma
// CHECK:       aie.dma_start(MM2S, 0, {{.*}}, {{.*}}, repeat_count = 4)
// 3 BD blocks (one per repeat_count)
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// Last BD goes to end block (non-circular)
// CHECK:       aie.next_bd ^bb4
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
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @of(%mem_tile_0_1, {%tile_0_3}, 1 : i32) {repeat_count = 3 : i32, iter_count = 5 : i32}
        : !aie.objectfifo<memref<16xi32>>

    %core_0_3 = aie.core(%tile_0_3) {
      %0 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    }
  }
}
