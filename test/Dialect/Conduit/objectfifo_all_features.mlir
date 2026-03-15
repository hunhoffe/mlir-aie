// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests repeat_count=2 + iter_count=3 + dimensionsToStream combined.
//
// MemTile (0,1) producer with dimensionsToStream → compute tile (0,3) consumer.
// Expected:
// - Producer lock init = depth * repeat_count = 1 * 2 = 2
// - DMAStartOp repeat_count = iter_count - 1 = 2
// - 2 BD blocks with dims [<size = 4, stride = 1>]
// - Last BD → end block (non-circular, due to iter_count)

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// Producer lock init = 1 * 2 = 2
// CHECK:     aie.lock({{.*}}) {init = 2 : i32
// Consumer lock init = 2
// CHECK:     aie.lock({{.*}}) {init = 2 : i32
// CHECK:     aie.flow
// Producer MemTile DMA: repeat_count = iter_count - 1 = 2
// CHECK:     aie.memtile_dma
// CHECK:       aie.dma_start(MM2S, 0, {{.*}}, {{.*}}, repeat_count = 2)
// BD blocks carry dimensionsToStream dims
// CHECK:       aie.dma_bd({{.*}} [<size = 4, stride = 1>])
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd({{.*}} [<size = 4, stride = 1>])
// Last BD → end block (non-circular due to iter_count)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:
// CHECK:       aie.end
// Consumer DMA also gets repeat_count = 2
// CHECK:     aie.mem
// CHECK:       aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 2)
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(xcve2302) {
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @of (%mem_tile_0_1 dimensionsToStream [<size = 4, stride = 1>],
                        {%tile_0_3}, 1 : i32) {repeat_count = 2 : i32, iter_count = 3 : i32}
        : !aie.objectfifo<memref<16xi32>>

    %core_0_3 = aie.core(%tile_0_3) {
      %0 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    }
  }
}
