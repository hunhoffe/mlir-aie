// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests repeat_count=2 + iter_count=3 + dimensionsToStream combined.
//
// Compute tile (0,2) producer with dimensionsToStream → compute tile (1,3) consumer
// (different columns, non-adjacent → DMA path).
// Expected:
// - DMA repeat_count = iter_count - 1 = 2
// - Producer MM2S BD blocks carry dims [<size = 4, stride = 1>]
// - Last BD → end block (non-circular, due to iter_count)

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// Producer lock
// CHECK:     aie.lock({{.*}}) {init = 2 : i32
// Consumer tile lock init = depth = 1
// CHECK:     aie.lock({{.*}}) {init = 1 : i32
// CHECK:     aie.flow
// Producer DMA: repeat_count = iter_count - 1 = 2, dims from dimensionsToStream
// CHECK:     aie.mem
// CHECK:       aie.dma_start(MM2S, 0, {{.*}}, {{.*}}, repeat_count = 2)
// CHECK:       aie.dma_bd(%{{.*}} : memref<16xi32>, 0, 16, [<size = 4, stride = 1>])
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd(%{{.*}} : memref<16xi32>, 0, 16, [<size = 4, stride = 1>])
// Last BD → end block (non-circular due to iter_count)
// CHECK:       aie.next_bd ^bb{{[0-9]+}}
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
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_3 = aie.tile(1, 3)

    aie.objectfifo @of (%tile_0_2 dimensionsToStream [<size = 4, stride = 1>],
                        {%tile_1_3}, 1 : i32) {repeat_count = 2 : i32, iter_count = 3 : i32}
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
