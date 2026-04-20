// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests all features combined: dimensionsToStream + repeat_count=2 +
// iter_count=3 + link on shim(0,0) → MemTile(0,1) → compute(0,2).
//
// Expected:
// - MemTile link lock init = depth × repeat_count = 2 × 2 = 4
// - MemTile MM2S: 4 BD blocks, each with dimensionsToStream layout
// - Consumer dma_start: repeat_count = iter_count - 1 = 2

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// MemTile link lock init = depth * repeat_count = 4
// CHECK:     aie.lock({{.*}}) {init = 4 : i32, sym_name = "of_in_link_prod_lock_0"}
// CHECK:     aie.lock({{.*}}) {init = 0 : i32, sym_name = "of_in_link_cons_lock_0"}
// MemTile DMA
// CHECK:     aie.memtile_dma
// MemTile S2MM ingest (circular)
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd ^bb1
// MemTile MM2S: 4 BD blocks with dimensionsToStream
// CHECK:       aie.dma_start(MM2S
// dimensionsToStream on each BD
// CHECK:       aie.dma_bd({{.*}}, 0, 16, [<size = 4, stride = 4>, <size = 4, stride = 1>])
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd({{.*}}, 0, 16, [<size = 4, stride = 4>, <size = 4, stride = 1>])
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd({{.*}}, 0, 16, [<size = 4, stride = 4>, <size = 4, stride = 1>])
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd({{.*}}, 0, 16, [<size = 4, stride = 4>, <size = 4, stride = 1>])
// Circular: last BD loops back
// CHECK:       aie.next_bd ^bb4
// Consumer DMA: repeat_count = iter_count - 1 = 2
// CHECK:     aie.mem
// CHECK:       aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 2)
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// Circular: last BD loops back
// CHECK:       aie.next_bd ^bb1
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_1}, 2 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_1 dimensionsToStream [<size = 4, stride = 4>, <size = 4, stride = 1>],
                           {%tile_0_2}, 2 : i32) {repeat_count = 2 : i32, iter_count = 3 : i32}
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of_in] -> [@of_out]([] [0])

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of_out(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of_out(Consume, 1)
      aie.end
    }
  }
}
