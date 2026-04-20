// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests iter_count=5 with depth=2 on MemTile → compute.
//
// Expected: consumer DMA has dma_start with repeat_count=4 (iter_count-1),
// circular BD chain with 2 BDs.

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// 2 consumer buffers
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_cons_buff_0"}
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_cons_buff_1"}
// Consumer lock init = depth = 2
// CHECK:     aie.lock({{.*}}) {init = 2 : i32, sym_name = "of_cons_prod_lock_0"}
// CHECK:     aie.lock({{.*}}) {init = 0 : i32, sym_name = "of_cons_cons_lock_0"}
// Consumer DMA: repeat_count = iter_count - 1 = 4
// CHECK:     aie.mem
// CHECK:       aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 4)
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
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of(%tile_0_1, {%tile_0_2}, 2 : i32) {iter_count = 5 : i32}
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    }
  }
}
