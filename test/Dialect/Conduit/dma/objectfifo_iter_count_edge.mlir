// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests iter_count=1 edge case with depth=2 on MemTile → compute.
//
// Expected: consumer DMA has dma_start with NO repeat_count attribute
// (iter_count-1=0 is the default, so the attribute is omitted).

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// 2 consumer buffers
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_cons_buff_0"}
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_cons_buff_1"}
// Consumer lock init = depth = 2
// CHECK:     aie.lock({{.*}}) {init = 2 : i32, sym_name = "of_cons_prod_lock_0"}
// CHECK:     aie.lock({{.*}}) {init = 0 : i32, sym_name = "of_cons_cons_lock_0"}
// Consumer DMA: NO repeat_count (iter_count-1=0, default)
// CHECK:     aie.mem
// CHECK:       aie.dma_start(S2MM, 0, {{.*}}, {{.*}})
// CHECK-NOT:   repeat_count
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

    aie.objectfifo @of(%tile_0_1, {%tile_0_2}, 2 : i32) {iter_count = 1 : i32}
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    }
  }
}
