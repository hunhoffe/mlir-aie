// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests repeat_count=1: should be identical to omitting repeat_count.
// Producer lock init = depth*1 = 1, single BD, circular (next_bd back to
// itself), core AcquireGreaterEqual 1.

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_buff_0"}
// Producer lock init = depth * repeat_count = 1 * 1 = 1
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock({{.*}}) {init = 1 : i32, sym_name = "of_prod_lock_0"}
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock({{.*}}) {init = 0 : i32, sym_name = "of_cons_lock_0"}
// Core acquires 1 (repeat_count=1)
// CHECK:     aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 1)
// Core releases 1
// CHECK:     aie.use_lock(%[[CONS_LOCK]], Release, 1)
// CHECK:     aie.flow
// Single BD, circular: loops back to itself
// CHECK:     aie.mem
// CHECK:       aie.dma_start(MM2S
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd ^bb1
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of(%tile_0_2, {%tile_0_0}, 1 : i32) {repeat_count = 1 : i32}
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Produce, 1)
      aie.end
    }
  }
}
