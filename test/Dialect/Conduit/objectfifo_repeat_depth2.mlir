// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests repeat_count=2 with depth=2 on non-adjacent tiles.
//
// Expected: 4 BD blocks (2 buffers × 2 repeats), circular chain.
// Producer lock init = depth * repeat_count = 2 * 2 = 4.
// Core acquires/releases 2 units (repeat_count).

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// 2 buffers allocated
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_buff_0"}
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_buff_1"}
// Producer lock init = depth * repeat_count = 2 * 2 = 4
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock({{.*}}) {init = 4 : i32
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock({{.*}}) {init = 0 : i32
// Core acquires repeat_count=2 units
// CHECK:     aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 2)
// Core releases repeat_count=2 units
// CHECK:     aie.use_lock(%[[CONS_LOCK]], Release, 2)
// CHECK:     aie.flow
// 4 BD blocks (2 buffers × 2 repeats)
// CHECK:     aie.mem
// CHECK:       aie.dma_start(MM2S
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
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
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of(%tile_0_2, {%tile_0_0}, 2 : i32) {repeat_count = 2 : i32}
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Produce, 1)
      aie.end
    }
  }
}
