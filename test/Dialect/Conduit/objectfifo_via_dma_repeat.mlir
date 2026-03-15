// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests via_DMA=true combined with repeat_count=2 on adjacent tiles.
//
// Adjacent tiles (0,2) and (0,3) would normally use shared memory.
// via_DMA forces DMA path: aie.flow emitted, BD chains created.
// repeat_count=2 unrolls 2 BD blocks, producer lock init = depth*repeat = 1*2 = 2.
// Core acquires/releases 2 units (repeat_count).

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// Producer lock init = 1 * 2 = 2
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock({{.*}}) {init = 2 : i32
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock({{.*}}) {init = 0 : i32
// Consumer lock init = 2
// CHECK:     aie.lock({{.*}}) {init = 2 : i32
// Core acquires repeat_count=2 units
// CHECK:     aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 2)
// Core releases repeat_count=2 units
// CHECK:     aie.use_lock(%[[CONS_LOCK]], Release, 2)
// via_DMA forces flow emission despite adjacent tiles
// CHECK:     aie.flow
// 2 BD blocks (one per repeat)
// CHECK:     aie.mem
// CHECK:       aie.dma_start(MM2S
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// Circular: loops back to ^bb1
// CHECK:       aie.next_bd ^bb1
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @of(%tile_0_2, {%tile_0_3}, 1 : i32) {via_DMA = true, repeat_count = 2 : i32}
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Produce, 1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %0 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    }
  }
}
