// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests disable_synchronization=true with depth=2 on non-adjacent tiles
// (DMA path — tile(0,2)→tile(2,3) on xcve2302).  Core bodies on both
// tiles exercise acquire/release lowering.
//
// No aie.lock or aie.use_lock should appear.  DMA flow and BD chains must
// be emitted.  Each aie.mem should have 2 BD blocks (one per buffer) with
// no use_lock ops.  Consumer core should have rotation counter arith ops.

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// CHECK-NOT: aie.lock
// CHECK-NOT: aie.use_lock
// Producer buffers
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_buff_0"} : memref<16xi32>
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_buff_1"} : memref<16xi32>
// Consumer buffers
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_cons_buff_0"} : memref<16xi32>
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_cons_buff_1"} : memref<16xi32>
// Consumer core: rotation counter management
// CHECK:     aie.core(%{{.*}}) {
// CHECK:       arith.constant 0 : i32
// CHECK:       memref.store
// CHECK:       memref.load
// CHECK:       arith.addi
// CHECK:       arith.remui
// CHECK:       memref.store
// CHECK:       aie.end
// DMA flow
// CHECK:     aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// Producer MM2S: 2 BD blocks, no use_lock
// CHECK:     aie.mem(%{{.*}}) {
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_bd(%{{.*}} : memref<16xi32>, 0, 16)
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd(%{{.*}} : memref<16xi32>, 0, 16)
// CHECK:       aie.next_bd ^bb1
// Consumer S2MM: 2 BD blocks, no use_lock
// CHECK:     aie.mem(%{{.*}}) {
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.dma_bd(%{{.*}} : memref<16xi32>, 0, 16)
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd(%{{.*}} : memref<16xi32>, 0, 16)
// CHECK:       aie.next_bd ^bb1
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(xcve2302) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_2_3 = aie.tile(2, 3)

    aie.objectfifo @of(%tile_0_2, {%tile_2_3}, 2 : i32) { disable_synchronization = true }
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Produce, 1)
      aie.end
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %0 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    }
  }
}
