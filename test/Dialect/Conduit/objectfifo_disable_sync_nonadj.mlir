// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests disable_synchronization=true on non-adjacent compute→compute tiles
// (tile(0,2) → tile(2,3) on xcve2302).  No locks or use_lock ops should
// appear; DMA flow and BD chains must still be emitted.

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_buff_0"}
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_cons_buff_0"}
// CHECK:     aie.flow
// Producer MM2S BD
// CHECK:     aie.mem({{.*}}) {
// CHECK:       aie.dma_start(MM2S
// CHECK:       aie.dma_bd
// Consumer S2MM BD
// CHECK:     aie.mem({{.*}}) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd
// No locks anywhere (disable_synchronization removes all sync).
// CHECK-NOT: aie.lock
// CHECK-NOT: aie.use_lock
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(xcve2302) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_2_3 = aie.tile(2, 3)

    aie.objectfifo @of(%tile_0_2, {%tile_2_3}, 1 : i32) { disable_synchronization = true }
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
