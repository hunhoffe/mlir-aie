// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests disable_synchronization=true with depth=2 on compute→MemTile.
// No aie.lock or aie.use_lock should appear.  DMA BD chains must still be
// emitted with 2 BD blocks per direction (one per buffer).

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// 4 buffers: 2 producer-side + 2 consumer-side (depth=2)
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_buff_0"}
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_buff_1"}
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_cons_buff_0"}
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_cons_buff_1"}
// CHECK:     aie.flow
// Producer MM2S: 2 BD blocks
// CHECK:     aie.mem
// CHECK:       aie.dma_start(MM2S
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd ^bb1
// Consumer S2MM: 2 BD blocks
// CHECK:     aie.memtile_dma
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd ^bb1
// No locks anywhere (disable_synchronization removes all sync).
// CHECK-NOT: aie.lock
// CHECK-NOT: aie.use_lock
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of(%tile_0_2, {%tile_0_1}, 2 : i32) { disable_synchronization = true }
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Produce, 1)
      aie.end
    }
  }
}
