// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests repeat_count=2 on a distribute link source fifo.
// Adapted from objectFifo-stateful-transform/repeat_count/link_distribute_repeat_count_test.mlir
// with repeat_count lowered from 3 to 2.
//
// Expected:
// - MemTile S2MM ingest: 4 BD blocks (of0.depth=2 × numDsts=2)
// - MemTile MM2S per destination: 4 BD blocks (of0.depth=2 × of1.repeat_count=2),
//   each source buffer repeated twice consecutively
// - Per-slice MemTile lock init = linkDepth * repeat_count = 2 * 2 = 4
// - Consumer tile lock init = depth * repeat_count = 2 * 2 = 4

// CHECK-LABEL: module @linkDistRepeat
// CHECK:   aie.device(npu1) {
// MemTile producer locks: init = repeat_count * depth = 2 * 2 = 4
// CHECK:     aie.lock({{.*}}) {init = 4 : i32, sym_name = "of2_prod_lock_0"}
// CHECK:     aie.lock({{.*}}) {init = 4 : i32, sym_name = "of1_prod_lock_0"}
// Consumer tile locks also init = 4
// CHECK:     aie.lock({{.*}}) {init = 4 : i32, sym_name = "of2_cons_prod_lock_0"}
// CHECK:     aie.lock({{.*}}) {init = 4 : i32, sym_name = "of1_cons_prod_lock_0"}
// Flows from shim and memtile
// CHECK:     aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// MemTile S2MM: 4 BD blocks (2 buffers x 2 distribute legs)
// CHECK:     aie.memtile_dma
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd ^bb1
// MemTile MM2S channel 0 (of1): 4 BD blocks (linkDepth=2 × repeat_count=2)
// Each source buffer is sent repeat_count=2 times consecutively.
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// Circular: loops back to first BD
// CHECK:       aie.next_bd ^bb6
// MemTile MM2S channel 1 (of2): 4 BD blocks (linkDepth=2 × repeat_count=2)
// CHECK:       aie.dma_start(MM2S, 1
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @linkDistRepeat {
 aie.device(npu1) {
    %tile10 = aie.tile(1, 0)
    %tile11 = aie.tile(1, 1)
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of0 (%tile10, {%tile11}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @of1 (%tile11, {%tile12}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2 (%tile11, {%tile33}, 2 : i32) {repeat_count = 2 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of0] -> [@of1, @of2] ([] [0, 16])
 }
}
