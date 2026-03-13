// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: 1 producer -> 3 consumer tiles (broadcast).
//
// Verifies that each consumer tile gets its own independent buffer and lock pair,
// that each core uses its own tile's resources (not consumer_tile[0]'s), and
// that N flows are generated (one per consumer) — Fix NF6.
//
// Resources:
//   aie.buffer:  3  (one per consumer tile, depth=1)
//   aie.lock:    8  (prod_lock + cons_lock per consumer tile × 3 consumers;
//                    plus prod_lock + cons_lock on shim tile)
//   aie.flow:    3  (shim DMA:0 → tile_0_2, shim DMA:1 → tile_0_3,
//                    shim DMA:2 → tile_0_4)
//   aie.mem:     3  (S2MM for each of the 3 consumer tiles)

// CHECK-LABEL: module @broadcast_3_consumers
// CHECK:   aie.device(npu1_1col) {
// --- All three consumer tiles declared ---
// CHECK:     aie.tile(0, 0)
// CHECK:     aie.tile(0, 2)
// CHECK:     aie.tile(0, 3)
// CHECK:     aie.tile(0, 4)
// --- Each consumer tile has its own buffer and lock pair ---
// CHECK:     aie.buffer(%{{.*}}tile_0_4)
// CHECK-SAME:   sym_name = "bcast_fifo_cons_2_buff_0"
// CHECK:     aie.lock(%{{.*}}tile_0_4
// CHECK-SAME:   sym_name = "bcast_fifo_cons_2_prod_lock_0"
// CHECK:     aie.lock(%{{.*}}tile_0_4
// CHECK-SAME:   sym_name = "bcast_fifo_cons_2_cons_lock_0"
// CHECK:     aie.buffer(%{{.*}}tile_0_3)
// CHECK-SAME:   sym_name = "bcast_fifo_cons_1_buff_0"
// CHECK:     aie.lock(%{{.*}}tile_0_3
// CHECK-SAME:   sym_name = "bcast_fifo_cons_1_prod_lock_0"
// CHECK:     aie.lock(%{{.*}}tile_0_3
// CHECK-SAME:   sym_name = "bcast_fifo_cons_1_cons_lock_0"
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "bcast_fifo_cons_0_buff_0"
// CHECK:     aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   sym_name = "bcast_fifo_cons_0_prod_lock_0"
// CHECK:     aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   sym_name = "bcast_fifo_cons_0_cons_lock_0"
// --- Each core uses its own tile's buffer and locks ---
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       aie.use_lock(%[[C0_CONS:bcast_fifo_cons_0_cons.*]], AcquireGreaterEqual, 1)
// CHECK:       func.call @consume_data(%{{.*}}bcast_fifo_cons_0_buff_0
// CHECK:       aie.use_lock(%[[C0_PROD:bcast_fifo_cons_0_prod.*]], Release, 1)
// CHECK:     aie.core(%{{.*}}tile_0_3) {
// CHECK:       aie.use_lock(%{{.*}}bcast_fifo_cons_1_cons{{.*}}, AcquireGreaterEqual, 1)
// CHECK:       func.call @consume_data(%{{.*}}bcast_fifo_cons_1_buff_0
// CHECK:       aie.use_lock(%{{.*}}bcast_fifo_cons_1_prod{{.*}}, Release, 1)
// CHECK:     aie.core(%{{.*}}tile_0_4) {
// CHECK:       aie.use_lock(%{{.*}}bcast_fifo_cons_2_cons{{.*}}, AcquireGreaterEqual, 1)
// CHECK:       func.call @consume_data(%{{.*}}bcast_fifo_cons_2_buff_0
// CHECK:       aie.use_lock(%{{.*}}bcast_fifo_cons_2_prod{{.*}}, Release, 1)
// --- Three flows: one per consumer tile (Fix NF6) ---
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 1, %{{.*}}tile_0_3, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 2, %{{.*}}tile_0_4, DMA : 0)
// --- Three aie.mem blocks: one S2MM for each consumer tile ---
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.use_lock(%{{.*}}bcast_fifo_cons_0_prod{{.*}}, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%{{.*}}bcast_fifo_cons_0_buff_0
// CHECK:       aie.use_lock(%{{.*}}bcast_fifo_cons_0_cons{{.*}}, Release, 1)
// CHECK:     aie.mem(%{{.*}}tile_0_3) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.use_lock(%{{.*}}bcast_fifo_cons_1_prod{{.*}}, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%{{.*}}bcast_fifo_cons_1_buff_0
// CHECK:       aie.use_lock(%{{.*}}bcast_fifo_cons_1_cons{{.*}}, Release, 1)
// CHECK:     aie.mem(%{{.*}}tile_0_4) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.use_lock(%{{.*}}bcast_fifo_cons_2_prod{{.*}}, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%{{.*}}bcast_fifo_cons_2_buff_0
// CHECK:       aie.use_lock(%{{.*}}bcast_fifo_cons_2_cons{{.*}}, Release, 1)
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @broadcast_3_consumers {
  aie.device(npu1_1col) {
    func.func @consume_data(%buf: memref<16xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    // 1 producer (shim tile 0,0), 3 consumers: broadcast semantics
    aie.objectfifo @bcast_fifo(%tile_0_0, {%tile_0_2, %tile_0_3, %tile_0_4}, 1 : i32)
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @bcast_fifo(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @consume_data(%1) : (memref<16xi32>) -> ()
        aie.objectfifo.release @bcast_fifo(Consume, 1)
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @bcast_fifo(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @consume_data(%1) : (memref<16xi32>) -> ()
        aie.objectfifo.release @bcast_fifo(Consume, 1)
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @bcast_fifo(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @consume_data(%1) : (memref<16xi32>) -> ()
        aie.objectfifo.release @bcast_fifo(Consume, 1)
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
