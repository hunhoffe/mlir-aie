// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
// XFAIL: *
//
// Pass A + Pass C end-to-end test: 1 producer -> 3 consumer tiles (broadcast).
//
// Known gap: Pass C currently only allocates buffers/locks for consumerTiles[0]
// and skips consumers 1 and 2.  Full broadcast requires three independent sets
// of buffers + lock pairs, one per consumer tile.  See CLAUDE.md "Known gaps
// in Pass C: Multi-consumer broadcast".
//
// This test defines the EXPECTED output once the broadcast fix lands.
//
// Expected resource counts:
//   aie.buffer:  3  (one per consumer tile, depth=1)
//   aie.lock:    6  (prod_lock + cons_lock per consumer tile × 3 consumers)
//   aie.flow:    3  (shim → tile_0_2, tile_0_3, tile_0_4)
//
// Naming convention (matching stateful-transform):
//   tile_0_2: bcast_fifo_cons_0_buff_0, bcast_fifo_cons_0_prod_lock_0, ..._cons_lock_0
//   tile_0_3: bcast_fifo_cons_1_buff_0, bcast_fifo_cons_1_prod_lock_0, ..._cons_lock_0
//   tile_0_4: bcast_fifo_cons_2_buff_0, bcast_fifo_cons_2_prod_lock_0, ..._cons_lock_0

// CHECK-LABEL: module @broadcast_3_consumers
// CHECK:   aie.device(npu1_1col) {
// CHECK:     %{{.*}}tile_0_0 = aie.tile(0, 0)
// CHECK:     %{{.*}}tile_0_2 = aie.tile(0, 2)
// CHECK:     %{{.*}}tile_0_3 = aie.tile(0, 3)
// CHECK:     %{{.*}}tile_0_4 = aie.tile(0, 4)
// --- Consumer 0 (tile_0_2) ---
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "bcast_fifo_cons_0_buff_0"
// CHECK:     aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   sym_name = "bcast_fifo_cons_0_prod_lock_0"
// CHECK:     aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   sym_name = "bcast_fifo_cons_0_cons_lock_0"
// --- Consumer 1 (tile_0_3) ---
// CHECK:     aie.buffer(%{{.*}}tile_0_3)
// CHECK-SAME:   sym_name = "bcast_fifo_cons_1_buff_0"
// CHECK:     aie.lock(%{{.*}}tile_0_3
// CHECK-SAME:   sym_name = "bcast_fifo_cons_1_prod_lock_0"
// CHECK:     aie.lock(%{{.*}}tile_0_3
// CHECK-SAME:   sym_name = "bcast_fifo_cons_1_cons_lock_0"
// --- Consumer 2 (tile_0_4) ---
// CHECK:     aie.buffer(%{{.*}}tile_0_4)
// CHECK-SAME:   sym_name = "bcast_fifo_cons_2_buff_0"
// CHECK:     aie.lock(%{{.*}}tile_0_4
// CHECK-SAME:   sym_name = "bcast_fifo_cons_2_prod_lock_0"
// CHECK:     aie.lock(%{{.*}}tile_0_4
// CHECK-SAME:   sym_name = "bcast_fifo_cons_2_cons_lock_0"
// --- Three flows from shim to each consumer ---
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : {{.*}}, %{{.*}}tile_0_3, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : {{.*}}, %{{.*}}tile_0_4, DMA : 0)
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
