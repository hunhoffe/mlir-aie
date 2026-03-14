// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// P1-G regression test: shim producer lock init value must equal depth.
//
// Background (Fix 1a):
//   Before Fix 1a, shim-side producer locks were initialized with init=0.
//   This is correct for depth-1 (trivially: 1 slot, init=1 would also work
//   but the stateful transform uses init=0 for shim depth-1), but for depth>1
//   it causes a silent hardware deadlock:
//
//     - The shim DMA producer needs to acquire a "free slot" lock before it
//       can write data into a buffer slot.
//     - On AIE2 (semaphore model), the producer lock counts free buffer slots.
//       init=0 means "no slots free" → the DMA engine immediately stalls,
//       waiting for a slot that the consumer has never filled (because the
//       consumer is also waiting for data).
//     - Result: mutual deadlock — neither side can make progress.
//
//   Fix 1a changed the shim producer lock to init=depth (all slots initially
//   free, matching the consumer-tile prod_lock convention).
//
// This test guards against regression: if anyone changes the init value back
// to 0, this test will fail.
//
// Topology: depth-2 shim-to-compute (shim tile [0,0] → compute tile [0,2]).
// Target: npu1_1col (AIE2).
//
// Expected locks:
//   Shim tile [0,0]:
//     prod_lock init=2  (depth=2 → 2 free slots, Fix 1a)
//     cons_lock init=0  (no filled slots initially)
//   Compute tile [0,2]:
//     cons_prod_lock init=2  (depth=2 → 2 free slots)
//     cons_cons_lock init=0  (no filled slots initially)

// CHECK-LABEL: module @shim_lock_init_test
// CHECK:   aie.device(npu1_1col) {

// --- Consumer-tile locks: prod_lock init=2, cons_lock init=0 ---
// CHECK:     aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 2
// CHECK-SAME:   sym_name = "shim_fifo_cons_prod_lock_0"
// CHECK:     aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "shim_fifo_cons_cons_lock_0"

// --- Shim DMA allocation ---
// CHECK:     aie.shim_dma_allocation @{{.*}}shim_alloc

// --- Shim-tile producer lock: init=2 (Fix 1a: must equal depth, NOT 0) ---
// CHECK:     aie.lock(%{{.*}}tile_0_0
// CHECK-SAME:   init = 2
// CHECK-SAME:   sym_name = "shim_fifo_prod_lock_0"

// --- Shim-tile consumer lock: init=0 ---
// CHECK:     aie.lock(%{{.*}}tile_0_0
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "shim_fifo_cons_lock_0"

// --- Flow from shim to compute tile ---
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @shim_lock_init_test {
  aie.device(npu1_1col) {
    func.func @process_data(%buf: memref<8xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    // depth=2: double-buffering; shim producer lock must be init=depth=2
    aie.objectfifo @shim_fifo(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @shim_fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @process_data(%1) : (memref<8xi32>) -> ()
        aie.objectfifo.release @shim_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
