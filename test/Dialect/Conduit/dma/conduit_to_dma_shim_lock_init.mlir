// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma --split-input-file %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma --split-input-file %s | FileCheck %s --check-prefix=CHECK-D1
//
// Regression test: shim producer lock init value must be 0.
//
// Background:
//   Shim-side locks (prod_lock and cons_lock on the shim tile) are programmed
//   by the host runtime via aiex.npu.dma_memcpy_nd token signaling.  The AIE
//   runtime handles lock initialization as part of DMA configuration, so both
//   shim locks must start at 0.
//
//   Pre-signaling depth free slots to a shim DMA that has not yet been
//   configured causes over-commitment.
//
// Topology: depth-2 shim-to-compute (shim tile [0,0] → compute tile [0,2]).
// Target: npu1_1col (AIE2).
//
// Expected locks:
//   Shim tile [0,0]:
//     prod_lock init=0  (host runtime programs shim locks via npu.dma_memcpy_nd)
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

// --- Shim-tile producer lock: init=0 (oracle match: host programs shim locks) ---
// CHECK:     aie.lock(%{{.*}}tile_0_0
// CHECK-SAME:   init = 0
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
    // depth=2: double-buffering; shim producer lock must be init=0 (oracle match)
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

// -----
// CHECK-D1-LABEL: module @shim_lock_init_depth1
// CHECK-D1:   aie.device(npu1_1col) {
// CHECK-D1:     aie.lock(%{{.*}}tile_0_2
// CHECK-D1-SAME:   init = 1
// CHECK-D1-SAME:   sym_name = "shim_d1_fifo_cons_prod_lock_0"
// CHECK-D1:     aie.lock(%{{.*}}tile_0_2
// CHECK-D1-SAME:   init = 0
// CHECK-D1-SAME:   sym_name = "shim_d1_fifo_cons_cons_lock_0"
// CHECK-D1:     aie.shim_dma_allocation @{{.*}}shim_alloc
// CHECK-D1:     aie.lock(%{{.*}}tile_0_0
// CHECK-D1-SAME:   init = 0
// CHECK-D1-SAME:   sym_name = "shim_d1_fifo_prod_lock_0"
// CHECK-D1:     aie.lock(%{{.*}}tile_0_0
// CHECK-D1-SAME:   init = 0
// CHECK-D1-SAME:   sym_name = "shim_d1_fifo_cons_lock_0"
// CHECK-D1:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
// Depth=1 shim-to-compute: verifies shim locks init=0 (no ping-pong buffers).
module @shim_lock_init_depth1 {
  aie.device(npu1_1col) {
    func.func @process_data_d1(%buf: memref<8xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    // depth=1: single-buffer; shim locks must be init=0
    aie.objectfifo @shim_d1_fifo(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<8xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @shim_d1_fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @process_data_d1(%1) : (memref<8xi32>) -> ()
        aie.objectfifo.release @shim_d1_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
