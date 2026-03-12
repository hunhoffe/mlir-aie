// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: depth-1 single-consumer objectfifo.
// Input: same as objectfifo_to_conduit_depth_one.mlir
//
// After --objectfifo-to-conduit --conduit-to-dma the module should contain
// the hardware ops that the existing --aie-objectFifo-stateful-transform
// produces for the same input.  We check for structural presence, not exact
// SSA names (those vary with lock ID assignment order).
//
// Resource comparison vs --aie-objectFifo-stateful-transform:
//   aie.buffer:  1  (depth × 1)
//   aie.lock:    2  (prod + cons)
//   aie.flow:    1  (shim tile 0,0 → tile 0,2)
//   aie.use_lock in core body: 2 per loop iteration (acquire + release)
//
// The CHECK lines below verify structural presence; exact counts are checked
// by compare_lowering_resources.py.

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// CHECK:     %{{.*}}tile_0_0 = aie.tile(0, 0)
// CHECK:     %{{.*}}tile_0_2 = aie.tile(0, 2)
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "input_fifo_buff_0"
// CHECK:     aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   sym_name = "input_fifo_prod_lock_0"
// CHECK:     aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   sym_name = "input_fifo_cons_lock_0"
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       scf.for
// CHECK:         aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock({{.*}}, Release, 1)
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @{{.*}}shim_alloc
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(npu1_1col) {
    func.func @passthrough_10_i32(%line_in: memref<10xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %0 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @passthrough_10_i32(%1) : (memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
