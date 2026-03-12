// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Pass A test: single depth-1 objectfifo with acquire/release in a core loop.
// Source: adapted from test/objectFifo-stateful-transform/dynamic_lowering/
//         depth_one_objectfifo_test.mlir
//
// After --objectfifo-to-conduit the module should contain:
//   - conduit.create for the input_fifo
//   - conduit.annotate for producer_tile, consumer_tile_0, element_type, depth
//   - conduit.acquire inside the core loop body
//   - conduit.release inside the core loop body
//   - NO aie.objectfifo (all lifted)

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// CHECK:     conduit.create
// CHECK-SAME:   name = "input_fifo"
// CHECK:     conduit.annotate
// CHECK-SAME:   key = "producer_tile"
// CHECK-SAME:   value = "tile(0,0)"
// CHECK:     conduit.annotate
// CHECK-SAME:   key = "consumer_tile_0"
// CHECK-SAME:   value = "tile(0,2)"
// CHECK:     conduit.annotate
// CHECK-SAME:   key = "element_type"
// CHECK:     conduit.annotate
// CHECK-SAME:   key = "depth"
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       scf.for
// CHECK:         conduit.acquire
// CHECK-SAME:       name = "input_fifo"
// CHECK:         conduit.release
// CHECK-SAME:       name = "input_fifo"
// CHECK:     }
// CHECK-NOT: aie.objectfifo

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
