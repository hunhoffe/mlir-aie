// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Test: Pass A erases aie.objectfifo.register_external_buffers ops.
// conduit.register_buffers has been removed; Pass A simply erases the
// objectfifo external buffer registration op.

module {
  aie.device(xcvc1902) {
    %tile70 = aie.tile(7, 0)
    %tile71 = aie.tile(7, 1)

    aie.objectfifo @ext_fifo(%tile70, {%tile71}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %ext_buf = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
    // CHECK-NOT: conduit.register_buffers
    // CHECK-NOT: aie.objectfifo.register_external_buffers
    aie.objectfifo.register_external_buffers @ext_fifo(%tile70, {%ext_buf}) : (memref<64xi32>)

    %core71 = aie.core(%tile71) {
      %subview = aie.objectfifo.acquire @ext_fifo(Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
      %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      aie.objectfifo.release @ext_fifo(Consume, 1)
      aie.end
    }
  }
}
