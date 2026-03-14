// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Test: Pass A lowers aie.objectfifo.register_external_buffers into
// conduit.register_external_buffers with the correct tile coordinates
// and external buffer SSA operands.

module {
  aie.device(xcvc1902) {
    %tile70 = aie.tile(7, 0)
    %tile71 = aie.tile(7, 1)

    aie.objectfifo @ext_fifo(%tile70, {%tile71}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %ext_buf = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
    // CHECK: conduit.register_external_buffers({{%.*}}) {name = "ext_fifo", tile_coord = array<i64: 7, 0>} : (memref<64xi32>)
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
