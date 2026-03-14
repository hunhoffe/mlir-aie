// RUN: aie-opt --objectfifo-to-conduit --verify-diagnostics %s
//
// Negative test: Pass A emits an error when it encounters
// aie.objectfifo.register_external_buffers (ext-buf gap).
//
// Fix 1c: ObjectFifoToConduit.cpp walks for
// ObjectFifoRegisterExternalBuffersOp and emits emitError() with the message
// "register_external_buffers not yet supported".
//
// Using --verify-diagnostics so the expected-error annotation is checked
// and the test passes (exit 0) even though the pass signals failure.

module {
  aie.device(xcvc1902) {
    %tile70 = aie.tile(7, 0)
    %tile71 = aie.tile(7, 1)

    aie.objectfifo @ext_fifo(%tile70, {%tile71}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %ext_buf = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
    // expected-error @+1 {{register_external_buffers not yet supported}}
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
