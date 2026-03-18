// RUN: aie-opt --objectfifo-to-conduit %s 2>&1 | FileCheck %s
//
// Pass A test: objectfifo with aie_stream attribute must be skipped entirely.
// aie_stream routes data through the Core AXI stream port (Core:N → DMA:0),
// not DMA, which requires a fundamentally different lowering than Pass A provides.
//
// Expected behavior:
//   - A remark is emitted naming the skipped fifo.
//   - The aie.objectfifo op is left intact in the output IR (NOT erased).
//   - No conduit.create is emitted for the aie_stream fifo.
//   - No buffers or locks are placed on the wrong tile.
//
// Source: based on test/objectFifo-stateful-transform/aie_stream/producer_stream_AIE2.mlir

// CHECK: remark: objectfifo-to-conduit: aie_stream ObjectFIFO not yet supported
// CHECK-SAME: skipping

// CHECK-LABEL: module @aie_stream_skip
// CHECK:   aie.device(xcve2302) {
// CHECK-NOT:   conduit.create
// CHECK-NOT:   aie.buffer
// CHECK-NOT:   aie.lock
// CHECK:     aie.objectfifo @of_stream
// CHECK-SAME:   aie_stream
// CHECK:   }
// CHECK: }

module @aie_stream_skip {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of_stream (%tile12, {%tile13}, 3 : i32) {aie_stream = 0 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}
