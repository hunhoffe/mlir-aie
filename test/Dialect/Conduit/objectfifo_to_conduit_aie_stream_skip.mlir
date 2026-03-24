// RUN: aie-opt --objectfifo-to-conduit %s 2>&1 | FileCheck %s
//
// Pass A test: objectfifo with aie_stream attribute is converted to
// conduit.create with routing_mode="stream" and aie_stream_port attribute.
// The producer core outputs data directly through the Core AXI stream port
// (Core:N), bypassing DMA. Pass C emits aie.flow(Core:N, ...) and skips
// producer-side buffer/lock allocation.
//
// Expected behavior:
//   - A conduit.create is emitted with routing_mode = #conduit.routing_mode<stream>.
//   - The aie_stream_port generic attribute is set.
//   - The aie.objectfifo op is erased.
//
// Source: based on test/objectFifo-stateful-transform/aie_stream/producer_stream_AIE2.mlir

// CHECK-LABEL: module @aie_stream_convert
// CHECK:   aie.device(xcve2302) {
// CHECK:     conduit.create @of_stream {
// CHECK-SAME:   aie_stream_port = 0
// CHECK-SAME:   routing_mode = #conduit.routing_mode<stream>
// CHECK-NOT:   aie.objectfifo
// CHECK:   }
// CHECK: }

module @aie_stream_convert {
  aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of_stream (%tile12, {%tile13}, 3 : i32) {aie_stream = 0 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>
  }
}
