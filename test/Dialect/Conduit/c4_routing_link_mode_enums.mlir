// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics %s
//
// C-4 regression: routing_mode and link_mode ODS enum syntax.
//
// Three cases verified:
//   1. Positive: all valid routing_mode enum values roundtrip correctly.
//   2. Positive: scatter and gather relay ops roundtrip correctly.
//   3. Positive: absent routing_mode (unresolved) is valid and not rejected.
//
// Negative cases (invalid string values rejected by ODS parser) are in
// invalid.mlir: @bad_routing_mode and @bad_unknown_mode.

aie.device(npu1) {

// -----------------------------------------------------------------------
// Case 1: routing_mode enum roundtrip — all four values.
// -----------------------------------------------------------------------

// CHECK: routing_mode = #conduit.routing_mode<circuit>
conduit.create @rm_circuit {depth = 0 : i64,
                element_type = memref<4xi32>,
                routing_mode = #conduit.routing_mode<circuit>}
// CHECK: routing_mode = #conduit.routing_mode<packet>
conduit.create @rm_packet {depth = 0 : i64,
                element_type = memref<4xi32>,
                routing_mode = #conduit.routing_mode<packet>}
// CHECK: routing_mode = #conduit.routing_mode<cascade>
conduit.create @rm_cascade {depth = 1 : i64,
                element_type = memref<64xi32>,
                routing_mode = #conduit.routing_mode<cascade>}
// CHECK: routing_mode = #conduit.routing_mode<stream>
conduit.create @rm_stream {depth = 0 : i64,
                element_type = memref<4xi32>,
                routing_mode = #conduit.routing_mode<stream>}

// -----------------------------------------------------------------------
// Case 2: relay op roundtrip — scatter and gather.
// -----------------------------------------------------------------------

conduit.create @src {depth = 0 : i64, element_type = memref<4xi32>}
conduit.create @dst0 {depth = 0 : i64, element_type = memref<4xi32>}
conduit.create @dst1 {depth = 0 : i64, element_type = memref<4xi32>}
conduit.create @src0 {depth = 0 : i64, element_type = memref<4xi32>}
conduit.create @src1 {depth = 0 : i64, element_type = memref<4xi32>}
conduit.create @dst {depth = 0 : i64, element_type = memref<4xi32>}
conduit.create @in_fwd {depth = 0 : i64, element_type = memref<4xi32>}
conduit.create @out_fwd {depth = 0 : i64, element_type = memref<4xi32>}

// -----------------------------------------------------------------------
// Case 3: absent routing_mode (unresolved) is valid.
// -----------------------------------------------------------------------

// CHECK: conduit.create @unresolved
// CHECK-NOT: routing_mode
conduit.create @unresolved {depth = 0 : i64, element_type = memref<4xi32>}

// CHECK-LABEL: func.func @routing_mode_circuit
func.func @routing_mode_circuit() {
  return
}

// CHECK-LABEL: func.func @routing_mode_packet
func.func @routing_mode_packet() {
  return
}

// CHECK-LABEL: func.func @routing_mode_cascade
func.func @routing_mode_cascade() {
  return
}

// CHECK-LABEL: func.func @routing_mode_stream
func.func @routing_mode_stream() {
  return
}

// CHECK-LABEL: func.func @relay_scatter
func.func @relay_scatter() {
  // CHECK: conduit.scatter{src = @src, dsts = [@dst0, @dst1]
  // CHECK-SAME: memtile = "tile(0,1)"
  conduit.scatter{src = @src, dsts = [@dst0, @dst1] {memtile = "tile(0,1)"}}
  return
}

// CHECK-LABEL: func.func @relay_gather
func.func @relay_gather() {
  // CHECK: conduit.gather{srcs = [@src0, @src1], dst = @dst
  // CHECK-SAME: memtile = "tile(0,1)"
  conduit.gather{srcs = [@src0, @src1], dst = @dst {memtile = "tile(0,1)"}}
  return
}

// CHECK-LABEL: func.func @relay_scatter_forward
func.func @relay_scatter_forward() {
  // CHECK: conduit.scatter{src = @in_fwd, dsts = [@out_fwd]
  // CHECK-SAME: memtile = "tile(0,1)"
  conduit.scatter{src = @in_fwd, dsts = [@out_fwd] {memtile = "tile(0,1)"}}
  return
}

// CHECK-LABEL: func.func @absent_routing_mode
func.func @absent_routing_mode() {
  return
}

} // aie.device
