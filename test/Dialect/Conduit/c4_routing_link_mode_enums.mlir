// RUN: aie-opt %s | FileCheck %s
// RUN: aie-opt --verify-diagnostics %s
//
// C-4 regression: routing_mode and link_mode ODS enum syntax.
//
// Three cases verified:
//   1. Positive: all valid routing_mode enum values roundtrip correctly.
//   2. Positive: all valid link_mode enum values roundtrip correctly.
//   3. Positive: absent routing_mode (unresolved) is valid and not rejected.
//
// Negative cases (invalid string values rejected by ODS parser) are in
// invalid.mlir: @bad_routing_mode and @bad_unknown_mode.

// -----------------------------------------------------------------------
// Case 1: routing_mode enum roundtrip — all four values.
// -----------------------------------------------------------------------

// CHECK-LABEL: func.func @routing_mode_circuit
func.func @routing_mode_circuit() {
  // CHECK: routing_mode = #conduit.routing_mode<circuit>
  conduit.create @rm_circuit {slot_elems = 4 : i64, depth = 0 : i64,
                  routing_mode = #conduit.routing_mode<circuit>}
  return
}

// CHECK-LABEL: func.func @routing_mode_packet
func.func @routing_mode_packet() {
  // CHECK: routing_mode = #conduit.routing_mode<packet>
  conduit.create @rm_packet {slot_elems = 4 : i64, depth = 0 : i64,
                  routing_mode = #conduit.routing_mode<packet>}
  return
}

// CHECK-LABEL: func.func @routing_mode_cascade
func.func @routing_mode_cascade() {
  // CHECK: routing_mode = #conduit.routing_mode<cascade>
  conduit.create @rm_cascade {slot_elems = 1 : i64, depth = 1 : i64,
                  routing_mode = #conduit.routing_mode<cascade>}
  return
}

// CHECK-LABEL: func.func @routing_mode_stream
func.func @routing_mode_stream() {
  // CHECK: routing_mode = #conduit.routing_mode<stream>
  conduit.create @rm_stream {slot_elems = 4 : i64, depth = 0 : i64,
                  routing_mode = #conduit.routing_mode<stream>}
  return
}

// -----------------------------------------------------------------------
// Case 2: link_mode enum roundtrip — all three values.
// -----------------------------------------------------------------------

// CHECK-LABEL: func.func @link_mode_distribute
func.func @link_mode_distribute() {
  conduit.create @src {slot_elems = 4 : i64, depth = 0 : i64}
  conduit.create @dst0 {slot_elems = 2 : i64, depth = 0 : i64}
  conduit.create @dst1 {slot_elems = 2 : i64, depth = 0 : i64}
  // CHECK: conduit.distribute
  // CHECK-SAME: memtile = "tile(0,1)"
  conduit.distribute {srcs = [@src], dsts = [@dst0, @dst1],
                memtile = "tile(0,1)"}
  return
}

// CHECK-LABEL: func.func @link_mode_join
func.func @link_mode_join() {
  conduit.create @src0 {slot_elems = 2 : i64, depth = 0 : i64}
  conduit.create @src1 {slot_elems = 2 : i64, depth = 0 : i64}
  conduit.create @dst {slot_elems = 4 : i64, depth = 0 : i64}
  // CHECK: conduit.join
  // CHECK-SAME: memtile = "tile(0,1)"
  conduit.join {srcs = [@src0, @src1], dsts = [@dst],
                memtile = "tile(0,1)"}
  return
}

// CHECK-LABEL: func.func @link_mode_forward
func.func @link_mode_forward() {
  conduit.create @in_fwd {slot_elems = 4 : i64, depth = 0 : i64}
  conduit.create @out_fwd {slot_elems = 4 : i64, depth = 0 : i64}
  // CHECK: conduit.forward
  // CHECK-SAME: memtile = "tile(0,1)"
  conduit.forward {srcs = [@in_fwd], dsts = [@out_fwd],
                memtile = "tile(0,1)"}
  return
}

// -----------------------------------------------------------------------
// Case 3: absent routing_mode (unresolved) is valid.
// A conduit.create with no routing_mode attribute must parse and print
// without error. The attribute is absent in the output.
// -----------------------------------------------------------------------

// CHECK-LABEL: func.func @absent_routing_mode
func.func @absent_routing_mode() {
  // CHECK: conduit.create @unresolved
  // CHECK-NOT: routing_mode
  conduit.create @unresolved {slot_elems = 4 : i64, depth = 0 : i64}
  return
}
