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
  conduit.create {name = "rm_circuit", capacity = 4 : i64,
                  routing_mode = #conduit.routing_mode<circuit>}
  return
}

// CHECK-LABEL: func.func @routing_mode_packet
func.func @routing_mode_packet() {
  // CHECK: routing_mode = #conduit.routing_mode<packet>
  conduit.create {name = "rm_packet", capacity = 4 : i64,
                  routing_mode = #conduit.routing_mode<packet>}
  return
}

// CHECK-LABEL: func.func @routing_mode_cascade
func.func @routing_mode_cascade() {
  // CHECK: routing_mode = #conduit.routing_mode<cascade>
  conduit.create {name = "rm_cascade", capacity = 1 : i64, depth = 1 : i64,
                  routing_mode = #conduit.routing_mode<cascade>}
  return
}

// CHECK-LABEL: func.func @routing_mode_stream
func.func @routing_mode_stream() {
  // CHECK: routing_mode = #conduit.routing_mode<stream>
  conduit.create {name = "rm_stream", capacity = 4 : i64,
                  routing_mode = #conduit.routing_mode<stream>}
  return
}

// -----------------------------------------------------------------------
// Case 2: link_mode enum roundtrip — all three values.
// -----------------------------------------------------------------------

// CHECK-LABEL: func.func @link_mode_distribute
func.func @link_mode_distribute() {
  conduit.create {name = "src", capacity = 4 : i64}
  conduit.create {name = "dst0", capacity = 2 : i64}
  conduit.create {name = "dst1", capacity = 2 : i64}
  // CHECK: mode = #conduit.link_mode<distribute>
  conduit.link {srcs = ["src"], dsts = ["dst0", "dst1"],
                mode = #conduit.link_mode<distribute>,
                memtile = "tile(0,1)"}
  return
}

// CHECK-LABEL: func.func @link_mode_join
func.func @link_mode_join() {
  conduit.create {name = "src0", capacity = 2 : i64}
  conduit.create {name = "src1", capacity = 2 : i64}
  conduit.create {name = "dst", capacity = 4 : i64}
  // CHECK: mode = #conduit.link_mode<join>
  conduit.link {srcs = ["src0", "src1"], dsts = ["dst"],
                mode = #conduit.link_mode<join>,
                memtile = "tile(0,1)"}
  return
}

// CHECK-LABEL: func.func @link_mode_forward
func.func @link_mode_forward() {
  conduit.create {name = "in_fwd", capacity = 4 : i64}
  conduit.create {name = "out_fwd", capacity = 4 : i64}
  // CHECK: mode = #conduit.link_mode<forward>
  conduit.link {srcs = ["in_fwd"], dsts = ["out_fwd"],
                mode = #conduit.link_mode<forward>,
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
  // CHECK: conduit.create
  // CHECK-SAME: name = "unresolved"
  // CHECK-NOT: routing_mode
  conduit.create {name = "unresolved", capacity = 4 : i64}
  return
}
