// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative tests for cascade op verifiers:
//   (a) put_cascade references a conduit with routing_mode != "cascade" → error
//   (b) get_cascade references a conduit with routing_mode != "cascade" → error
//   (c) put_cascade with non-integer type → error
//   (d) put_cascade with wrong-width integer (64 bits) → error
//   (e) conduit.link cascade mode → error (cascade mode not supported in link)
//
// These test the PutCascade::verify() and GetCascade::verify() paths in
// ConduitOps.cpp that call checkCascadeConduit() and checkCascadeValueType().

// -----

// (a) put_cascade referencing a circuit-mode conduit must be rejected.
// checkCascadeConduit() finds the conduit.create, sees routing_mode="circuit"
// (not "cascade"), and emits the error.

conduit.create {name = "circuit_ch", capacity = 4 : i64,
                routing_mode = #conduit.routing_mode<circuit>}

func.func @put_cascade_wrong_routing_mode() {
  %v = arith.constant dense<0> : vector<16xi32>
  // expected-error @+1 {{'conduit.put_cascade' op references conduit 'circuit_ch' which does not have routing_mode = #conduit.routing_mode<cascade>}}
  conduit.put_cascade "circuit_ch" (%v : vector<16xi32>)
  return
}

// -----

// (b) get_cascade referencing a packet-mode conduit must be rejected.

conduit.create {name = "packet_ch", capacity = 4 : i64,
                routing_mode = #conduit.routing_mode<packet>}

func.func @get_cascade_wrong_routing_mode() {
  // expected-error @+1 {{'conduit.get_cascade' op references conduit 'packet_ch' which does not have routing_mode = #conduit.routing_mode<cascade>}}
  %v = conduit.get_cascade "packet_ch" : vector<16xi32>
  return
}

// -----

// (c) put_cascade with a floating-point type must be rejected.
// checkCascadeValueType() returns 0 bits for f32, emits the "not an integer
// or integer vector type" error.

conduit.create {name = "cas_float", capacity = 1 : i64,
                routing_mode = #conduit.routing_mode<cascade>}

func.func @put_cascade_float_type() {
  %v = arith.constant 0.0 : f32
  // expected-error @+1 {{'conduit.put_cascade' op cascade value type 'f32' is not an integer or integer vector type}}
  conduit.put_cascade "cas_float" (%v : f32)
  return
}

// -----

// (d) put_cascade with a 64-bit integer must be rejected.
// checkCascadeValueType() computes 64 bits, which is neither 384 (AIE1) nor
// 512 (AIE2); emits the "has width N bits; must be 384 bits ... or 512 bits" error.

conduit.create {name = "cas_narrow", capacity = 1 : i64,
                routing_mode = #conduit.routing_mode<cascade>}

func.func @put_cascade_wrong_width() {
  %v = arith.constant 0 : i64
  // expected-error @+1 {{'conduit.put_cascade' op cascade value type 'i64' has width 64 bits; must be 384 bits}}
  conduit.put_cascade "cas_narrow" (%v : i64)
  return
}

// -----

// (e) conduit.link with mode="cascade" is rejected by the ODS enum parser.
// "cascade" is not a valid LinkMode enum value (only distribute/join/forward).

func.func @link_cascade_mode() {
  // expected-error @+1 {{attribute 'mode' failed to satisfy constraint: Conduit link mode}}
  conduit.link {srcs = ["src"], dsts = ["dst"],
                mode = "cascade", memtile = "tile(0,1)"}
  return
}
