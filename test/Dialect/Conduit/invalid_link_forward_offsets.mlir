// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative tests for conduit.forward with structural violations.
//
// Forward::verify() enforces exactly 1 src and 1 dst. This test verifies
// the structural invariant for edge cases.
//
// This supplements invalid.mlir which covers distribute/join offset mismatches
// and forward mode with 2 srcs/2 dsts. Here we add:
//   - forward with 0 srcs (edge case)
//   - forward with 0 dsts (edge case)
//
// Note: the forward error message says "got N src(s) and M dst(s)".

// -----

// forward with 0 srcs — structural violation.
func.func @forward_no_srcs() {
  // expected-error @+1 {{'conduit.forward' op forward requires exactly 1 src and 1 dst, got 0 src(s) and 1 dst(s)}}
  conduit.forward {srcs = [], dsts = ["out"], memtile = "tile(0,1)"}
  return
}

// -----

// forward with 0 dsts — structural violation.
func.func @forward_no_dsts() {
  // expected-error @+1 {{'conduit.forward' op forward requires exactly 1 src and 1 dst, got 1 src(s) and 0 dst(s)}}
  conduit.forward {srcs = ["in"], dsts = [], memtile = "tile(0,1)"}
  return
}
