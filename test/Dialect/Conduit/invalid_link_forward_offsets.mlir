// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative tests for conduit.link forward mode with offsets.
//
// Link::verify() does not check offsets for "forward" mode specifically —
// only distribute and join modes have offset-count invariants. However,
// the mode validation catches structural errors. This test verifies the
// forward mode structural invariant (exactly 1 src and 1 dst).
//
// This supplements invalid.mlir which covers distribute/join offset mismatches
// and forward mode with 2 srcs/2 dsts. Here we add:
//   - forward with 0 srcs (edge case)
//   - forward with 0 dsts (edge case)
//
// Note: the forward mode error message says "got N src(s) and M dst(s)".

// -----

// forward with 0 srcs — structural violation.
func.func @forward_no_srcs() {
  // expected-error @+1 {{'conduit.link' op forward mode requires exactly 1 src and 1 dst, got 0 src(s) and 1 dst(s)}}
  conduit.link {srcs = [], dsts = ["out"],
                mode = "forward", memtile = "tile(0,1)"}
  return
}

// -----

// forward with 0 dsts — structural violation.
func.func @forward_no_dsts() {
  // expected-error @+1 {{'conduit.link' op forward mode requires exactly 1 src and 1 dst, got 1 src(s) and 0 dst(s)}}
  conduit.link {srcs = ["in"], dsts = [],
                mode = "forward", memtile = "tile(0,1)"}
  return
}
