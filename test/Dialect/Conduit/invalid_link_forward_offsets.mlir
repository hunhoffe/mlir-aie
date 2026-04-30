// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative tests for conduit.scatter with structural violations.
//
// ScatterOp::verify() enforces at least 1 dst. This test verifies
// the structural invariant for edge cases.
//
// Note: conduit.scatter takes a single `src` (FlatSymbolRefAttr),
// so the 0-srcs case cannot arise at the parser level.

// -----

// scatter with 0 dsts — structural violation.
func.func @scatter_no_dsts() {
  %mt = aie.tile(0, 1)
  // expected-error @+1 {{'conduit.scatter' op scatter requires at least 1 dst, got 0}}
  conduit.scatter{src = @in, dsts = [], memtile = %mt}
  return
}
