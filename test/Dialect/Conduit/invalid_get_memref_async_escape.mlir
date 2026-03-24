// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative test: M10 token escape on conduit.get_memref_async.
//
// GetMemrefAsync::verify() calls checkTokenDoesNotEscape() on its result token.
// This complements invalid.mlir which covers put_memref_async escape and
// acquire_async escape. The get_memref_async path (Tier 3 receive) is
// symmetric but not covered elsewhere.

// -----

// get_memref_async dma.token escapes via return.
func.func @get_memref_async_escape_return() -> !conduit.dma.token {
  conduit.create @recv_ch {capacity = 64 : i64}
  // expected-error @+1 {{'conduit.get_memref_async' op M10: token escapes function scope via return}}
  %tok = conduit.get_memref_async {name = "recv_ch", num_elems = 64 : i64,
             offsets = array<i64: 0>, sizes = array<i64: 64>,
             strides = array<i64: 1>} : !conduit.dma.token
  return %tok : !conduit.dma.token
}
