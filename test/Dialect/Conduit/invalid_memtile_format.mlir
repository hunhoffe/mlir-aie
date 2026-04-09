// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// Negative tests: memtile attribute format validation on relay ops.
//
// The memtile attribute must be of the form "tile(col,row)". Any other
// format is rejected by the verifier, which calls parseTileCoordForVerifier()
// and rejects strings that don't match the expected pattern.

// -----

// scatter with malformed memtile — not "tile(col,row)" format.
func.func @bad_scatter_memtile_format() {
  // expected-error@+1 {{'conduit.scatter' op memtile attribute must be of the form 'tile(col,row)', got 'bad_format'}}
  conduit.scatter{src = @src_chan, dsts = [@dst0] {memtile = "bad_format"}}
  return
}

// -----

// gather with malformed memtile.
func.func @bad_gather_memtile_format() {
  // expected-error@+1 {{'conduit.gather' op memtile attribute must be of the form 'tile(col,row)', got 'memtile_0_1'}}
  conduit.gather{srcs = [@src0], dst = @dst_chan {memtile = "memtile_0_1"}}
  return
}

// -----

// transpose with malformed memtile.
func.func @bad_transpose_memtile_format() {
  // expected-error@+1 {{'conduit.transpose' op memtile attribute must be of the form 'tile(col,row)', got '(0,1)'}}
  conduit.transpose{srcs = [[@s0]], dsts = [[@d0]] {memtile = "(0,1)", offsets = array<i64: 0>}}
  return
}
