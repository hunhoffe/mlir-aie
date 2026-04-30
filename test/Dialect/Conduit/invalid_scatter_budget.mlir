// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// Negative test: conduit.scatter DMA budget check.
//
// A MemTile has 6 MM2S + 6 S2MM channels (12 total). A scatter op uses
// 1 S2MM for the source and N MM2S for the destinations.  With N > 11
// the total exceeds 12 and must be rejected.

// -----

// scatter with 12 dsts: 1 S2MM + 12 MM2S = 13 channels — exceeds MemTile budget.
func.func @bad_scatter_too_many_dsts() {
  %mt = aie.tile(0, 1)
  // expected-error@+1 {{'conduit.scatter' op scatter DMA budget exceeded: 1 src + 12 dsts = 13 channels, maximum is 12 (MemTile has 6 MM2S + 6 S2MM)}}
  conduit.scatter{src = @src_chan, dsts = [@dst0, @dst1, @dst2, @dst3, @dst4, @dst5,
          @dst6, @dst7, @dst8, @dst9, @dst10, @dst11], memtile = %mt}
  return
}
