// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// Negative test: conduit.gather DMA budget check.
//
// A MemTile has 6 MM2S + 6 S2MM channels (12 total). A gather op uses
// N S2MM for the sources and 1 MM2S for the destination.  With N > 11
// the total exceeds 12 and must be rejected.

// -----

// gather with 12 srcs: 12 S2MM + 1 MM2S = 13 channels — exceeds MemTile budget.
func.func @bad_gather_too_many_srcs() {
  // expected-error@+1 {{'conduit.gather' op gather DMA budget exceeded: 12 srcs + 1 dst = 13 channels, maximum is 12 (MemTile has 6 MM2S + 6 S2MM)}}
  conduit.gather{srcs = [@src0, @src1, @src2, @src3, @src4, @src5,
          @src6, @src7, @src8, @src9, @src10, @src11], dst = @dst_chan {memtile = "tile(0,1)"}}
  return
}
