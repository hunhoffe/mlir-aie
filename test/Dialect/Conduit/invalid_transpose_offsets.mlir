// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// Negative tests for conduit.transpose verifier checks:
//   1. offsets size: offsets.size() must equal srcs.size() * dsts.size().
//   2. Packet ID budget: srcs.size() * dsts.size() <= 32 (AIE2 packet ID space).
//
// Note: TransposeOp requires $offsets (DenseI64ArrayAttr, not optional).

// -----

// transpose with offsets.size() != srcs.size() * dsts.size().
// 2 srcs * 2 dsts = 4 expected offsets, but only 2 provided.
func.func @bad_transpose_offsets_size() {
  // expected-error@+1 {{'conduit.transpose' op offsets size must equal srcs.size() * dsts.size() = 4, got 2}}
  conduit.transpose{srcs = [[@s0, @s1]], dsts = [[@d0, @d1]] {memtile = "tile(0,1)", offsets = array<i64: 0, 512>}}
  return
}

// -----

// transpose with 6 srcs * 6 dsts = 36 packet IDs — exceeds AIE2 packet ID budget of 32.
// offsets must be provided: 36 entries.
func.func @bad_transpose_packet_budget() {
  // expected-error@+1 {{'conduit.transpose' op transpose packet ID budget exceeded: 6 * 6 = 36, maximum is 32 (AIE2 packet ID space)}}
  conduit.transpose{srcs = [[@s0, @s1, @s2, @s3, @s4, @s5]], dsts = [[@d0, @d1, @d2, @d3, @d4, @d5]] {memtile = "tile(0,1)", offsets = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                  20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                  30, 31, 32, 33, 34, 35>}}
  return
}
