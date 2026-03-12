// RUN: aie-opt %s -split-input-file -verify-diagnostics

// distribute mode: offsets count must equal dsts count
func.func @bad_distribute_offsets() {
  // expected-error@+1 {{'conduit.objectfifo_link' op distribute mode: offsets count (1) must equal dsts count (2)}}
  conduit.objectfifo_link {srcs = ["in"], dsts = ["out0", "out1"],
                           mode = "distribute", memtile = "tile(0,1)",
                           offsets = array<i64: 0>}
  return
}

// -----

// join mode: offsets count must equal srcs count
func.func @bad_join_offsets() {
  // expected-error@+1 {{'conduit.objectfifo_link' op join mode: offsets count (1) must equal srcs count (2)}}
  conduit.objectfifo_link {srcs = ["in0", "in1"], dsts = ["out"],
                           mode = "join", memtile = "tile(0,1)",
                           offsets = array<i64: 0>}
  return
}
