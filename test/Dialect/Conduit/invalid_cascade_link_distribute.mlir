// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// Regression test (A-10): cascade channels cannot be used in distribute or
// join links.
//
// Cascade routing has no FIFO buffering and no DMA channels — it is a
// register-level rendezvous. Using a cascade conduit as a source or destination
// in a distribute/join link would silently produce incorrect hardware code
// (Pass C emits no flow for cascade, so other participants deadlock).
//
// The Link::verify() must detect and reject this combination.

// -----

// distribute mode referencing a cascade source channel — must error.
func.func @bad_distribute_cascade_src() {
  conduit.create {name = "casc_src", capacity = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64,
                  routing_mode = "cascade"}
  conduit.create {name = "out0", capacity = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  conduit.create {name = "out1", capacity = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 5>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  // expected-error@+1 {{'conduit.link' op cascade channel 'casc_src' cannot be used in a 'distribute' link}}
  conduit.link {srcs = ["casc_src"], dsts = ["out0", "out1"],
                mode = "distribute", memtile = "tile(0,1)"}
  return
}

// -----

// join mode referencing a cascade destination channel — must error.
func.func @bad_join_cascade_dst() {
  conduit.create {name = "in0", capacity = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  conduit.create {name = "in1", capacity = 1 : i64,
                  producer_tile = array<i64: 0, 3>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  conduit.create {name = "casc_dst", capacity = 1 : i64,
                  producer_tile = array<i64: 0, 4>,
                  consumer_tiles = array<i64: 0, 5>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64,
                  routing_mode = "cascade"}
  // expected-error@+1 {{'conduit.link' op cascade channel 'casc_dst' cannot be used in a 'join' link}}
  conduit.link {srcs = ["in0", "in1"], dsts = ["casc_dst"],
                mode = "join", memtile = "tile(0,1)"}
  return
}
