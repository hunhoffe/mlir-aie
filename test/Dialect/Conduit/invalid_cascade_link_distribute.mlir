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
  conduit.create @casc_src {capacity = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64,
                  routing_mode = #conduit.routing_mode<cascade>}
  conduit.create @out0 {capacity = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  conduit.create @out1 {capacity = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 5>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  // expected-error@+1 {{'conduit.distribute' op cascade channel 'casc_src' cannot be used in a distribute src}}
  conduit.distribute {srcs = ["casc_src"], dsts = ["out0", "out1"], memtile = "tile(0,1)"}
  return
}

// -----

// join mode referencing a cascade destination channel — must error.
func.func @bad_join_cascade_dst() {
  conduit.create @in0 {capacity = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  conduit.create @in1 {capacity = 1 : i64,
                  producer_tile = array<i64: 0, 3>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  conduit.create @casc_dst {capacity = 1 : i64,
                  producer_tile = array<i64: 0, 4>,
                  consumer_tiles = array<i64: 0, 5>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64,
                  routing_mode = #conduit.routing_mode<cascade>}
  // expected-error@+1 {{'conduit.join' op cascade channel 'casc_dst' cannot be used in a join dst}}
  conduit.join {srcs = ["in0", "in1"], dsts = ["casc_dst"], memtile = "tile(0,1)"}
  return
}
