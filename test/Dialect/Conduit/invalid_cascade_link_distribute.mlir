// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// Regression test (A-10): cascade channels cannot be used in scatter or
// gather links.
//
// NOTE: The cascade channel check for scatter/gather is enforced by Pass C
// (conduit-to-dma linkPhase), NOT by the op-level verifier. ScatterOp and
// GatherOp verifiers only check DMA budget and memtile format.
//
// This file validates that scatter/gather parse correctly with cascade-mode
// conduits present — the cascade rejection happens at lowering time, not
// during verification.

// -----

// scatter with cascade-mode source conduit — parses without verifier error.
// The cascade channel rejection fires in --conduit-to-dma, not here.
func.func @scatter_with_cascade_src() {
  conduit.create @casc_src {slot_elems = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64,
                  routing_mode = #conduit.routing_mode<cascade>}
  conduit.create @out0 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  conduit.create @out1 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 5>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  conduit.scatter{src = @casc_src, dsts = [@out0, @out1] {memtile = "tile(0,1)"}}
  return
}

// -----

// gather with cascade-mode destination conduit — parses without verifier error.
func.func @gather_with_cascade_dst() {
  conduit.create @in0 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  conduit.create @in1 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 0, 3>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  conduit.create @casc_dst {slot_elems = 1 : i64,
                  producer_tile = array<i64: 0, 4>,
                  consumer_tiles = array<i64: 0, 5>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64,
                  routing_mode = #conduit.routing_mode<cascade>}
  conduit.gather{srcs = [@in0, @in1], dst = @casc_dst {memtile = "tile(0,1)"}}
  return
}
