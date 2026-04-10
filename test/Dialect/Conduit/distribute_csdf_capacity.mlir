// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// M6-dist / M7-dist: 1:N distribute with CSDF rate annotations.
//
// Denolf et al. 2007 (DOI: 10.1155/2007/84078) defines the 1:N multi-consumer
// (nondestructive read / distribute) pattern as reducible to standard CSDF via
// per-edge Bilsen balance applied to each (src, dst_i) pair independently.
//
// Implementation note: MLIR verifies conduit.create ops before conduit.link.
// Errors on individual conduit.create ops (imbalanced rates, undersized buffers)
// fire via Create::verify() (M6/M7) before Link::verify() (M6-dist/M7-dist)
// is reached.  The conduit.link M6-dist/M7-dist checks serve as a secondary
// pass that catches cross-conduit consistency issues when rates are present.
//
// Section 1: 1→3 distribute, all conduits balanced and correctly sized — PASS
// Section 2: dst2 conduit has imbalanced rates (classic period-imbalanced case) — FAIL
// Section 3: dst1 conduit has balanced rates but undersized buffer — FAIL

// -----

// Section 1: PASS — 1→3 distribute, all individually balanced, sufficient buffers.
//
// src: P=[2], C=[2] → 2*1 == 2*1 ✓
// dst1, dst2, dst3: P=[1], C=[1] → 1*1 == 1*1 ✓

func.func @distribute_all_pass() {
  conduit.create @dist_src_ok {slot_elems = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<4xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  conduit.create @dist_d1_ok {slot_elems = 2 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<2xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 1>,
                  consumer_rates = array<i64: 1>}
  conduit.create @dist_d2_ok {slot_elems = 2 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 1, 2>,
                  element_type = memref<2xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 1>,
                  consumer_rates = array<i64: 1>}
  conduit.create @dist_d3_ok {slot_elems = 2 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 2, 2>,
                  element_type = memref<2xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 1>,
                  consumer_rates = array<i64: 1>}
  conduit.scatter{src = @dist_src_ok,
                dsts = [@dist_d1_ok, @dist_d2_ok, @dist_d3_ok] {memtile = "tile(0,1)"}}
  return
}

// -----

// Section 2: FAIL — dst2 conduit has imbalanced rates.
//
// dst2: producer_rates=[3] (sum=3, period=1), consumer_rates=[1,2] (sum=3, period=2)
//   M6: sum(P)*len(C) = 3*2 = 6 != sum(C)*len(P) = 3*1 = 3 → FAIL
//   Error fires on conduit.create for dst2 (Create::verify M6 runs first).

func.func @distribute_dst2_imbalanced() {
  conduit.create @dist2_src {slot_elems = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<i32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  conduit.create @dist2_d1 {slot_elems = 2 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<i32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 1>,
                  consumer_rates = array<i64: 1>}
  // expected-error@+1 {{'conduit.create' op CSDF rate imbalance: sum(producer_rates)*len(consumer_rates)=6 != sum(consumer_rates)*len(producer_rates)=3}}
  conduit.create @dist2_d2_bad {slot_elems = 3 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 1, 2>,
                  element_type = memref<i32>,
                  depth = 3 : i64,
                  producer_rates = array<i64: 3>,
                  consumer_rates = array<i64: 1, 2>}
  conduit.create @dist2_d3 {slot_elems = 2 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 2, 2>,
                  element_type = memref<i32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 1>,
                  consumer_rates = array<i64: 1>}
  conduit.scatter{src = @dist2_src,
                dsts = [@dist2_d1, @dist2_d2_bad, @dist2_d3] {memtile = "tile(0,1)"}}
  return
}

// -----

// Section 3: FAIL — dst1 conduit is balanced but buffer undersized.
//
// dst1: producer_rates=[3,1] (sum=4, period=2), consumer_rates=[2] (sum=2, period=1)
//   M6: 4*1 == 2*2 ✓  (passes M6)
//   M7: hyper-period H=2: t=0: produce 3 (occ=3), consume 2 (occ=1) — peak=3
//                          t=1: produce 1 (occ=2), consume 2 (occ=0)
//   peak=3 > slot_elems =2 → M7 error fires on conduit.create dst1.

func.func @distribute_dst1_capacity() {
  conduit.create @dist3_src {slot_elems = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<i32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  // expected-error@+1 {{M7: CSDF buffer capacity insufficient: peak token occupancy over one hyper-period=3 exceeds slot_elems =2}}
  conduit.create @dist3_d1_small {slot_elems = 2 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<i32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 3, 1>,
                  consumer_rates = array<i64: 2>}
  conduit.create @dist3_d2 {slot_elems = 2 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 1, 2>,
                  element_type = memref<i32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 1>,
                  consumer_rates = array<i64: 1>}
  conduit.scatter{src = @dist3_src,
                dsts = [@dist3_d1_small, @dist3_d2] {memtile = "tile(0,1)"}}
  return
}
