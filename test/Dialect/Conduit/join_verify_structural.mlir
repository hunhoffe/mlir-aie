// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// M6-join: structural verification for N:1 join.
//
// Denolf et al. 2007 §3.3.4 proves that N:1 join (multi-producer) channels
// have NO equivalent standard CSDF channel: "there does not exist an
// equivalent standard channel for a channel with multiple producers.  The
// reason is that token order depends on runtime response time."
//
// Consequence: the join verifier applies per-edge Bilsen 1:1 checks to each
// source and the destination conduit as a conservative structural approximation.
// No cross-conduit composed-produce analysis is attempted because the theory
// does not support it — unlike the 1:N distribute case (Denolf §3.3.3) where
// composed consume is well-defined.
//
// Section 1: 3→1 join, all conduits balanced — PASS (structural check OK).
// Section 2: 2→1 join, source has imbalanced rates — M6 fires on conduit.create.
// Section 3: join mode requires exactly 1 dst — M3 structural error.

// -----

// Section 1: PASS — 3→1 join, all individually balanced.
//
// src1: P=[2], C=[2] → per-edge OK ✓
// src2: P=[1,1], C=[1,1] → per-edge OK ✓ (sum=2, period=2)
// src3: P=[4], C=[4] → per-edge OK ✓
// dst: P=[8], C=[8] → per-edge OK ✓
//
// NOTE: no cross-conduit composed-produce check is performed here because
// Denolf §3.3.4 proves it has no CSDF equivalent.  The structural check
// verifies each conduit in isolation.

func.func @join_three_sources_pass() {
  conduit.create {name = "js_s1", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  conduit.create {name = "js_s2", capacity = 4 : i64,
                  producer_tile = array<i64: 1, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<4xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 1, 1>,
                  consumer_rates = array<i64: 1, 1>}
  conduit.create {name = "js_s3", capacity = 8 : i64,
                  producer_tile = array<i64: 2, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 4>,
                  consumer_rates = array<i64: 4>}
  conduit.create {name = "js_dst", capacity = 16 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 0>,
                  element_type = memref<16xi32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 8>,
                  consumer_rates = array<i64: 8>}
  conduit.link {srcs = ["js_s1", "js_s2", "js_s3"], dsts = ["js_dst"],
                mode = #conduit.link_mode<join>, memtile = "tile(0,1)"}
  return
}

// -----

// Section 2: FAIL — source conduit has imbalanced rates.
//
// src1: P=[3], C=[1,1] → M6: sum(P)*len(C) = 3*2 = 6 != sum(C)*len(P) = 2*1 = 2
//   Error fires on conduit.create (Create::verify M6 runs first).

func.func @join_src_imbalanced() {
  // expected-error@+1 {{'conduit.create' op CSDF rate imbalance: sum(producer_rates)*len(consumer_rates)=6 != sum(consumer_rates)*len(producer_rates)=2}}
  conduit.create {name = "ji_s1_bad", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<i32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 3>,
                  consumer_rates = array<i64: 1, 1>}
  conduit.create {name = "ji_dst", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 0>,
                  element_type = memref<i32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 1>,
                  consumer_rates = array<i64: 1>}
  conduit.link {srcs = ["ji_s1_bad"], dsts = ["ji_dst"],
                mode = #conduit.link_mode<join>, memtile = "tile(0,1)"}
  return
}

// -----

// Section 3: FAIL — join mode requires exactly 1 dst.
//
// M3 structural invariant violation: join mode can only have 1 destination.

func.func @join_multiple_dsts() {
  conduit.create {name = "jm_s1", capacity = 2 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<i32>,
                  depth = 1 : i64}
  conduit.create {name = "jm_d1", capacity = 2 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 0>,
                  element_type = memref<i32>,
                  depth = 1 : i64}
  conduit.create {name = "jm_d2", capacity = 2 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 1, 0>,
                  element_type = memref<i32>,
                  depth = 1 : i64}
  // expected-error@+1 {{'conduit.link' op join mode requires exactly 1 dst, got 2}}
  conduit.link {srcs = ["jm_s1"], dsts = ["jm_d1", "jm_d2"],
                mode = #conduit.link_mode<join>, memtile = "tile(0,1)"}
  return
}
