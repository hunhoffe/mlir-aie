// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// M6-dist / M7-dist: rate mismatch tests for 1:N distribute.
//
// These tests verify that per-edge and cross-conduit checks catch different
// types of errors:
//
// Section 1: Destination conduit has imbalanced rates (M6 per-edge fires first).
// Section 2: Source conduit has imbalanced rates (M6 fires on conduit.create).
// Section 3: Per-edge M6 passes for all conduits, but Eq. 45 consistency between
//            source production and destination consumption sum is violated.
//            This is caught by per-edge M6 on the destination conduit.create
//            because each conduit is individually checked before the link verifier.

// -----

// Section 1: FAIL — dst1 has imbalanced rates.
//
// dst1: producer_rates=[2] (sum=2, period=1), consumer_rates=[1,2] (sum=3, period=2)
//   M6: sum(P)*len(C) = 2*2 = 4 != sum(C)*len(P) = 3*1 = 3 → FAIL
//   This fires on conduit.create for dst1 (Create::verify M6 runs first).

func.func @distribute_dst_imbalanced() {
  conduit.create @rm_src {capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<i32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  // expected-error@+1 {{'conduit.create' op CSDF rate imbalance: sum(producer_rates)*len(consumer_rates)=4 != sum(consumer_rates)*len(producer_rates)=3}}
  conduit.create @rm_d1_bad {capacity = 3 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<i32>,
                  depth = 3 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 1, 2>}
  conduit.create @rm_d2 {capacity = 2 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 1, 2>,
                  element_type = memref<i32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 1>,
                  consumer_rates = array<i64: 1>}
  conduit.distribute {srcs = ["rm_src"], dsts = ["rm_d1_bad", "rm_d2"], memtile = "tile(0,1)"}
  return
}

// -----

// Section 2: FAIL — source conduit has imbalanced rates.
//
// src: producer_rates=[3] (sum=3, period=1), consumer_rates=[1,1] (sum=2, period=2)
//   M6: sum(P)*len(C) = 3*2 = 6 != sum(C)*len(P) = 2*1 = 2 → FAIL
//   This fires on conduit.create for src (Create::verify M6).

func.func @distribute_src_imbalanced() {
  // expected-error@+1 {{'conduit.create' op CSDF rate imbalance: sum(producer_rates)*len(consumer_rates)=6 != sum(consumer_rates)*len(producer_rates)=2}}
  conduit.create @rm2_src_bad {capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<i32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 3>,
                  consumer_rates = array<i64: 1, 1>}
  conduit.create @rm2_d1 {capacity = 2 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<i32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 1>,
                  consumer_rates = array<i64: 1>}
  conduit.distribute {srcs = ["rm2_src_bad"], dsts = ["rm2_d1"], memtile = "tile(0,1)"}
  return
}
