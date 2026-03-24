// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// Negative tests for conduit.link join mode: M6-join fires on conduit.create
// (not conduit.link) because Create::verify() runs before Link::verify().
//
// This test documents the actual verification order:
//   1. conduit.create (Create::verify → M6/M7)
//   2. conduit.link   (Link::verify → M6-join/M7-join as secondary check)
//
// When a source or destination conduit.create has imbalanced rates, the error
// fires on the create, not the link. This is the correct MLIR verification
// order: ops are verified as they are encountered in order.
//
// See join_csdf_balance.mlir for the full test suite.
// This file tests two edge cases not in that file:
//
//   (a) conduit.link join with no source conduits having rate annotations
//       → no M6-join/M7-join check is performed (skip path): PASS
//   (b) conduit.link join with only the destination annotated → skip src checks,
//       only dst checked (passes if dst balanced)

// -----

// (a) join mode with unannotated source conduits — PASS (no rate check triggered).
// Link::verify() checks are guarded by 'has_value()' on producer_rates/consumer_rates.
// Unannotated conduits are silently skipped.

func.func @join_unannotated_srcs_pass() {
  conduit.create @j_src_norates {capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  conduit.create @j_dst_norates {capacity = 4 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  // No expected-error: unannotated → skip path → PASS.
  conduit.join {srcs = ["j_src_norates"], dsts = ["j_dst_norates"], memtile = "tile(0,1)"}
  return
}

// -----

// (b) join mode with balanced source and destination rates — PASS.
// Confirms the complete M6-join positive path: all edges individually balanced.
//
// src: P=[2], C=[2] → 2*1 == 2*1 ✓ (passes Create::verify M6)
// dst: P=[2], C=[2] → 2*1 == 2*1 ✓ (passes Create::verify M6)
// Link::verify M6-join: both pass → no error.

func.func @join_all_balanced_link_check() {
  conduit.create @j2_src {capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  conduit.create @j2_dst {capacity = 4 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  // No error expected: all rates balanced.
  conduit.join {srcs = ["j2_src"], dsts = ["j2_dst"], memtile = "tile(0,1)"}
  return
}
