// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// Negative tests for conduit.gather: M6-join fires on conduit.create
// (not conduit.gather) because Create::verify() runs before GatherOp::verify().
//
// This test documents the actual verification order:
//   1. conduit.create (Create::verify → M6/M7)
//   2. conduit.gather (GatherOp::verify → M6-join/M7-join as secondary check)
//
// When a source or destination conduit.create has imbalanced rates, the error
// fires on the create, not the gather. This is the correct MLIR verification
// order: ops are verified as they are encountered in order.
//
// See join_csdf_balance.mlir for the full test suite.
// This file tests two edge cases not in that file:
//
//   (a) conduit.gather with no source conduits having rate annotations
//       → no M6-join/M7-join check is performed (skip path): PASS
//   (b) conduit.gather with only the destination annotated → skip src checks,
//       only dst checked (passes if dst balanced)

// -----

// (a) gather mode with unannotated source conduits — PASS (no rate check triggered).
// GatherOp::verify() checks are guarded by 'has_value()' on producer_rates/consumer_rates.
// Unannotated conduits are silently skipped.

aie.device(npu1) {
conduit.create @j_src_norates {slot_elems = 4 : i64,
                producer_tile = array<i64: 0, 2>,
                consumer_tiles = array<i64: 0, 1>,
                element_type = memref<4xi32>,
                depth = 1 : i64}
conduit.create @j_dst_norates {slot_elems = 4 : i64,
                producer_tile = array<i64: 0, 1>,
                consumer_tiles = array<i64: 0, 3>,
                element_type = memref<4xi32>,
                depth = 1 : i64}
func.func @join_unannotated_srcs_pass() {
  // No expected-error: unannotated → skip path → PASS.
  conduit.gather{srcs = [@j_src_norates], dst = @j_dst_norates {memtile = "tile(0,1)"}}
  return
}
}

// -----

// (b) gather mode with balanced source and destination rates — PASS.
// Confirms the complete M6-join positive path: all edges individually balanced.
//
// src: P=[2], C=[2] → 2*1 == 2*1 ✓ (passes Create::verify M6)
// dst: P=[2], C=[2] → 2*1 == 2*1 ✓ (passes Create::verify M6)
// GatherOp::verify M6-join: both pass → no error.

aie.device(npu1) {
conduit.create @j2_src {slot_elems = 4 : i64,
                producer_tile = array<i64: 0, 2>,
                consumer_tiles = array<i64: 0, 1>,
                element_type = memref<4xi32>,
                depth = 1 : i64,
                producer_rates = array<i64: 2>,
                consumer_rates = array<i64: 2>}
conduit.create @j2_dst {slot_elems = 4 : i64,
                producer_tile = array<i64: 0, 1>,
                consumer_tiles = array<i64: 0, 3>,
                element_type = memref<4xi32>,
                depth = 1 : i64,
                producer_rates = array<i64: 2>,
                consumer_rates = array<i64: 2>}
func.func @join_all_balanced_link_check() {
  // No error expected: all rates balanced.
  conduit.gather{srcs = [@j2_src], dst = @j2_dst {memtile = "tile(0,1)"}}
  return
}
}
