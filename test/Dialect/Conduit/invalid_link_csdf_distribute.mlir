// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// Tests for conduit.scatter with CSDF rate annotations.
//
// This file covers the skip path and pass path for the Level 2 composed-consume
// check (Denolf Eq. 48) in ScatterOp::verify() / checkDistributeComposedConsume().
//
// Background: Create::verify() runs M6/M7 on each conduit independently.
// ScatterOp::verify() runs a second pass (checkDistributeComposedConsume) that is
// unique to scatter: it computes the composed (minimum) consume across
// ALL N destination consumers to verify the source buffer can accommodate the
// worst-case occupancy when a slow consumer gates buffer reuse.
//
// Note on Denolf Eq. 48 negative test: triggering a genuine composed-consume
// overflow that is not caught first by conduit.create M7 is non-trivial because
// the constraints are coupled (see comment in invalid_link_csdf_distribute.mlir).
// The negative path is indirectly covered by distribute_csdf_capacity.mlir
// (Section 3) which documents the verification order.
//
// This file tests:
//   (a) scatter with unannotated destinations → Level 2 skipped → PASS
//   (b) scatter with 2 symmetric consumers (both annotated, same rate) → PASS
//   (c) scatter with single consumer → Level 2 skip (size < 2) → PASS

// -----

// (a) scatter with unannotated destinations — Level 2 skipped.
// checkDistributeComposedConsume: allDstsHaveRates=false → return success().

aie.device(npu1) {
conduit.create @src_skip {slot_elems = 4 : i64,
                element_type = memref<4xi32>,
                depth = 1 : i64,
                producer_rates = array<i64: 1>,
                consumer_rates = array<i64: 1>}
conduit.create @dst0_skip {slot_elems = 4 : i64,
                element_type = memref<4xi32>,
                depth = 1 : i64}
conduit.create @dst1_skip {slot_elems = 4 : i64,
                element_type = memref<4xi32>,
                depth = 1 : i64}
func.func @distribute_unannotated_skip() {
  // No error: dst conduits lack rate annotations → skip Level 2.
  conduit.scatter{src = @src_skip, dsts = [@dst0_skip, @dst1_skip] {memtile = "tile(0,1)"}}
  return
}
}

// -----

// (b) scatter with 2 symmetric annotated consumers — Level 2 PASS.
// Both dst consumers drain at 1/step. Composed consume = 1/step.
// src P=[1], slot_elems =4: H=1: cumProd=1, composed=1, occ=0 ≤ 4 → PASS.

aie.device(npu1) {
conduit.create @src_sym {slot_elems = 4 : i64,
                element_type = memref<4xi32>,
                depth = 4 : i64,
                producer_rates = array<i64: 1>,
                consumer_rates = array<i64: 1>}
conduit.create @dst0_sym {slot_elems = 4 : i64,
                element_type = memref<4xi32>,
                depth = 4 : i64,
                producer_rates = array<i64: 1>,
                consumer_rates = array<i64: 1>}
conduit.create @dst1_sym {slot_elems = 4 : i64,
                element_type = memref<4xi32>,
                depth = 4 : i64,
                producer_rates = array<i64: 1>,
                consumer_rates = array<i64: 1>}
func.func @distribute_symmetric_pass() {
  // No error: composed consume = per-consumer = 1, source capacity sufficient.
  conduit.scatter{src = @src_sym, dsts = [@dst0_sym, @dst1_sym] {memtile = "tile(0,1)"}}
  return
}
}

// -----

// (c) Single-consumer scatter — Level 2 skip (dstConsRates.size() < 2).
// checkDistributeComposedConsume returns success immediately for N < 2.

aie.device(npu1) {
conduit.create @src_one {slot_elems = 4 : i64,
                element_type = memref<4xi32>,
                depth = 4 : i64,
                producer_rates = array<i64: 1>,
                consumer_rates = array<i64: 1>}
conduit.create @dst0_one {slot_elems = 4 : i64,
                element_type = memref<4xi32>,
                depth = 4 : i64,
                producer_rates = array<i64: 1>,
                consumer_rates = array<i64: 1>}
func.func @distribute_single_consumer_skip() {
  // No error: single destination → Level 2 skip (per-edge check only).
  conduit.scatter{src = @src_one, dsts = [@dst0_one] {memtile = "tile(0,1)"}}
  return
}
}
