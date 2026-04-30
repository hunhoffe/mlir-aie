// RUN: aie-opt -split-input-file -verify-diagnostics %s | FileCheck %s
//
// Lit tests for conduit.scatter, conduit.gather (successors to
// conduit.distribute / conduit.join / conduit.forward).
// Covers: valid ops, structural verifier errors.

// -----

// Valid conduit.scatter: 1 src → 2 dsts.
// CHECK-LABEL: func.func @valid_scatter
func.func @valid_scatter() {
  // CHECK: %[[MT:.*]] = aie.tile(0, 1)
  // CHECK: conduit.scatter{src = @src, dsts = [@dst0, @dst1]
  // CHECK-SAME: memtile = %[[MT]]
  %mt = aie.tile(0, 1)
  conduit.scatter{src = @src, dsts = [@dst0, @dst1], memtile = %mt}
  return
}

// -----

// Valid conduit.gather: 2 srcs → 1 dst with offsets.
// CHECK-LABEL: func.func @valid_gather
func.func @valid_gather() {
  // CHECK: %[[MT:.*]] = aie.tile(0, 1)
  // CHECK: conduit.gather{srcs = [@src0, @src1], dst = @dst
  // CHECK-SAME: memtile = %[[MT]]
  %mt = aie.tile(0, 1)
  conduit.gather{srcs = [@src0, @src1], dst = @dst, memtile = %mt,
                  offsets = [0, 512]}
  return
}

// -----

// Valid conduit.scatter with single dst (replaces conduit.forward 1:1 relay).
// CHECK-LABEL: func.func @valid_scatter_single_dst
func.func @valid_scatter_single_dst() {
  // CHECK: %[[MT:.*]] = aie.tile(0, 1)
  // CHECK: conduit.scatter{src = @in, dsts = [@out]
  // CHECK-SAME: memtile = %[[MT]]
  %mt = aie.tile(0, 1)
  conduit.scatter{src = @in, dsts = [@out], memtile = %mt}
  return
}

// -----

// Invalid conduit.scatter: empty dsts.
func.func @invalid_scatter_empty_dsts() {
  %mt = aie.tile(0, 1)
  // expected-error@+1 {{'conduit.scatter' op scatter requires at least 1 dst, got 0}}
  conduit.scatter{src = @src, dsts = [], memtile = %mt}
  return
}

// -----

// Invalid conduit.gather: empty srcs.
func.func @invalid_gather_empty_srcs() {
  %mt = aie.tile(0, 1)
  // expected-error@+1 {{'conduit.gather' op gather requires at least 1 src, got 0}}
  conduit.gather{srcs = [], dst = @dst, memtile = %mt}
  return
}
