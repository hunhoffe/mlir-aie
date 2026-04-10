// RUN: aie-opt -split-input-file -verify-diagnostics %s | FileCheck %s
//
// Lit tests for conduit.scatter, conduit.gather (successors to
// conduit.distribute / conduit.join / conduit.forward).
// Covers: valid ops, structural verifier errors.

// -----

// Valid conduit.scatter: 1 src → 2 dsts.
// CHECK-LABEL: func.func @valid_scatter
func.func @valid_scatter() {
  // CHECK: conduit.scatter{src = @src, dsts = [@dst0, @dst1]
  // CHECK-SAME: memtile = "tile(0,1)"
  conduit.scatter{src = @src, dsts = [@dst0, @dst1] {memtile = "tile(0,1)"}}
  return
}

// -----

// Valid conduit.gather: 2 srcs → 1 dst with offsets.
// CHECK-LABEL: func.func @valid_gather
func.func @valid_gather() {
  // CHECK: conduit.gather{srcs = [@src0, @src1], dst = @dst
  // CHECK-SAME: memtile = "tile(0,1)"
  conduit.gather{srcs = [@src0, @src1], dst = @dst {memtile = "tile(0,1)",
                  offsets = array<i64: 0, 512>}}
  return
}

// -----

// Valid conduit.scatter with single dst (replaces conduit.forward 1:1 relay).
// CHECK-LABEL: func.func @valid_scatter_single_dst
func.func @valid_scatter_single_dst() {
  // CHECK: conduit.scatter{src = @in, dsts = [@out]
  // CHECK-SAME: memtile = "tile(0,1)"
  conduit.scatter{src = @in, dsts = [@out] {memtile = "tile(0,1)"}}
  return
}

// -----

// Invalid conduit.scatter: empty dsts.
func.func @invalid_scatter_empty_dsts() {
  // expected-error@+1 {{'conduit.scatter' op scatter requires at least 1 dst, got 0}}
  conduit.scatter{src = @src, dsts = [] {memtile = "tile(0,1)"}}
  return
}

// -----

// Invalid conduit.gather: empty srcs.
func.func @invalid_gather_empty_srcs() {
  // expected-error@+1 {{'conduit.gather' op gather requires at least 1 src, got 0}}
  conduit.gather{srcs = [], dst = @dst {memtile = "tile(0,1)"}}
  return
}
