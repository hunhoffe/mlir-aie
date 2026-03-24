// RUN: aie-opt -split-input-file -verify-diagnostics %s | FileCheck %s
//
// Lit tests for conduit.distribute, conduit.join, conduit.forward.
// Covers: valid ops, structural verifier errors.

// -----

// Valid conduit.distribute: 1 src → 2 dsts.
// CHECK-LABEL: func.func @valid_distribute
func.func @valid_distribute() {
  // CHECK: conduit.distribute
  // CHECK-SAME: dsts = ["dst0", "dst1"]
  // CHECK-SAME: memtile = "tile(0,1)"
  // CHECK-SAME: srcs = ["src"]
  conduit.distribute {srcs = ["src"], dsts = ["dst0", "dst1"],
                      memtile = "tile(0,1)"}
  return
}

// -----

// Valid conduit.join: 2 srcs → 1 dst with offsets.
// CHECK-LABEL: func.func @valid_join
func.func @valid_join() {
  // CHECK: conduit.join
  // CHECK-SAME: dsts = ["dst"]
  // CHECK-SAME: memtile = "tile(0,1)"
  // CHECK-SAME: srcs = ["src0", "src1"]
  conduit.join {srcs = ["src0", "src1"], dsts = ["dst"],
                memtile = "tile(0,1)",
                offsets = array<i64: 0, 512>}
  return
}

// -----

// Valid conduit.forward: 1 src → 1 dst.
// CHECK-LABEL: func.func @valid_forward
func.func @valid_forward() {
  // CHECK: conduit.forward
  // CHECK-SAME: dsts = ["out"]
  // CHECK-SAME: memtile = "tile(0,1)"
  // CHECK-SAME: srcs = ["in"]
  conduit.forward {srcs = ["in"], dsts = ["out"],
                   memtile = "tile(0,1)"}
  return
}

// -----

// Invalid conduit.distribute: empty dsts.
func.func @invalid_distribute_empty_dsts() {
  // expected-error@+1 {{'conduit.distribute' op distribute requires at least 1 dst, got 0}}
  conduit.distribute {srcs = ["src"], dsts = [], memtile = "tile(0,1)"}
  return
}

// -----

// Invalid conduit.join: empty srcs.
func.func @invalid_join_empty_srcs() {
  // expected-error@+1 {{'conduit.join' op join requires at least 1 src, got 0}}
  conduit.join {srcs = [], dsts = ["dst"], memtile = "tile(0,1)"}
  return
}
