// RUN: aie-opt --allow-unregistered-dialect --cse %s | FileCheck %s
//
// Verify that conduit.put_cascade and conduit.get_cascade have correct memory
// side effects declared (MemWrite and MemRead respectively), which prevents
// CSE from deduplicating them.
//
// Without side effects declared, CSE could eliminate the second put_cascade
// or second get_cascade (treating them as pure duplicate subexpressions).
// With MemWrite / MemRead, CSE must keep all occurrences.
//
// Two identical put_cascade calls (same name, same value) must both survive.
// Two identical get_cascade calls (same name) must both survive.

// CHECK-LABEL: func.func @test_put_cascade_not_cse_eliminated
func.func @test_put_cascade_not_cse_eliminated() {
  %v = arith.constant dense<1> : vector<16xi32>
  // CHECK: conduit.put_cascade
  conduit.put_cascade "cas" (%v : vector<16xi32>)
  // CHECK: conduit.put_cascade
  conduit.put_cascade "cas" (%v : vector<16xi32>)
  return
}

// CHECK-LABEL: func.func @test_get_cascade_not_cse_eliminated
func.func @test_get_cascade_not_cse_eliminated() {
  // CHECK: conduit.get_cascade
  %a = conduit.get_cascade "cas" : vector<16xi32>
  // CHECK: conduit.get_cascade
  %b = conduit.get_cascade "cas" : vector<16xi32>
  // Use results to prevent trivial DCE of unused values.
  // (The effect declarations ensure CSE cannot merge the two get_cascade ops.)
  "test.use"(%a, %b) : (vector<16xi32>, vector<16xi32>) -> ()
  return
}
