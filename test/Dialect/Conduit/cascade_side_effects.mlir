// RUN: aie-opt --allow-unregistered-dialect --cse %s | FileCheck %s
//
// Verify that aie.put_cascade and aie.get_cascade have correct memory
// side effects declared, which prevents CSE from deduplicating them.
//
// After cascade migration (#27), conduit.put_cascade / conduit.get_cascade
// no longer exist.  Pass A/B emit aie.put_cascade / aie.get_cascade directly.
// This test verifies the AIE ops have the same CSE-blocking side effects.
//
// Two identical put_cascade calls (same value) must both survive CSE.
// Two identical get_cascade calls must both survive CSE.
//
// Must be inside aie.device(npu2) + aie.core so the AIE verifier
// accepts vector<16xi32> (512 bits = AIE2 cascade width).

module {
  aie.device(npu2) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    // CHECK: aie.core
    aie.core(%tile03) {
      %v = arith.constant dense<1> : vector<16xi32>
      // CHECK: aie.put_cascade
      aie.put_cascade(%v : vector<16xi32>)
      // CHECK: aie.put_cascade
      aie.put_cascade(%v : vector<16xi32>)
      aie.end
    }

    aie.core(%tile13) {
      // CHECK: aie.get_cascade
      %a = aie.get_cascade() : vector<16xi32>
      // CHECK: aie.get_cascade
      %b = aie.get_cascade() : vector<16xi32>
      // Use results to prevent trivial DCE of unused values.
      "test.use"(%a, %b) : (vector<16xi32>, vector<16xi32>) -> ()
      aie.end
    }
  }
}
