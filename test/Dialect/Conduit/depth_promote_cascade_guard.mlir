// RUN: aie-opt --conduit-depth-promote %s | FileCheck %s
//
// P0-B: depth_promote_cascade_guard
//
// The cascade stream is a hardware rendezvous register — there is no FIFO,
// so depth-2 cannot be implemented. --conduit-depth-promote must silently
// skip cascade conduits even when every other promotion criterion is satisfied.
//
// This test has one cascade conduit ("cas") with depth=1 inside a loop so
// that criteria 3 (no loop) and 4 (passthrough) would both be satisfied for
// a DMA conduit. If the cascade guard is absent, the pass would promote to
// depth=2, causing silent hardware incorrectness.
//
// Expected output:
//   - conduit.create has routing_mode = #conduit.routing_mode<cascade>
//   - depth stays at 1 (NOT promoted to 2; CHECK-NOT: depth = 2)
//   - No promotion remark (cascade skip is silent)

// The conduit.create line should contain routing_mode = #conduit.routing_mode<cascade> and depth = 1
// CHECK: conduit.create {capacity = 1 : i64, {{.*}}depth = 1 : i64,{{.*}}routing_mode = #conduit.routing_mode<cascade>
// Depth must NOT be promoted to 2.
// CHECK-NOT: depth = 2

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    conduit.create {name = "cas", capacity = 1 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    element_type = memref<1xvector<16xi32>>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}

    // Producer core — acquire+compute inside a loop so criteria 3+4 are met.
    // The cascade guard (criterion 0) must still prevent promotion.
    aie.core(%tile03) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %v = arith.constant dense<1> : vector<16xi32>
        conduit.put_cascade "cas" (%v : vector<16xi32>)
      }
      aie.end
    }

    // Consumer core.
    aie.core(%tile13) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %r = conduit.get_cascade "cas" : vector<16xi32>
      }
      aie.end
    }
  }
}
