// RUN: aie-opt --aie-check-cascade-pairing %s 2>&1 | FileCheck %s
//
// P1-C: Cascade pairing check — unmatched aie.get_cascade.
//
// After cascade migration (#27), conduit.get_cascade no longer exists.
// The AIE dialect --aie-check-cascade-pairing pass checks aie.cascade_flow
// vs. core body aie.put_cascade / aie.get_cascade pairing.
//
// An aie.get_cascade in a consumer core with no corresponding aie.put_cascade
// in any producer core and no aie.cascade_flow naming this tile as destination.
//
// CHECK: get_cascade

module {
  aie.device(npu1) {
    %tile13 = aie.tile(1, 3)

    conduit.create @cas {capacity = 1 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}

    // Consumer core: aie.get_cascade with NO matching aie.put_cascade anywhere.
    aie.core(%tile13) {
      %r = aie.get_cascade() : vector<16xi32>
      aie.end
    }
  }
}
