// RUN: aie-opt --conduit-check-pairing %s 2>&1 | FileCheck %s
//
// P1-C: Cascade pairing check — unmatched get_cascade.
//
// A conduit.get_cascade with no corresponding conduit.put_cascade in any
// core body.  The M9 check should emit a warning on the get_cascade.
//
// CHECK: warning
// CHECK: unmatched conduit.get_cascade: no corresponding put_cascade found in any producer core

module {
  aie.device(npu1) {
    %tile13 = aie.tile(1, 3)

    conduit.create {name = "cas", capacity = 1 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}

    // Consumer core: get_cascade with NO matching put_cascade anywhere.
    aie.core(%tile13) {
      %r = conduit.get_cascade "cas" : vector<16xi32>
      aie.end
    }
  }
}
