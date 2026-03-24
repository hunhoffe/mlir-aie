// RUN: aie-opt --conduit-check-pairing %s 2>&1 | FileCheck %s
//
// P1-C: Cascade pairing check — valid case.
//
// A matched conduit.put_cascade / conduit.get_cascade pair (same conduit name)
// in producer and consumer cores respectively.  The M9 pairing check should
// emit no warnings.
//
// CHECK-NOT: warning
// CHECK-NOT: M9

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    conduit.create @cas {capacity = 1 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}

    // Producer core: put_cascade matched by get_cascade below.
    aie.core(%tile03) {
      %v = arith.constant dense<42> : vector<16xi32>
      conduit.put_cascade @cas (%v : vector<16xi32>)
      aie.end
    }

    // Consumer core: get_cascade matches the put_cascade above.
    aie.core(%tile13) {
      %r = conduit.get_cascade @cas : vector<16xi32>
      aie.end
    }
  }
}
