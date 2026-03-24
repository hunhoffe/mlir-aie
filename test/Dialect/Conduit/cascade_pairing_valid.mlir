// RUN: aie-opt --conduit-check-pairing %s 2>&1 | FileCheck %s
//
// P1-C: Cascade pairing check — valid case.
//
// After cascade migration (#27), conduit.put_cascade / conduit.get_cascade
// no longer exist.  Core-body cascade ops are aie.put_cascade /
// aie.get_cascade.  The --conduit-check-pairing pass no longer checks cascade
// pairing (deferred to --aie-check-cascade-pairing); this test verifies that
// --conduit-check-pairing emits no spurious warnings on a valid cascade design
// with aie.put_cascade / aie.get_cascade.
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

    // Producer core: aie.put_cascade matched by aie.get_cascade below.
    aie.core(%tile03) {
      %v = arith.constant dense<42> : vector<16xi32>
      aie.put_cascade(%v : vector<16xi32>)
      aie.end
    }

    // Consumer core: aie.get_cascade matches the put_cascade above.
    aie.core(%tile13) {
      %r = aie.get_cascade() : vector<16xi32>
      aie.end
    }
  }
}
