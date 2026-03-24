// RUN: aie-opt --conduit-check-pairing %s 2>&1 | FileCheck %s
//
// P1-C: Cascade pairing check — two aie.get_cascade ops (valid from
// --conduit-check-pairing perspective; cascade hardware supports exactly
// one consumer, so this is actually invalid at hardware level, but
// --conduit-check-pairing no longer checks cascade pairing after migration #27).
//
// After cascade migration (#27), conduit.put_cascade / conduit.get_cascade
// no longer exist; the cascade pairing check is deferred to
// --aie-check-cascade-pairing.  This test verifies --conduit-check-pairing
// does not crash or emit spurious errors on designs with aie.get_cascade.
//
// CHECK-NOT: M9

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)
    %tile23 = aie.tile(2, 3)

    conduit.create @cas {capacity = 1 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}

    // Producer core.
    aie.core(%tile03) {
      %v = arith.constant dense<5> : vector<16xi32>
      aie.put_cascade(%v : vector<16xi32>)
      aie.end
    }

    // First consumer: valid aie.get_cascade.
    aie.core(%tile13) {
      %r = aie.get_cascade() : vector<16xi32>
      aie.end
    }

    // Second consumer: duplicate aie.get_cascade (hardware-invalid,
    // but --conduit-check-pairing does not check this after migration #27;
    // use --aie-check-cascade-pairing for this validation).
    aie.core(%tile23) {
      %r2 = aie.get_cascade() : vector<16xi32>
      aie.end
    }
  }
}
