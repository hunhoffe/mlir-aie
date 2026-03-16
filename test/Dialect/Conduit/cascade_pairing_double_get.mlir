// RUN: aie-opt --conduit-check-pairing %s 2>&1 | FileCheck %s
//
// P1-C: Cascade pairing check — two get_cascade ops for the same conduit name.
//
// Two cores both issue conduit.get_cascade "cas".  Cascade hardware is
// point-to-point (single consumer); multiple get_cascade for the same name
// is ambiguous.  The M9 check should emit a warning on each get_cascade.
//
// CHECK: warning
// CHECK: ambiguous cascade: multiple get_cascade ops for the same conduit name

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)
    %tile23 = aie.tile(2, 3)

    conduit.create {name = "cas", capacity = 1 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    depth = 1 : i64,
                    routing_mode = "cascade"}

    // Producer core.
    aie.core(%tile03) {
      %v = arith.constant dense<5> : vector<16xi32>
      conduit.put_cascade "cas" (%v : vector<16xi32>)
      aie.end
    }

    // First consumer: valid get_cascade.
    aie.core(%tile13) {
      %r = conduit.get_cascade "cas" : vector<16xi32>
      aie.end
    }

    // Second consumer: duplicate get_cascade for the same conduit name.
    aie.core(%tile23) {
      %r2 = conduit.get_cascade "cas" : vector<16xi32>
      aie.end
    }
  }
}
