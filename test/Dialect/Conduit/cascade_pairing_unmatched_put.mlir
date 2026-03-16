// RUN: aie-opt --conduit-check-pairing %s 2>&1 | FileCheck %s
//
// P1-C: Cascade pairing check — unmatched put_cascade.
//
// A conduit.put_cascade with no corresponding conduit.get_cascade in any
// core body.  The M9 check should emit a warning on the put_cascade.
//
// CHECK: warning
// CHECK: unmatched conduit.put_cascade: no corresponding get_cascade found in any consumer core

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)

    conduit.create {name = "cas", capacity = 1 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    depth = 1 : i64,
                    routing_mode = "cascade"}

    // Producer core: put_cascade with NO matching get_cascade anywhere.
    aie.core(%tile03) {
      %v = arith.constant dense<7> : vector<16xi32>
      conduit.put_cascade "cas" (%v : vector<16xi32>)
      aie.end
    }
  }
}
