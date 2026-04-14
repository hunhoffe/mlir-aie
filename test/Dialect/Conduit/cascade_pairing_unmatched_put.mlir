// RUN: not aie-opt --aie-check-cascade-pairing %s 2>&1 | FileCheck %s
//
// P1-C: Cascade pairing check — unmatched aie.put_cascade.
//
// An aie.put_cascade in a producer core with no corresponding
// aie.get_cascade in any consumer core, and no aie.cascade_flow naming
// the producer tile as source.  The AIE cascade pairing check should
// emit a warning/error on the put_cascade.
//
// After cascade migration (#27), conduit.put_cascade / conduit.get_cascade
// no longer exist.  Pass A/B emit aie.put_cascade / aie.get_cascade
// directly.  The --aie-check-cascade-pairing pass (AIE dialect) handles
// the structural pairing check.
//
// CHECK: put_cascade

module {
  aie.device(npu2) {
    %tile03 = aie.tile(0, 3)

    conduit.create @cas {slot_elems = 1 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}

    // Producer core: aie.put_cascade with NO matching aie.get_cascade anywhere
    // and no aie.cascade_flow for this tile.
    // npu2 (AIE2): cascade width = 512 bits → vector<16xi32> is valid.
    aie.core(%tile03) {
      %v = arith.constant dense<7> : vector<16xi32>
      aie.put_cascade(%v : vector<16xi32>) {conduit_channel = @cas}
      aie.end
    }
  }
}
