// RUN: aie-opt --conduit-to-dma --verify-diagnostics %s
//
// P1-B: cascade depth assertion at Pass C entry.
//
// The hardware cascade stream is a blocking register (rendezvous channel) —
// it has no FIFO. A cascade conduit with depth != 1 cannot be implemented
// in hardware; Pass C must reject it with a hard error at Phase 1 (collect).
//
// Expected: emitError fires on the conduit.create with depth=2 and
// routing_mode="cascade". --verify-diagnostics exits 0 when the
// expected-error annotation is matched.

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    // expected-error @+1 {{cascade conduit must have depth = 1; hardware has no FIFO on the cascade stream}}
    conduit.create @cas {slot_elems = 2 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    element_type = memref<1xvector<16xi32>>,
                    depth = 2 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}

    aie.core(%tile03) {
      %v = arith.constant dense<42> : vector<16xi32>
      aie.put_cascade(%v : vector<16xi32>)
      aie.end
    }

    aie.core(%tile13) {
      %r = aie.get_cascade() : vector<16xi32>
      aie.end
    }
  }
}
