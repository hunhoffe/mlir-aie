// RUN: aie-opt --conduit-to-dma --verify-diagnostics %s
//
// P2-A Step 3.5d: Convergence hazard warning — two packet flows to same
// consumer tile through the same physical MM2S channel.
//
// Two explicit packet conduits (pkt_a, pkt_b) occupy both MM2S channels on
// (0,3) as packet-mode.  Two mode=any conduits (fallback1, fallback2) both
// target (2,3) and both fall back to packet mode via channel 0.
//
// fallback1: channel 0 → (2,3) — occupancy records (ch0 → (2,3)); no warning.
// fallback2: channel 0 → (2,3) — Step 3.5d finds existing (ch0 → (2,3)) with
//            same destination → ordering hazard warning emitted.
//
// Expected: 4 aie.packet_flow ops; 1 ordering hazard warning.

module @pkt_fallback_convergence_warning {
  // expected-warning @+1 {{conduit-to-dma: packet DMA ordering hazard: conduit 'fallback2'}}
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %t23 = aie.tile(2, 3)
    %t33 = aie.tile(3, 3)

    // pkt_a and pkt_b: explicit packet, filling MM2S ch 0 and ch 1.
    conduit.create {name = "pkt_a", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 3, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "pkt_b", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 3, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}

    // fallback1: circuit exhausted; Step 3.5c finds ch 0; no prior (ch0→(2,3)).
    // Records occupancy. No warning.
    conduit.create {name = "fallback1", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 2, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64
                    }

    // fallback2: Step 3.5d finds (ch0 → (2,3)) already recorded → hazard.
    // Warning emitted on aie.device op (annotation above).
    conduit.create {name = "fallback2", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 2, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64
                    }

    %core03 = aie.core(%t03) { aie.end }
    %core23 = aie.core(%t23) { aie.end }
    %core33 = aie.core(%t33) { aie.end }
  }
}
