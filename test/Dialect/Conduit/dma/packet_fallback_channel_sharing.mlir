// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// P2-A Step 3.5c: Channel sharing — two mode=any packet fallbacks share one
// physical MM2S channel on the producer tile.
//
// Both fallback conduits share MM2S channel 0 (the first packet-mode channel
// found by Step 3.5c).  Only ONE MM2S physical channel is consumed by both.
//
// Topology: pkt_a and pkt_b fill both MM2S channels as packet-mode.
// fallback1 shares ch 0; fallback2 also shares ch 0.
// DMA BD chains on (0,3): MM2S 0 is used by pkt_a, fallback1, and fallback2
// (multiplexed via different packet flow IDs).
//
// Expected: 4 aie.packet_flow ops (IDs 1-4).
// aie.mem for (0,3): MM2S 0 has 3 BD chains (pkt_a, fallback1, fallback2);
//                    MM2S 1 has 1 BD chain (pkt_b).

// CHECK-LABEL: module @pkt_fallback_channel_sharing
// CHECK:       aie.packet_flow(1)
// CHECK:       aie.packet_flow(2)
// CHECK:       aie.packet_flow(3)
// CHECK:       aie.packet_flow(4)
// CHECK:       aie.mem(%{{.*}}tile_0_3
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_start(MM2S, 1
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_start(MM2S, 0
// CHECK-NOT:   conduit.create

module @pkt_fallback_channel_sharing {
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %t23 = aie.tile(2, 3)
    %t33 = aie.tile(3, 3)
    %t15 = aie.tile(1, 5)
    %t25 = aie.tile(2, 5)

    // pkt_a: MM2S ch 0 designated as packet-mode.
    conduit.create @pkt_a {                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}
    // pkt_b: MM2S ch 1 designated as packet-mode.
    conduit.create @pkt_b {                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}

    // fallback1: circuit exhausted; Step 3.5c finds ch 0 (packet-mode).
    // Shares ch 0 with pkt_a. Flow ID 2.
    conduit.create @fallback1 {                    element_type = memref<4xi32>, depth = 1 : i64
                    }

    // fallback2: Step 3.5c finds ch 0 again (first packet-mode channel).
    // Shares ch 0 with pkt_a and fallback1. Flow ID 3. Only 1 MM2S consumed.
    conduit.create @fallback2 {                    element_type = memref<4xi32>, depth = 1 : i64
                    }

    %core03 = aie.core(%t03) {
      conduit.acquire {name = @pkt_a, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @pkt_b, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @fallback1, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @fallback2, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core23 = aie.core(%t23) {
      conduit.acquire {name = @pkt_a, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core33 = aie.core(%t33) {
      conduit.acquire {name = @pkt_b, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core15 = aie.core(%t15) {
      conduit.acquire {name = @fallback1, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core25 = aie.core(%t25) {
      conduit.acquire {name = @fallback2, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
  }
}
