// RUN: aie-opt --conduit-to-dma --verify-diagnostics %s
//
// P2-A Step 3.5b: Packet DMA fallback — BD budget exhausted on producer.
//
// Two explicit packet conduits fill both MM2S channels on (0,3) (packet-mode).
// A mode=any conduit attempts Step 3.5, but finds the BD budget exhausted:
// prior packet fallbacks have consumed all 16 BDs on the producer tile.
// (tileBDUsed[prodTile] = 16 after the first successful fallback with depth=16)
//
// Topology (npu1):
//   pkt_a:    (0,3) → (2,3) routing_mode="packet" MM2S ch 0  [uses 1 BD]
//   pkt_b:    (0,3) → (3,3) routing_mode="packet" MM2S ch 1  [uses 1 BD]
//   fallback1: (0,3) → (1,5) routing_mode="any" depth=14
//             → Step 3.5 succeeds; tileBDUsed[prodTile] += 14
//   fallback2: (0,3) → (1,4) routing_mode="any" depth=3
//             → Step 3.5b: prodBDTotal(16) - prodBDUsed(14) = 2 < depth(3) → fail
//             → Step 4 error: no DMA resources
//
// Expected: Step 4 error for 'fallback2'.

module @pkt_fallback_bd_exhaustion {
  // expected-error @+1 {{conduit-to-dma: no DMA resources available for conduit 'fallback2'}}
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %t23 = aie.tile(2, 3)
    %t33 = aie.tile(3, 3)
    %t14 = aie.tile(1, 4)
    %t15 = aie.tile(1, 5)

    // Two packet conduits fill both MM2S channels as packet-mode.
    conduit.create @pkt_a {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 2, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}
    conduit.create @pkt_b {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 3, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}

    // First mode=any fallback: depth=14 consumes 14 BD slots on (0,3).
    // tileBDUsed[(0,3)] = 14 after this.
    conduit.create @fallback1 {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 5>,
                    element_type = memref<4xi32>, depth = 14 : i64
                    }

    // Second mode=any fallback: depth=3 requires 3 BDs, but only 2 remain.
    // Step 3.5b: prodBDTotal(16) - prodBDUsed(14) = 2 < depth(3) → failure.
    conduit.create @fallback2 {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 4>,
                    element_type = memref<4xi32>, depth = 3 : i64
                    }

    %core03 = aie.core(%t03) { aie.end }
    %core23 = aie.core(%t23) { aie.end }
    %core33 = aie.core(%t33) { aie.end }
    %core14 = aie.core(%t14) { aie.end }
    %core15 = aie.core(%t15) { aie.end }
  }
}
