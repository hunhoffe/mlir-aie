// RUN: aie-opt --conduit-to-dma --aie-lower-cascade-flows %s | FileCheck %s
//
// P2-A Step 3.5: Mixed cascade + packet DMA fallback design.
//
// Cascade is preferred when eligible.  Non-cascade conduits from producer
// tile (0,3) use DMA.  When both MM2S channels are full (packet-mode),
// a mode=any conduit triggers Step 3.5 and shares the packet channel.
//
// Topology:
//   cascade_ab: (0,3) → (0,4)  routing_mode="cascade"  (no DMA/locks)
//   pkt_a:      (0,3) → (2,3)  routing_mode="packet"   MM2S ch 0
//   pkt_b:      (0,3) → (3,3)  routing_mode="packet"   MM2S ch 1
//   fallback:   (0,3) → (1,5)  routing_mode="any"      → packet fallback ch 0
//
// Expected:
//   aie.cascade_flow for cascade_ab
//   aie.configure_cascade (after --aie-lower-cascade-flows)
//   3 aie.packet_flow ops (pkt_a, pkt_b, fallback)
//   0 circuit aie.flow ops

// CHECK-LABEL: module @pkt_fallback_with_cascade
// CHECK:       aie.packet_flow(0)
// CHECK:       aie.packet_flow(1)
// CHECK:       aie.packet_flow(2)
// CHECK:       aie.configure_cascade
// CHECK-NOT:   conduit.create

module @pkt_fallback_with_cascade {
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %t13 = aie.tile(1, 3)
    %t23 = aie.tile(2, 3)
    %t33 = aie.tile(3, 3)
    %t15 = aie.tile(1, 5)

    // Cascade conduit — zero DMA, zero locks.
    // (0,3) → (1,3): East direction (col+1, same row). Valid cascade topology.
    conduit.create @cascade_ab {                    element_type = memref<1xvector<16xi32>>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}

    // Packet conduits — fill both MM2S channels on (0,3) as packet-mode.
    conduit.create @pkt_a {                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}
    conduit.create @pkt_b {                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}

    // mode=any: circuit exhausted → Step 3.5 shares packet ch 0.
    conduit.create @fallback {                    element_type = memref<4xi32>, depth = 1 : i64
                    }

    %core03 = aie.core(%t03) {
      conduit.acquire {name = @pkt_a, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @pkt_b, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @fallback, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      %v = arith.constant dense<0> : vector<16xi32>
      aie.put_cascade(%v : vector<16xi32>) {conduit_channel = @cascade_ab}
      aie.end
    }
    %core13 = aie.core(%t13) {
      %r = aie.get_cascade() {conduit_channel = @cascade_ab} : vector<16xi32>
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
      conduit.acquire {name = @fallback, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
  }
}
