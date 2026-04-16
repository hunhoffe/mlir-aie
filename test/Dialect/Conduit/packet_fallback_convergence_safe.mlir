// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// P2-A Step 3.5d: Convergence safe — two packet flows to DIFFERENT consumers
// through the same physical MM2S channel on the same producer tile.
//
// fallback1 → (2,3) and fallback2 → (3,3): same MM2S channel (ch 0),
// different destinations.  Step 3.5d finds no convergence hazard (different
// destination tiles), so no warning is emitted.
//
// Expected: 4 aie.packet_flow ops; 0 warnings about ordering hazard.

// CHECK-LABEL: module @pkt_fallback_convergence_safe
// CHECK:       aie.packet_flow(0)
// CHECK:       aie.packet_flow(1)
// CHECK:       aie.packet_flow(2)
// CHECK:       aie.packet_flow(3)
// CHECK-NOT:   warning
// CHECK-NOT:   conduit.create

module @pkt_fallback_convergence_safe {
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %t15 = aie.tile(1, 5)
    %t23 = aie.tile(2, 3)
    %t33 = aie.tile(3, 3)

    // pkt_a, pkt_b: fill MM2S ch 0 and ch 1 as packet-mode.
    conduit.create @pkt_a {                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}
    conduit.create @pkt_b {                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}

    // fallback1: mode=any → (2,3); Step 3.5c picks ch 0. Records (ch0→(2,3)).
    conduit.create @fallback1 {                    element_type = memref<4xi32>, depth = 1 : i64
                    }

    // fallback2: mode=any → (3,3); Step 3.5c picks ch 0 (same channel).
    // Step 3.5d: (ch0→(2,3)) exists, but (3,3) ≠ (2,3) → no hazard. No warn.
    conduit.create @fallback2 {                    element_type = memref<4xi32>, depth = 1 : i64
                    }

    %core03 = aie.core(%t03) {
      conduit.acquire {name = @pkt_a, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @pkt_b, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @fallback1, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @fallback2, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core15 = aie.core(%t15) {
      conduit.acquire {name = @pkt_a, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core23 = aie.core(%t23) {
      conduit.acquire {name = @fallback1, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core33 = aie.core(%t33) {
      conduit.acquire {name = @pkt_b, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @fallback2, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
  }
}
