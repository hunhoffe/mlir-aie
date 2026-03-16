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
    conduit.create {name = "pkt_a", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 5>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "pkt_b", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 5>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}

    // fallback1: mode=any → (2,3); Step 3.5c picks ch 0. Records (ch0→(2,3)).
    conduit.create {name = "fallback1", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 2, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "any"}

    // fallback2: mode=any → (3,3); Step 3.5c picks ch 0 (same channel).
    // Step 3.5d: (ch0→(2,3)) exists, but (3,3) ≠ (2,3) → no hazard. No warn.
    conduit.create {name = "fallback2", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 3, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "any"}

    %core03 = aie.core(%t03) { aie.end }
    %core15 = aie.core(%t15) { aie.end }
    %core23 = aie.core(%t23) { aie.end }
    %core33 = aie.core(%t33) { aie.end }
  }
}
