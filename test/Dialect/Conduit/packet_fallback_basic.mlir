// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// P2-A Step 3.5: Packet DMA fallback — basic case.
//
// Producer tile (0,3) has 2 MM2S channels.  Two explicit packet-mode
// conduits consume both physical channels (marking them as packet-mode in
// PacketChannelState).  A third conduit with routing_mode="any" triggers
// Step 3.5: no free circuit channels exist, but Step 3.5c finds an existing
// packet-mode channel and shares it — emitting a second packet flow on the
// same physical MM2S channel with a new flow ID.
//
// Topology (npu1, 4 columns):
//   pkt_a: (0,3) → (2,3)  routing_mode = "packet"  MM2S ch 0  flow ID 0
//   pkt_b: (0,3) → (3,3)  routing_mode = "packet"  MM2S ch 1  flow ID 1
//   fallback: (0,3) → (1,5)  routing_mode = "any"
//             → circuit exhausted; ch 0 is packet-mode → shares ch 0
//             → emits packet_flow(2) with new ID
//
// Expected output:
//   aie.packet_flow(0) for pkt_a
//   aie.packet_flow(1) for pkt_b
//   aie.packet_flow(2) for fallback (Step 3.5 shared channel 0)
//   No circuit aie.flow ops
//   No "no DMA resources" error

// CHECK-LABEL: module @pkt_fallback_basic
// CHECK:       aie.packet_flow(0)
// CHECK:       aie.packet_flow(1)
// CHECK:       aie.packet_flow(2)
// CHECK-NOT:   aie.flow(
// CHECK-NOT:   conduit.create

module @pkt_fallback_basic {
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %t23 = aie.tile(2, 3)
    %t33 = aie.tile(3, 3)
    %t15 = aie.tile(1, 5)

    // pkt_a: explicit packet mode — MM2S ch 0 designated packet-mode.
    conduit.create {name = "pkt_a", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 2, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}

    // pkt_b: explicit packet mode — MM2S ch 1 designated packet-mode.
    conduit.create {name = "pkt_b", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 3, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}

    // fallback: mode=any; both MM2S channels allocated (packet-mode ch 0 and
    // ch 1); Step 3.5c finds existing packet-mode ch 0, shares it.
    conduit.create {name = "fallback", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 5>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "any"}

    %core03 = aie.core(%t03) { aie.end }
    %core23 = aie.core(%t23) { aie.end }
    %core33 = aie.core(%t33) { aie.end }
    %core15 = aie.core(%t15) { aie.end }
  }
}
