// RUN: aie-opt --conduit-to-dma --verify-diagnostics %s
//
// P2-A Step 3.5a: Packet DMA fallback — global ID budget exhausted.
//
// 32 explicit packet conduits from shim tiles consume all 32 flow IDs
// (through Phase 4a).  Then two explicit packet conduits from a compute
// tile fill both of its MM2S channels as packet-mode.  A final conduit
// with routing_mode="any" from the same compute tile triggers Step 3.5:
// Step 3.5c finds an existing packet-mode channel (ch 0), but Step 3.5a
// fires first (remaining() == 0) → returns false → Step 4 error emitted
// on the device op.
//
// Expected: error about packet flow ID exhaustion.

// expected-error @+1 {{packet flow ID exhausted: design requires more than 32 distinct packet flows}}
module @pkt_fallback_id_exhaustion {
  aie.device(xcvc1902) {
    // Shim-produced packet conduits that consume IDs 0-31 via Phase 4a.
    %ts2_0 = aie.tile(2, 0)
    %ts2_1 = aie.tile(2, 1)  %ts2_2 = aie.tile(2, 2)  %ts2_3 = aie.tile(2, 3)
    %ts2_4 = aie.tile(2, 4)  %ts2_5 = aie.tile(2, 5)  %ts2_6 = aie.tile(2, 6)
    %ts2_7 = aie.tile(2, 7)  %ts2_8 = aie.tile(2, 8)
    %ts3_0 = aie.tile(3, 0)
    %ts3_1 = aie.tile(3, 1)  %ts3_2 = aie.tile(3, 2)  %ts3_3 = aie.tile(3, 3)
    %ts3_4 = aie.tile(3, 4)  %ts3_5 = aie.tile(3, 5)  %ts3_6 = aie.tile(3, 6)
    %ts3_7 = aie.tile(3, 7)  %ts3_8 = aie.tile(3, 8)
    %ts6_0 = aie.tile(6, 0)
    %ts6_1 = aie.tile(6, 1)  %ts6_2 = aie.tile(6, 2)  %ts6_3 = aie.tile(6, 3)
    %ts6_4 = aie.tile(6, 4)  %ts6_5 = aie.tile(6, 5)  %ts6_6 = aie.tile(6, 6)
    %ts6_7 = aie.tile(6, 7)  %ts6_8 = aie.tile(6, 8)
    %ts7_0 = aie.tile(7, 0)
    %ts7_1 = aie.tile(7, 1)  %ts7_2 = aie.tile(7, 2)  %ts7_3 = aie.tile(7, 3)
    %ts7_4 = aie.tile(7, 4)  %ts7_5 = aie.tile(7, 5)  %ts7_6 = aie.tile(7, 6)
    %ts7_7 = aie.tile(7, 7)  %ts7_8 = aie.tile(7, 8)
    // Compute tile for the mode=any conduit.
    %tc5_1 = aie.tile(5, 1)  %tc5_3 = aie.tile(5, 3)  %tc5_5 = aie.tile(5, 5)

    // IDs 0-7: shim col 2 → compute col 2
    conduit.create {name = "p00", capacity = 4 : i64, producer_tile = array<i64: 2, 0>, consumer_tiles = array<i64: 2, 1>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p01", capacity = 4 : i64, producer_tile = array<i64: 2, 0>, consumer_tiles = array<i64: 2, 2>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p02", capacity = 4 : i64, producer_tile = array<i64: 2, 0>, consumer_tiles = array<i64: 2, 3>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p03", capacity = 4 : i64, producer_tile = array<i64: 2, 0>, consumer_tiles = array<i64: 2, 4>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p04", capacity = 4 : i64, producer_tile = array<i64: 2, 0>, consumer_tiles = array<i64: 2, 5>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p05", capacity = 4 : i64, producer_tile = array<i64: 2, 0>, consumer_tiles = array<i64: 2, 6>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p06", capacity = 4 : i64, producer_tile = array<i64: 2, 0>, consumer_tiles = array<i64: 2, 7>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p07", capacity = 4 : i64, producer_tile = array<i64: 2, 0>, consumer_tiles = array<i64: 2, 8>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    // IDs 8-15: shim col 3
    conduit.create {name = "p08", capacity = 4 : i64, producer_tile = array<i64: 3, 0>, consumer_tiles = array<i64: 3, 1>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p09", capacity = 4 : i64, producer_tile = array<i64: 3, 0>, consumer_tiles = array<i64: 3, 2>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p10", capacity = 4 : i64, producer_tile = array<i64: 3, 0>, consumer_tiles = array<i64: 3, 3>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p11", capacity = 4 : i64, producer_tile = array<i64: 3, 0>, consumer_tiles = array<i64: 3, 4>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p12", capacity = 4 : i64, producer_tile = array<i64: 3, 0>, consumer_tiles = array<i64: 3, 5>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p13", capacity = 4 : i64, producer_tile = array<i64: 3, 0>, consumer_tiles = array<i64: 3, 6>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p14", capacity = 4 : i64, producer_tile = array<i64: 3, 0>, consumer_tiles = array<i64: 3, 7>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p15", capacity = 4 : i64, producer_tile = array<i64: 3, 0>, consumer_tiles = array<i64: 3, 8>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    // IDs 16-23: shim col 6
    conduit.create {name = "p16", capacity = 4 : i64, producer_tile = array<i64: 6, 0>, consumer_tiles = array<i64: 6, 1>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p17", capacity = 4 : i64, producer_tile = array<i64: 6, 0>, consumer_tiles = array<i64: 6, 2>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p18", capacity = 4 : i64, producer_tile = array<i64: 6, 0>, consumer_tiles = array<i64: 6, 3>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p19", capacity = 4 : i64, producer_tile = array<i64: 6, 0>, consumer_tiles = array<i64: 6, 4>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p20", capacity = 4 : i64, producer_tile = array<i64: 6, 0>, consumer_tiles = array<i64: 6, 5>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p21", capacity = 4 : i64, producer_tile = array<i64: 6, 0>, consumer_tiles = array<i64: 6, 6>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p22", capacity = 4 : i64, producer_tile = array<i64: 6, 0>, consumer_tiles = array<i64: 6, 7>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p23", capacity = 4 : i64, producer_tile = array<i64: 6, 0>, consumer_tiles = array<i64: 6, 8>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    // IDs 24-31: shim col 7
    conduit.create {name = "p24", capacity = 4 : i64, producer_tile = array<i64: 7, 0>, consumer_tiles = array<i64: 7, 1>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p25", capacity = 4 : i64, producer_tile = array<i64: 7, 0>, consumer_tiles = array<i64: 7, 2>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p26", capacity = 4 : i64, producer_tile = array<i64: 7, 0>, consumer_tiles = array<i64: 7, 3>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p27", capacity = 4 : i64, producer_tile = array<i64: 7, 0>, consumer_tiles = array<i64: 7, 4>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p28", capacity = 4 : i64, producer_tile = array<i64: 7, 0>, consumer_tiles = array<i64: 7, 5>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p29", capacity = 4 : i64, producer_tile = array<i64: 7, 0>, consumer_tiles = array<i64: 7, 6>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p30", capacity = 4 : i64, producer_tile = array<i64: 7, 0>, consumer_tiles = array<i64: 7, 7>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "p31", capacity = 4 : i64, producer_tile = array<i64: 7, 0>, consumer_tiles = array<i64: 7, 8>, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}

    // Explicit packet conduits from (5,1) fill both its MM2S channels.
    // These would normally use IDs 32+, but the allocator already returned
    // nullopt — so passFailed is set when pkt_c1/c2 are processed.
    // Wait — actually these are processed in Phase 4.5a, AFTER Phase 4a.
    // At this point remaining() == 0 because Phase 4a consumed 32 IDs.
    // pkt_c1 tries to allocate ID 32 → error fires HERE (explicit packet).
    conduit.create {name = "pkt_c1", capacity = 4 : i64,
                    producer_tile = array<i64: 5, 1>,
                    consumer_tiles = array<i64: 5, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}
    conduit.create {name = "pkt_c2", capacity = 4 : i64,
                    producer_tile = array<i64: 5, 1>,
                    consumer_tiles = array<i64: 5, 5>,
                    element_type = memref<4xi32>, depth = 1 : i64
                    }
  }
}
