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
    conduit.create @p00 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p01 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p02 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p03 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p04 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p05 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p06 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p07 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    // IDs 8-15: shim col 3
    conduit.create @p08 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p09 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p10 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p11 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p12 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p13 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p14 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p15 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    // IDs 16-23: shim col 6
    conduit.create @p16 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p17 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p18 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p19 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p20 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p21 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p22 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p23 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    // IDs 24-31: shim col 7
    conduit.create @p24 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p25 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p26 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p27 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p28 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p29 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p30 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p31 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}

    // Explicit packet conduits from (5,1) fill both its MM2S channels.
    // pkt_c1 tries to allocate ID 32 → error fires (explicit packet).
    conduit.create @pkt_c1 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}
    conduit.create @pkt_c2 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>, depth = 1 : i64
                    }

    // Shim producer allocations.
    aie.shim_dma_allocation @p00_shim_alloc(%ts2_0, MM2S, 0)
    aie.shim_dma_allocation @p01_shim_alloc(%ts2_0, MM2S, 0)
    aie.shim_dma_allocation @p02_shim_alloc(%ts2_0, MM2S, 0)
    aie.shim_dma_allocation @p03_shim_alloc(%ts2_0, MM2S, 0)
    aie.shim_dma_allocation @p04_shim_alloc(%ts2_0, MM2S, 0)
    aie.shim_dma_allocation @p05_shim_alloc(%ts2_0, MM2S, 0)
    aie.shim_dma_allocation @p06_shim_alloc(%ts2_0, MM2S, 0)
    aie.shim_dma_allocation @p07_shim_alloc(%ts2_0, MM2S, 0)
    aie.shim_dma_allocation @p08_shim_alloc(%ts3_0, MM2S, 0)
    aie.shim_dma_allocation @p09_shim_alloc(%ts3_0, MM2S, 0)
    aie.shim_dma_allocation @p10_shim_alloc(%ts3_0, MM2S, 0)
    aie.shim_dma_allocation @p11_shim_alloc(%ts3_0, MM2S, 0)
    aie.shim_dma_allocation @p12_shim_alloc(%ts3_0, MM2S, 0)
    aie.shim_dma_allocation @p13_shim_alloc(%ts3_0, MM2S, 0)
    aie.shim_dma_allocation @p14_shim_alloc(%ts3_0, MM2S, 0)
    aie.shim_dma_allocation @p15_shim_alloc(%ts3_0, MM2S, 0)
    aie.shim_dma_allocation @p16_shim_alloc(%ts6_0, MM2S, 0)
    aie.shim_dma_allocation @p17_shim_alloc(%ts6_0, MM2S, 0)
    aie.shim_dma_allocation @p18_shim_alloc(%ts6_0, MM2S, 0)
    aie.shim_dma_allocation @p19_shim_alloc(%ts6_0, MM2S, 0)
    aie.shim_dma_allocation @p20_shim_alloc(%ts6_0, MM2S, 0)
    aie.shim_dma_allocation @p21_shim_alloc(%ts6_0, MM2S, 0)
    aie.shim_dma_allocation @p22_shim_alloc(%ts6_0, MM2S, 0)
    aie.shim_dma_allocation @p23_shim_alloc(%ts6_0, MM2S, 0)
    aie.shim_dma_allocation @p24_shim_alloc(%ts7_0, MM2S, 0)
    aie.shim_dma_allocation @p25_shim_alloc(%ts7_0, MM2S, 0)
    aie.shim_dma_allocation @p26_shim_alloc(%ts7_0, MM2S, 0)
    aie.shim_dma_allocation @p27_shim_alloc(%ts7_0, MM2S, 0)
    aie.shim_dma_allocation @p28_shim_alloc(%ts7_0, MM2S, 0)
    aie.shim_dma_allocation @p29_shim_alloc(%ts7_0, MM2S, 0)
    aie.shim_dma_allocation @p30_shim_alloc(%ts7_0, MM2S, 0)
    aie.shim_dma_allocation @p31_shim_alloc(%ts7_0, MM2S, 0)

    // Consumer cores for shim conduits.
    aie.core(%ts2_1) { conduit.acquire {name = @p00, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts2_2) { conduit.acquire {name = @p01, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts2_3) { conduit.acquire {name = @p02, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts2_4) { conduit.acquire {name = @p03, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts2_5) { conduit.acquire {name = @p04, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts2_6) { conduit.acquire {name = @p05, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts2_7) { conduit.acquire {name = @p06, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts2_8) { conduit.acquire {name = @p07, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts3_1) { conduit.acquire {name = @p08, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts3_2) { conduit.acquire {name = @p09, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts3_3) { conduit.acquire {name = @p10, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts3_4) { conduit.acquire {name = @p11, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts3_5) { conduit.acquire {name = @p12, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts3_6) { conduit.acquire {name = @p13, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts3_7) { conduit.acquire {name = @p14, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts3_8) { conduit.acquire {name = @p15, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts6_1) { conduit.acquire {name = @p16, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts6_2) { conduit.acquire {name = @p17, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts6_3) { conduit.acquire {name = @p18, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts6_4) { conduit.acquire {name = @p19, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts6_5) { conduit.acquire {name = @p20, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts6_6) { conduit.acquire {name = @p21, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts6_7) { conduit.acquire {name = @p22, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts6_8) { conduit.acquire {name = @p23, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts7_1) { conduit.acquire {name = @p24, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts7_2) { conduit.acquire {name = @p25, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts7_3) { conduit.acquire {name = @p26, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts7_4) { conduit.acquire {name = @p27, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts7_5) { conduit.acquire {name = @p28, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts7_6) { conduit.acquire {name = @p29, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts7_7) { conduit.acquire {name = @p30, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%ts7_8) { conduit.acquire {name = @p31, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }

    // Producer and consumer cores for compute conduits (pkt_c1, pkt_c2).
    aie.core(%tc5_1) {
      conduit.acquire {name = @pkt_c1, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @pkt_c2, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    aie.core(%tc5_3) { conduit.acquire {name = @pkt_c1, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%tc5_5) { conduit.acquire {name = @pkt_c2, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
  }
}
