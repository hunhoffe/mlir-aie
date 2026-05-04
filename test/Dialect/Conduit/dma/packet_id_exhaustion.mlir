// RUN: aie-opt --conduit-to-dma --verify-diagnostics %s
//
// P1-A: Packet flow ID allocator — one over the per-MemTile 32-ID limit.
//
// 33 packet conduits all routed through shim tile (2,0), placing them in
// the same per-MemTile domain (column 2).  Each conduit contributes exactly
// one aie.packet_flow op.  The 33rd conduit exhausts the 5-bit hardware
// ID space within the domain (IDs 0-31 are valid; 32 is out of range),
// causing the PacketIDAllocator to emit a hard error.
//
// Consumer tiles are spread across columns 2, 3, 6, 7, and 10 to provide
// 33 unique destinations while keeping all producer allocations on one shim.
//
// Verifies:
//   - The allocator fires when next >= limit (32) within a single domain
//   - The error message contains "packet flow ID exhausted in MemTile domain"
//   - --verify-diagnostics exits 0 when the annotation matches

// expected-error @+1 {{packet flow ID exhausted in MemTile domain: design requires more than 31 distinct packet flows per MemTile}}
module @pkt_id_exhaustion {
  aie.device(xcvc1902) {
    // Shim tile: single producer for all 33 conduits (same per-MemTile domain).
    %t2_0 = aie.tile(2, 0)
    // Column 2: compute tiles (2,1)-(2,8) — 8 consumers.
    %t2_1 = aie.tile(2, 1)
    %t2_2 = aie.tile(2, 2)
    %t2_3 = aie.tile(2, 3)
    %t2_4 = aie.tile(2, 4)
    %t2_5 = aie.tile(2, 5)
    %t2_6 = aie.tile(2, 6)
    %t2_7 = aie.tile(2, 7)
    %t2_8 = aie.tile(2, 8)
    // Column 3: compute tiles (3,1)-(3,8) — 8 consumers.
    %t3_1 = aie.tile(3, 1)
    %t3_2 = aie.tile(3, 2)
    %t3_3 = aie.tile(3, 3)
    %t3_4 = aie.tile(3, 4)
    %t3_5 = aie.tile(3, 5)
    %t3_6 = aie.tile(3, 6)
    %t3_7 = aie.tile(3, 7)
    %t3_8 = aie.tile(3, 8)
    // Column 6: compute tiles (6,1)-(6,8) — 8 consumers.
    %t6_1 = aie.tile(6, 1)
    %t6_2 = aie.tile(6, 2)
    %t6_3 = aie.tile(6, 3)
    %t6_4 = aie.tile(6, 4)
    %t6_5 = aie.tile(6, 5)
    %t6_6 = aie.tile(6, 6)
    %t6_7 = aie.tile(6, 7)
    %t6_8 = aie.tile(6, 8)
    // Column 7: compute tiles (7,1)-(7,8) — 8 consumers.
    %t7_1 = aie.tile(7, 1)
    %t7_2 = aie.tile(7, 2)
    %t7_3 = aie.tile(7, 3)
    %t7_4 = aie.tile(7, 4)
    %t7_5 = aie.tile(7, 5)
    %t7_6 = aie.tile(7, 6)
    %t7_7 = aie.tile(7, 7)
    %t7_8 = aie.tile(7, 8)
    // Column 10: one extra consumer tile (10,1) — the 33rd.
    %t10_1 = aie.tile(10, 1)

    // 33 packet conduits (p00-p32), all sharing the same MemTile domain.
    conduit.create @p00 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p01 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p02 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p03 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p04 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p05 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p06 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p07 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p08 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p09 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p10 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p11 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p12 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p13 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p14 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p15 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p16 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p17 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p18 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p19 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p20 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p21 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p22 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p23 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p24 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p25 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p26 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p27 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p28 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p29 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p30 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p31 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}

    // 33rd conduit: exceeds the 32-ID per-MemTile limit for column 2.
    conduit.create @p32 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}

    // ALL shim producer allocations reference the same shim tile (2,0),
    // placing all 33 conduits in column 2's per-MemTile domain.
    aie.shim_dma_allocation @p00_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p01_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p02_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p03_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p04_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p05_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p06_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p07_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p08_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p09_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p10_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p11_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p12_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p13_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p14_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p15_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p16_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p17_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p18_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p19_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p20_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p21_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p22_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p23_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p24_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p25_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p26_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p27_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p28_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p29_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p30_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p31_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p32_shim_alloc(%t2_0, MM2S, 0)

    // Consumer cores — 33 unique tiles across columns.
    aie.core(%t2_1) { conduit.acquire {name = @p00, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t2_2) { conduit.acquire {name = @p01, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t2_3) { conduit.acquire {name = @p02, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t2_4) { conduit.acquire {name = @p03, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t2_5) { conduit.acquire {name = @p04, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t2_6) { conduit.acquire {name = @p05, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t2_7) { conduit.acquire {name = @p06, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t2_8) { conduit.acquire {name = @p07, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t3_1) { conduit.acquire {name = @p08, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t3_2) { conduit.acquire {name = @p09, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t3_3) { conduit.acquire {name = @p10, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t3_4) { conduit.acquire {name = @p11, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t3_5) { conduit.acquire {name = @p12, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t3_6) { conduit.acquire {name = @p13, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t3_7) { conduit.acquire {name = @p14, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t3_8) { conduit.acquire {name = @p15, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t6_1) { conduit.acquire {name = @p16, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t6_2) { conduit.acquire {name = @p17, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t6_3) { conduit.acquire {name = @p18, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t6_4) { conduit.acquire {name = @p19, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t6_5) { conduit.acquire {name = @p20, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t6_6) { conduit.acquire {name = @p21, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t6_7) { conduit.acquire {name = @p22, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t6_8) { conduit.acquire {name = @p23, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t7_1) { conduit.acquire {name = @p24, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t7_2) { conduit.acquire {name = @p25, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t7_3) { conduit.acquire {name = @p26, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t7_4) { conduit.acquire {name = @p27, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t7_5) { conduit.acquire {name = @p28, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t7_6) { conduit.acquire {name = @p29, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t7_7) { conduit.acquire {name = @p30, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t7_8) { conduit.acquire {name = @p31, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t10_1) { conduit.acquire {name = @p32, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
  }
}
