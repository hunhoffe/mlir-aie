// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// P1-A: Packet flow ID allocator — exactly at the 32-ID hardware limit.
//
// 32 packet conduits spread across 4 shim NOC tiles on xcvc1902 (AIE1,
// 8 conduits per column).  Each conduit contributes exactly one
// aie.packet_flow op, consuming one ID from the PacketIDAllocator.
// The 32nd conduit must succeed (ID 31 is the last valid 5-bit value).
//
// Topology: shim tiles at columns 2, 3, 6, 7 each drive 8 compute tiles
// in their respective columns (rows 1-8).  All consumer tiles are unique
// so no S2MM channel conflicts arise (each gets S2MM channel 0).
//
// AIE1 (xcvc1902) is used so that no per-shim locks are allocated in
// Phase 4a (AIE1 shim locks are managed by the runtime, not Conduit).
// This avoids per-tile lock overflow while testing all 32 packet IDs.
//
// Verifies:
//   - All 32 aie.packet_flow ops are emitted (no error)
//   - Packet flow IDs are allocated monotonically from 0 to 31
//   - No "packet flow ID exhausted" diagnostic is emitted
//   - conduit.create ops are fully erased by Phase 7

// CHECK-LABEL: module @pkt_id_limit_ok
// CHECK:       aie.packet_flow(0)
// CHECK:       aie.packet_flow(1)
// CHECK:       aie.packet_flow(2)
// CHECK:       aie.packet_flow(3)
// CHECK:       aie.packet_flow(4)
// CHECK:       aie.packet_flow(5)
// CHECK:       aie.packet_flow(6)
// CHECK:       aie.packet_flow(7)
// CHECK:       aie.packet_flow(8)
// CHECK:       aie.packet_flow(9)
// CHECK:       aie.packet_flow(10)
// CHECK:       aie.packet_flow(11)
// CHECK:       aie.packet_flow(12)
// CHECK:       aie.packet_flow(13)
// CHECK:       aie.packet_flow(14)
// CHECK:       aie.packet_flow(15)
// CHECK:       aie.packet_flow(16)
// CHECK:       aie.packet_flow(17)
// CHECK:       aie.packet_flow(18)
// CHECK:       aie.packet_flow(19)
// CHECK:       aie.packet_flow(20)
// CHECK:       aie.packet_flow(21)
// CHECK:       aie.packet_flow(22)
// CHECK:       aie.packet_flow(23)
// CHECK:       aie.packet_flow(24)
// CHECK:       aie.packet_flow(25)
// CHECK:       aie.packet_flow(26)
// CHECK:       aie.packet_flow(27)
// CHECK:       aie.packet_flow(28)
// CHECK:       aie.packet_flow(29)
// CHECK:       aie.packet_flow(30)
// CHECK:       aie.packet_flow(31)
// CHECK-NOT:   conduit.create

module @pkt_id_limit_ok {
  aie.device(xcvc1902) {
    // Column 2: shim NOC at (2,0), compute tiles (2,1)-(2,8)
    %t2_0 = aie.tile(2, 0)
    %t2_1 = aie.tile(2, 1)
    %t2_2 = aie.tile(2, 2)
    %t2_3 = aie.tile(2, 3)
    %t2_4 = aie.tile(2, 4)
    %t2_5 = aie.tile(2, 5)
    %t2_6 = aie.tile(2, 6)
    %t2_7 = aie.tile(2, 7)
    %t2_8 = aie.tile(2, 8)
    // Column 3: shim NOC at (3,0), compute tiles (3,1)-(3,8)
    %t3_0 = aie.tile(3, 0)
    %t3_1 = aie.tile(3, 1)
    %t3_2 = aie.tile(3, 2)
    %t3_3 = aie.tile(3, 3)
    %t3_4 = aie.tile(3, 4)
    %t3_5 = aie.tile(3, 5)
    %t3_6 = aie.tile(3, 6)
    %t3_7 = aie.tile(3, 7)
    %t3_8 = aie.tile(3, 8)
    // Column 6: shim NOC at (6,0), compute tiles (6,1)-(6,8)
    %t6_0 = aie.tile(6, 0)
    %t6_1 = aie.tile(6, 1)
    %t6_2 = aie.tile(6, 2)
    %t6_3 = aie.tile(6, 3)
    %t6_4 = aie.tile(6, 4)
    %t6_5 = aie.tile(6, 5)
    %t6_6 = aie.tile(6, 6)
    %t6_7 = aie.tile(6, 7)
    %t6_8 = aie.tile(6, 8)
    // Column 7: shim NOC at (7,0), compute tiles (7,1)-(7,8)
    %t7_0 = aie.tile(7, 0)
    %t7_1 = aie.tile(7, 1)
    %t7_2 = aie.tile(7, 2)
    %t7_3 = aie.tile(7, 3)
    %t7_4 = aie.tile(7, 4)
    %t7_5 = aie.tile(7, 5)
    %t7_6 = aie.tile(7, 6)
    %t7_7 = aie.tile(7, 7)
    %t7_8 = aie.tile(7, 8)

    // Column 2: 8 packet conduits, each to a unique consumer tile.
    // Packet IDs 0-7.
    conduit.create @p00 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p01 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p02 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p03 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p04 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p05 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p06 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p07 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    // Column 3: Packet IDs 8-15.
    conduit.create @p08 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p09 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p10 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p11 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p12 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p13 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p14 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p15 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    // Column 6: Packet IDs 16-23.
    conduit.create @p16 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p17 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p18 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p19 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p20 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p21 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p22 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p23 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    // Column 7: Packet IDs 24-31.
    conduit.create @p24 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p25 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p26 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p27 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p28 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p29 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p30 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p31 {slot_elems = 4 : i64, element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}

    // Shim producer allocations (inferAllTiles Source 3: MM2S → producer tile).
    aie.shim_dma_allocation @p00_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p01_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p02_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p03_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p04_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p05_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p06_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p07_shim_alloc(%t2_0, MM2S, 0)
    aie.shim_dma_allocation @p08_shim_alloc(%t3_0, MM2S, 0)
    aie.shim_dma_allocation @p09_shim_alloc(%t3_0, MM2S, 0)
    aie.shim_dma_allocation @p10_shim_alloc(%t3_0, MM2S, 0)
    aie.shim_dma_allocation @p11_shim_alloc(%t3_0, MM2S, 0)
    aie.shim_dma_allocation @p12_shim_alloc(%t3_0, MM2S, 0)
    aie.shim_dma_allocation @p13_shim_alloc(%t3_0, MM2S, 0)
    aie.shim_dma_allocation @p14_shim_alloc(%t3_0, MM2S, 0)
    aie.shim_dma_allocation @p15_shim_alloc(%t3_0, MM2S, 0)
    aie.shim_dma_allocation @p16_shim_alloc(%t6_0, MM2S, 0)
    aie.shim_dma_allocation @p17_shim_alloc(%t6_0, MM2S, 0)
    aie.shim_dma_allocation @p18_shim_alloc(%t6_0, MM2S, 0)
    aie.shim_dma_allocation @p19_shim_alloc(%t6_0, MM2S, 0)
    aie.shim_dma_allocation @p20_shim_alloc(%t6_0, MM2S, 0)
    aie.shim_dma_allocation @p21_shim_alloc(%t6_0, MM2S, 0)
    aie.shim_dma_allocation @p22_shim_alloc(%t6_0, MM2S, 0)
    aie.shim_dma_allocation @p23_shim_alloc(%t6_0, MM2S, 0)
    aie.shim_dma_allocation @p24_shim_alloc(%t7_0, MM2S, 0)
    aie.shim_dma_allocation @p25_shim_alloc(%t7_0, MM2S, 0)
    aie.shim_dma_allocation @p26_shim_alloc(%t7_0, MM2S, 0)
    aie.shim_dma_allocation @p27_shim_alloc(%t7_0, MM2S, 0)
    aie.shim_dma_allocation @p28_shim_alloc(%t7_0, MM2S, 0)
    aie.shim_dma_allocation @p29_shim_alloc(%t7_0, MM2S, 0)
    aie.shim_dma_allocation @p30_shim_alloc(%t7_0, MM2S, 0)
    aie.shim_dma_allocation @p31_shim_alloc(%t7_0, MM2S, 0)

    // Consumer cores (inferAllTiles Source 2: Acquire{Consume} → consumer tile).
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
  }
}
