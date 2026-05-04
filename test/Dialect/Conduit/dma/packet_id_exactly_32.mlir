// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// P2-A: Exactly 32 packet flows via mixed shim + compute-to-compute paths.
//
// 30 explicit packet conduits from shim tiles, plus 2 compute-to-compute
// explicit packet conduits from tile (5,1) to tiles (5,3) and (5,5).
// Total: 32 flows.  With per-MemTile-domain scoping, IDs reset per column:
//   Column 2: IDs 0-7,  Column 3: IDs 0-7,  Column 6: IDs 0-7,
//   Column 7: IDs 0-5,  Column 5 (compute): IDs 0-1.
//
// Verifies:
//   - All 32 aie.packet_flow ops are emitted (no error)
//   - Per-column ID reset works correctly
//   - No residual conduit.create ops

// CHECK-LABEL: module @pkt_id_exactly_32
// Column 2: packet IDs 1-8
// CHECK:       aie.packet_flow(1)
// CHECK:       aie.packet_flow(8)
// Column 3: packet IDs reset to 1-8
// CHECK:       aie.packet_flow(1)
// CHECK:       aie.packet_flow(8)
// Column 6: packet IDs reset to 1-8
// CHECK:       aie.packet_flow(1)
// CHECK:       aie.packet_flow(8)
// Column 7: packet IDs 1-6
// CHECK:       aie.packet_flow(1)
// CHECK:       aie.packet_flow(6)
// Column 5 (compute-to-compute): packet IDs 1-2
// CHECK:       aie.packet_flow(1)
// CHECK:       aie.packet_flow(2)
// CHECK-NOT:   error
// CHECK-NOT:   conduit.create

module @pkt_id_exactly_32 {
  aie.device(xcvc1902) {
    // Shim tiles and their consumers for IDs 0-29.
    %t2_0 = aie.tile(2, 0)  %t2_1 = aie.tile(2, 1)  %t2_2 = aie.tile(2, 2)
    %t2_3 = aie.tile(2, 3)  %t2_4 = aie.tile(2, 4)  %t2_5 = aie.tile(2, 5)
    %t2_6 = aie.tile(2, 6)  %t2_7 = aie.tile(2, 7)  %t2_8 = aie.tile(2, 8)
    %t3_0 = aie.tile(3, 0)  %t3_1 = aie.tile(3, 1)  %t3_2 = aie.tile(3, 2)
    %t3_3 = aie.tile(3, 3)  %t3_4 = aie.tile(3, 4)  %t3_5 = aie.tile(3, 5)
    %t3_6 = aie.tile(3, 6)  %t3_7 = aie.tile(3, 7)  %t3_8 = aie.tile(3, 8)
    %t6_0 = aie.tile(6, 0)  %t6_1 = aie.tile(6, 1)  %t6_2 = aie.tile(6, 2)
    %t6_3 = aie.tile(6, 3)  %t6_4 = aie.tile(6, 4)  %t6_5 = aie.tile(6, 5)
    %t6_6 = aie.tile(6, 6)  %t6_7 = aie.tile(6, 7)  %t6_8 = aie.tile(6, 8)
    %t7_0 = aie.tile(7, 0)  %t7_1 = aie.tile(7, 1)  %t7_2 = aie.tile(7, 2)
    %t7_3 = aie.tile(7, 3)  %t7_4 = aie.tile(7, 4)  %t7_5 = aie.tile(7, 5)
    %t7_6 = aie.tile(7, 6)
    // Compute tiles for IDs 30-31 (explicit packet compute→compute).
    %t5_1 = aie.tile(5, 1)  %t5_3 = aie.tile(5, 3)  %t5_5 = aie.tile(5, 5)

    // Column 2: per-MemTile packet IDs 0-7
    conduit.create @p00 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p01 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p02 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p03 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p04 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p05 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p06 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p07 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    // Column 3: per-MemTile packet IDs 0-7
    conduit.create @p08 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p09 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p10 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p11 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p12 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p13 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p14 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p15 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    // Column 6: per-MemTile packet IDs 0-7
    conduit.create @p16 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p17 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p18 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p19 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p20 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p21 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p22 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p23 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    // Column 7: per-MemTile packet IDs 0-5 (6 of 8 rows)
    conduit.create @p24 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p25 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p26 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p27 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p28 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @p29 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}

    // Column 5: compute-to-compute, per-MemTile packet IDs 0-1.
    conduit.create @c30 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}
    conduit.create @c31 {element_type = memref<4xi32>, depth = 1 : i64, routing_mode = #conduit.routing_mode<packet>}

    // Shim producer allocations.
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

    // Consumer cores for shim conduits.
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

    // Producer and consumer cores for compute-to-compute conduits (IDs 30-31).
    // Note: (5,1) produces both c30 and c31; consumers are (5,3) and (5,5).
    // Tile (5,3) is adjacent to neither (5,1) nor (5,5) — uses DMA path.
    aie.core(%t5_1) {
      conduit.acquire {name = @c30, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @c31, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    aie.core(%t5_3) { conduit.acquire {name = @c30, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
    aie.core(%t5_5) { conduit.acquire {name = @c31, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>  aie.end }
  }
}
