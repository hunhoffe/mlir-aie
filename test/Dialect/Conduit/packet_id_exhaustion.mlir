// RUN: aie-opt --conduit-to-dma --verify-diagnostics %s
//
// P1-A: Packet flow ID allocator — one over the 32-ID hardware limit.
//
// 33 packet conduits on xcvc1902.  Each conduit contributes exactly one
// aie.packet_flow op.  The 33rd conduit exhausts the 5-bit hardware ID
// space (IDs 0-31 are valid; 32 is out of range), causing the
// PacketIDAllocator to emit a hard error on the module op.
//
// Verifies:
//   - The allocator fires when next >= limit (32)
//   - The error message contains "packet flow ID exhausted"
//   - The limit count (32) appears in the diagnostic
//   - --verify-diagnostics exits 0 when the annotation matches
//
// Topology: same 4-column xcvc1902 structure as packet_id_limit_ok.mlir,
// plus one additional conduit in column 10 (also a shim NOC column) that
// pushes the count to 33.

// expected-error @+1 {{packet flow ID exhausted: design requires more than 32 distinct packet flows}}
module @pkt_id_exhaustion {
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
    // Column 10: shim NOC at (10,0), one extra compute tile (10,1)
    %t10_0 = aie.tile(10, 0)
    %t10_1 = aie.tile(10, 1)

    // Conduits p00-p31: same as packet_id_limit_ok.mlir (packet IDs 0-31).

    conduit.create {name = "p00", capacity = 4 : i64,
                    producer_tile = array<i64: 2, 0>,
                    consumer_tiles = array<i64: 2, 1>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p01", capacity = 4 : i64,
                    producer_tile = array<i64: 2, 0>,
                    consumer_tiles = array<i64: 2, 2>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p02", capacity = 4 : i64,
                    producer_tile = array<i64: 2, 0>,
                    consumer_tiles = array<i64: 2, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p03", capacity = 4 : i64,
                    producer_tile = array<i64: 2, 0>,
                    consumer_tiles = array<i64: 2, 4>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p04", capacity = 4 : i64,
                    producer_tile = array<i64: 2, 0>,
                    consumer_tiles = array<i64: 2, 5>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p05", capacity = 4 : i64,
                    producer_tile = array<i64: 2, 0>,
                    consumer_tiles = array<i64: 2, 6>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p06", capacity = 4 : i64,
                    producer_tile = array<i64: 2, 0>,
                    consumer_tiles = array<i64: 2, 7>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p07", capacity = 4 : i64,
                    producer_tile = array<i64: 2, 0>,
                    consumer_tiles = array<i64: 2, 8>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p08", capacity = 4 : i64,
                    producer_tile = array<i64: 3, 0>,
                    consumer_tiles = array<i64: 3, 1>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p09", capacity = 4 : i64,
                    producer_tile = array<i64: 3, 0>,
                    consumer_tiles = array<i64: 3, 2>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p10", capacity = 4 : i64,
                    producer_tile = array<i64: 3, 0>,
                    consumer_tiles = array<i64: 3, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p11", capacity = 4 : i64,
                    producer_tile = array<i64: 3, 0>,
                    consumer_tiles = array<i64: 3, 4>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p12", capacity = 4 : i64,
                    producer_tile = array<i64: 3, 0>,
                    consumer_tiles = array<i64: 3, 5>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p13", capacity = 4 : i64,
                    producer_tile = array<i64: 3, 0>,
                    consumer_tiles = array<i64: 3, 6>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p14", capacity = 4 : i64,
                    producer_tile = array<i64: 3, 0>,
                    consumer_tiles = array<i64: 3, 7>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p15", capacity = 4 : i64,
                    producer_tile = array<i64: 3, 0>,
                    consumer_tiles = array<i64: 3, 8>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p16", capacity = 4 : i64,
                    producer_tile = array<i64: 6, 0>,
                    consumer_tiles = array<i64: 6, 1>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p17", capacity = 4 : i64,
                    producer_tile = array<i64: 6, 0>,
                    consumer_tiles = array<i64: 6, 2>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p18", capacity = 4 : i64,
                    producer_tile = array<i64: 6, 0>,
                    consumer_tiles = array<i64: 6, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p19", capacity = 4 : i64,
                    producer_tile = array<i64: 6, 0>,
                    consumer_tiles = array<i64: 6, 4>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p20", capacity = 4 : i64,
                    producer_tile = array<i64: 6, 0>,
                    consumer_tiles = array<i64: 6, 5>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p21", capacity = 4 : i64,
                    producer_tile = array<i64: 6, 0>,
                    consumer_tiles = array<i64: 6, 6>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p22", capacity = 4 : i64,
                    producer_tile = array<i64: 6, 0>,
                    consumer_tiles = array<i64: 6, 7>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p23", capacity = 4 : i64,
                    producer_tile = array<i64: 6, 0>,
                    consumer_tiles = array<i64: 6, 8>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p24", capacity = 4 : i64,
                    producer_tile = array<i64: 7, 0>,
                    consumer_tiles = array<i64: 7, 1>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p25", capacity = 4 : i64,
                    producer_tile = array<i64: 7, 0>,
                    consumer_tiles = array<i64: 7, 2>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p26", capacity = 4 : i64,
                    producer_tile = array<i64: 7, 0>,
                    consumer_tiles = array<i64: 7, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p27", capacity = 4 : i64,
                    producer_tile = array<i64: 7, 0>,
                    consumer_tiles = array<i64: 7, 4>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p28", capacity = 4 : i64,
                    producer_tile = array<i64: 7, 0>,
                    consumer_tiles = array<i64: 7, 5>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p29", capacity = 4 : i64,
                    producer_tile = array<i64: 7, 0>,
                    consumer_tiles = array<i64: 7, 6>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p30", capacity = 4 : i64,
                    producer_tile = array<i64: 7, 0>,
                    consumer_tiles = array<i64: 7, 7>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "p31", capacity = 4 : i64,
                    producer_tile = array<i64: 7, 0>,
                    consumer_tiles = array<i64: 7, 8>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}

    // 33rd conduit: exceeds the 32-ID hardware limit.
    // PacketIDAllocator.allocate() returns nullopt for this call,
    // emitting the "packet flow ID exhausted" error on the module op.
    conduit.create {name = "p32", capacity = 4 : i64,
                    producer_tile = array<i64: 10, 0>,
                    consumer_tiles = array<i64: 10, 1>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
  }
}
