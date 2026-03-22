// RUN: not aie-opt --objectfifo-to-conduit --conduit-to-dma %s 2>&1 | FileCheck %s
//
// Negative test: MemTile BD parity pool overflow.
//
// AIE2 MemTiles have 48 BDs partitioned by channel parity:
//   BDs 0-23  (24): EVEN-numbered channels (0, 2, 4)
//   BDs 24-47 (24): ODD-numbered channels  (1, 3, 5)
//
// This test creates a 1->4 distribute through MemTile(2,1) with depth=5.
// The resulting BD allocation on the MemTile:
//   S2MM channel 0 (even): 5 * 4 = 20 ingest BDs
//   MM2S channel 0 (even): 5 BDs (destination d1)
//   MM2S channel 1 (odd):  5 BDs (destination d2)
//   MM2S channel 2 (even): 5 BDs (destination d3)
//   MM2S channel 3 (odd):  5 BDs (destination d4)
//
// Even pool total: 20 + 5 + 5 = 30 > 24  --> MUST ERROR
// Odd pool total:  5 + 5 = 10 <= 24      --> OK
//
// CHECK: error:{{.*}}MemTile BD parity constraint violated: even-channel pool has {{[0-9]+}} BDs, exceeds hardware limit of 24

module @bd_parity_error {
  aie.device(xcve2302) {
    %shim = aie.tile(2, 0)
    %mem  = aie.tile(2, 1)
    %t22  = aie.tile(2, 2)
    %t23  = aie.tile(2, 3)
    %t32  = aie.tile(3, 2)
    %t33  = aie.tile(3, 3)

    // Source: shim -> MemTile, depth=5, 100 elements
    aie.objectfifo @src (%shim, {%mem}, 5 : i32) : !aie.objectfifo<memref<100xi32>>

    // 4 destinations from MemTile -> compute tiles, depth=5, 25 elements each
    aie.objectfifo @d1 (%mem, {%t22}, 5 : i32) : !aie.objectfifo<memref<25xi32>>
    aie.objectfifo @d2 (%mem, {%t23}, 5 : i32) : !aie.objectfifo<memref<25xi32>>
    aie.objectfifo @d3 (%mem, {%t32}, 5 : i32) : !aie.objectfifo<memref<25xi32>>
    aie.objectfifo @d4 (%mem, {%t33}, 5 : i32) : !aie.objectfifo<memref<25xi32>>

    // Distribute: 1 source -> 4 destinations through MemTile(2,1)
    aie.objectfifo.link [@src] -> [@d1, @d2, @d3, @d4] ([][0, 25, 50, 75])
  }
}
