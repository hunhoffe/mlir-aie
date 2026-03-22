// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Positive test: MemTile BD parity pool within limits.
//
// This test creates a 1->3 distribute through MemTile(2,1) with depth=2.
// The resulting BD allocation on the MemTile:
//   S2MM channel 0 (even): 2 * 3 = 6 ingest BDs
//   MM2S channel 0 (even): 2 BDs (destination d1)
//   MM2S channel 1 (odd):  2 BDs (destination d2)
//   MM2S channel 2 (even): 2 BDs (destination d3)
//
// Even pool total: 6 + 2 + 2 = 10 <= 24  --> OK
// Odd pool total:  2 <= 24                --> OK
//
// Verifies no false positives from the parity pool check.

// CHECK:       aie.device
// CHECK:       aie.memtile_dma
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_start(MM2S, 1
// CHECK:       aie.dma_start(MM2S, 2
// CHECK:       aie.end
// CHECK-NOT:   error

module @bd_parity_ok {
  aie.device(xcve2302) {
    %shim = aie.tile(2, 0)
    %mem  = aie.tile(2, 1)
    %t22  = aie.tile(2, 2)
    %t23  = aie.tile(2, 3)
    %t33  = aie.tile(3, 3)

    // Source: shim -> MemTile, depth=2, 48 elements
    aie.objectfifo @src (%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    // 3 destinations from MemTile -> compute tiles, depth=2
    aie.objectfifo @d1 (%mem, {%t22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @d2 (%mem, {%t23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @d3 (%mem, {%t33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

    // Distribute: 1 source -> 3 destinations through MemTile(2,1)
    aie.objectfifo.link [@src] -> [@d1, @d2, @d3] ([][0, 16, 36])
  }
}
