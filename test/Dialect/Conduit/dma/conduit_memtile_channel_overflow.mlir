// RUN: not aie-opt --objectfifo-to-conduit --conduit-to-dma %s 2>&1 | FileCheck %s
//
// Negative test: MemTile DMA channel overflow.
//
// AIE2 MemTiles have 6 MM2S and 6 S2MM DMA channels (indices 0-5).
// This test creates a 1->7 distribute through MemTile(2,1), requiring
// 7 MM2S channels on the MemTile. Channel index 6 exceeds the hardware
// maximum of 5.
//
// Pass C must emit a compile-time error instead of silently generating
// DMA : 6 (which crashes aiecc with an AIEPathFinder assertion:
// `i < sb.srcPorts.size()`).
//
// CHECK: error:{{.*}}DMA channel budget exceeded{{.*}}needs 7 MM2S channels{{.*}}maximum is 6

module @memtile_channel_overflow {
  aie.device(xcve2302) {
    %shim = aie.tile(2, 0)
    %mem  = aie.tile(2, 1)
    %t22  = aie.tile(2, 2)
    %t23  = aie.tile(2, 3)
    %t32  = aie.tile(3, 2)
    %t33  = aie.tile(3, 3)
    %t42  = aie.tile(4, 2)
    %t43  = aie.tile(4, 3)
    %t52  = aie.tile(5, 2)

    // Source: shim -> MemTile, depth=2, 700 elements
    aie.objectfifo @src (%shim, {%mem}, 2 : i32) : !aie.objectfifo<memref<700xi32>>

    // 7 destinations from MemTile -> compute tiles, depth=2, 100 elements each
    aie.objectfifo @d1 (%mem, {%t22}, 2 : i32) : !aie.objectfifo<memref<100xi32>>
    aie.objectfifo @d2 (%mem, {%t23}, 2 : i32) : !aie.objectfifo<memref<100xi32>>
    aie.objectfifo @d3 (%mem, {%t32}, 2 : i32) : !aie.objectfifo<memref<100xi32>>
    aie.objectfifo @d4 (%mem, {%t33}, 2 : i32) : !aie.objectfifo<memref<100xi32>>
    aie.objectfifo @d5 (%mem, {%t42}, 2 : i32) : !aie.objectfifo<memref<100xi32>>
    aie.objectfifo @d6 (%mem, {%t43}, 2 : i32) : !aie.objectfifo<memref<100xi32>>
    aie.objectfifo @d7 (%mem, {%t52}, 2 : i32) : !aie.objectfifo<memref<100xi32>>

    // Distribute: 1 source -> 7 destinations through MemTile(2,1)
    // Requires 7 MM2S channels on the MemTile (max is 6)
    aie.objectfifo.link [@src] -> [@d1, @d2, @d3, @d4, @d5, @d6, @d7] ([][0, 100, 200, 300, 400, 500, 600])
  }
}
