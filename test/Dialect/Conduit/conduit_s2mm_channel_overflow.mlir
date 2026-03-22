// RUN: not aie-opt --objectfifo-to-conduit --conduit-to-dma %s 2>&1 | FileCheck %s
//
// Negative test: S2MM DMA channel overflow on a compute tile.
//
// A compute tile on xcve2302 has 2 S2MM DMA channels. Three non-adjacent
// producers all route to the same consumer tile, exhausting the S2MM budget.
//
// CHECK: error:{{.*}}S2MM DMA channel exhausted on tile (2,2)

module @s2mm_overflow {
  aie.device(xcve2302) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t42 = aie.tile(4, 2)
    %t22 = aie.tile(2, 2)

    // Three non-adjacent producers all targeting tile(2,2).
    // tile(0,2): col diff = 2, non-adjacent.
    // tile(0,3): different col and row, non-adjacent.
    // tile(4,2): col diff = 2, non-adjacent.
    // tile(2,2) has only 2 S2MM channels — the third must fail.
    aie.objectfifo @a (%t02, {%t22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @b (%t03, {%t22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @c (%t42, {%t22}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
  }
}
