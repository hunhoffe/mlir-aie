// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests iter_count=K: DMAStartOp.repeat_count should be K-1, and the last BD
// in the chain should be non-circular (last BD → end block, not back to bb1).
//
// For iter_count = 5 on a depth-2 MemTile fifo, DMAStartOp repeat_count = 4.

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// CHECK:     aie.memtile_dma
// CHECK:       aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 4)
// The last BD must NOT loop back to ^bb1; instead it goes to the end block.
// CHECK:       aie.next_bd ^bb3

module {
  aie.device(xcve2302) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)

    aie.objectfifo @in(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) {iter_count = 5 : i32}
        : !aie.objectfifo<memref<1024xi32>>
  }
}
