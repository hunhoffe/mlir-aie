// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests iter_count=1 (edge case K=1): DMAStartOp should have NO repeat_count
// attribute (K-1 = 0, which is the default).  The BD chain must be non-circular
// (last BD goes to end block, not back to ^bb1).

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {
// CHECK:     aie.lock({{.*}}) {init = 2 : i32
// CHECK:     aie.lock({{.*}}) {init = 0 : i32
// DMAStartOp must NOT have repeat_count attribute (default 0 = K-1 where K=1)
// CHECK:     aie.memtile_dma
// CHECK:       aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK-NOT:   repeat_count
// BD chain is non-circular: last BD goes to end block
// CHECK:     ^bb1:
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb3:
// CHECK:       aie.end

module {
  aie.device(xcve2302) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)

    aie.objectfifo @in(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) {iter_count = 1 : i32}
        : !aie.objectfifo<memref<1024xi32>>
  }
}
