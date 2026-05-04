// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test: dma_repeat (from iter_count) must NOT propagate to
// AIE compute tile DMAs.  Compute tile DMAs cycle infinitely
// (repeat_count = 0, the default) -- the core body controls lifetime
// via main().  Only MemTile and Shim DMAs use finite repeat_count.
//
// Bug: Pass C set repeat_count on all DMAs including compute tiles,
// causing them to stop mid-chain and deadlock tiles waiting for data.
//
// Topology: MemTile(0,1) -> compute(0,2), iter_count = 5
//
// Expected:
//   MemTile MM2S DMA: repeat_count = 4  (iter_count - 1)
//   Compute S2MM DMA: NO repeat_count   (cycles infinitely)

// CHECK-LABEL: module @compute_tile_repeat_count
// CHECK:   aie.device(npu1_1col) {

// Compute tile DMA: S2MM must NOT have repeat_count
// CHECK:     aie.mem
// CHECK:       aie.dma_start(S2MM
// CHECK-NOT:   repeat_count
// CHECK:       aie.dma_bd

// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @compute_tile_repeat_count {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of(%tile_0_1, {%tile_0_2}, 2 : i32) {iter_count = 5 : i32}
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    }
  }
}
