// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests dimensionsToStream on a shim→MemTile→compute distribute pattern.
// The MemTile MM2S send BDs should carry the dims [<size = 4, stride = 1>];
// the consumer S2MM BDs should NOT carry any dims (no dimensionsFromStream).

// CHECK-LABEL: module
// CHECK:   aie.device(xcve2302) {

// Flows: shim→MemTile, MemTile→compute×2
// CHECK:     aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// CHECK:     aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// CHECK:     aie.flow(%{{.*}}, DMA : 1, %{{.*}}, DMA : 0)

// MemTile DMA: S2MM (link receive) + MM2S×2 (distribute send with dims)
// CHECK:     %{{.*}} = aie.memtile_dma(%{{.*}}) {
// CHECK:       aie.dma_start(S2MM
// MM2S channel 0: must carry dims
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_bd(%{{.*}}, [<size = 4, stride = 1>])
// CHECK:       aie.dma_bd(%{{.*}}, [<size = 4, stride = 1>])
// MM2S channel 1: must carry dims
// CHECK:       aie.dma_start(MM2S, 1
// CHECK:       aie.dma_bd(%{{.*}}, [<size = 4, stride = 1>])
// CHECK:       aie.dma_bd(%{{.*}}, [<size = 4, stride = 1>])

// Consumer 0 S2MM BD: NO dims
// CHECK:     %{{.*}} = aie.mem(%{{.*}}) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(%{{.*}} : memref<128xi32>, 0, 128)
// CHECK-NOT:   [<size
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd(%{{.*}} : memref<128xi32>, 0, 128)
// CHECK-NOT:   [<size
// CHECK:       aie.next_bd

// Consumer 1 S2MM BD: NO dims
// CHECK:     %{{.*}} = aie.mem(%{{.*}}) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(%{{.*}} : memref<128xi32>, 0, 128)
// CHECK-NOT:   [<size
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd(%{{.*}} : memref<128xi32>, 0, 128)
// CHECK-NOT:   [<size
// CHECK:       aie.next_bd

// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(xcve2302) {
    %tile_1_0 = aie.tile(1, 0)
    %tile_1_1 = aie.tile(1, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)

    aie.objectfifo @of0 (%tile_1_0, {%tile_1_1},
                         2 : i32) : !aie.objectfifo<memref<256xi32>>

    aie.objectfifo @of1 (%tile_1_1 dimensionsToStream [<size = 4, stride = 1>],
                        {%tile_2_2}, 2 : i32) : !aie.objectfifo<memref<128xi32>>

    aie.objectfifo @of2 (%tile_1_1 dimensionsToStream [<size = 4, stride = 1>],
                        {%tile_2_3}, 2 : i32) : !aie.objectfifo<memref<128xi32>>

    aie.objectfifo.link [ @of0 ] -> [ @of1, @of2 ] ([][0, 512])
  }
}
