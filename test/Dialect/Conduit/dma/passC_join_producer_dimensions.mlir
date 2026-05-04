// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test: producer_dimensions (dimensionsToStream) must be
// preserved on the MemTile join MM2S BD.
//
// Bug: ConduitToDMALink.cpp was dropping producer_dimensions when emitting
// join MM2S BDs, so the MemTile DMA sent data with linear layout
// instead of the requested strided layout.
//
// Topology:
//   compute(2,2) --+
//                  +--> MemTile(2,1) --> shim(2,0)
//   compute(2,3) --+
//   join_src_a: compute(2,2)->MemTile, 16xi32
//   join_src_b: compute(2,3)->MemTile, 16xi32
//   join_out: MemTile->shim, 32xi32, dimensionsToStream=[<size=2, stride=4>, <size=4, stride=1>]
//   objectfifo.link joins join_src_a + join_src_b -> join_out through MemTile(2,1)
//
// Expected:
//   The MemTile MM2S BD (output side) must carry the producer dimensions
//   [<size = 2, stride = 4>, <size = 4, stride = 1>].

// CHECK-LABEL: module @join_producer_dims
// CHECK:   aie.device(xcve2302) {

// MemTile DMA: MM2S output must carry producer_dimensions
// CHECK:     aie.memtile_dma
// CHECK:       aie.dma_start(MM2S
// CHECK:       aie.dma_bd({{.*}} : memref<32xi32>, 0, 16, [<size = 2, stride = 4>, <size = 4, stride = 1>])

// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @join_producer_dims {
  aie.device(xcve2302) {
    %tile_2_0 = aie.tile(2, 0)
    %tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)

    // Source objectfifos: compute tiles -> MemTile (no dims).
    aie.objectfifo @join_src_a (%tile_2_2, {%tile_2_1}, 2 : i32)
        : !aie.objectfifo<memref<16xi32>>

    aie.objectfifo @join_src_b (%tile_2_3, {%tile_2_1}, 2 : i32)
        : !aie.objectfifo<memref<16xi32>>

    // Destination objectfifo: MemTile -> shim with dimensionsToStream.
    aie.objectfifo @join_out (%tile_2_1 dimensionsToStream [<size = 2, stride = 4>, <size = 4, stride = 1>],
                              {%tile_2_0}, 2 : i32)
        : !aie.objectfifo<memref<32xi32>>

    // Join link: 2 sources -> 1 destination through MemTile(2,1).
    aie.objectfifo.link [@join_src_a, @join_src_b] -> [@join_out] ([0, 16][])

    %core_2_2 = aie.core(%tile_2_2) {
      %0 = aie.objectfifo.acquire @join_src_a(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @join_src_a(Produce, 1)
      aie.end
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %0 = aie.objectfifo.acquire @join_src_b(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @join_src_b(Produce, 1)
      aie.end
    }
  }
}
