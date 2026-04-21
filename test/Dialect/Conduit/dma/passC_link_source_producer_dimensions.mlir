// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test: producer_dimensions (dimensionsToStream) must be
// preserved on the compute tile MM2S BD when the conduit is a link source.
//
// Bug: ConduitToDMALink.cpp was dropping producer_dimensions from link
// source conduits, so compute tile DMAs sent data with linear layout
// instead of the requested strided layout.
//
// Topology:
//   compute(0,2) -> MemTile(0,1) -> shim(0,0)
//   of_src: compute->MemTile, depth=2, dimensionsToStream=[<size=4, stride=8>, <size=8, stride=1>]
//   of_dst: MemTile->shim, depth=2
//   objectfifo.link chains of_src -> of_dst through MemTile(0,1)
//
// Expected:
//   The compute tile MM2S BD must carry the producer dimensions
//   [<size = 4, stride = 8>, <size = 8, stride = 1>].

// CHECK-LABEL: module @link_source_producer_dims
// CHECK:   aie.device(npu1_1col) {

// Compute tile MM2S BD must carry producer_dimensions
// CHECK:     aie.mem(
// CHECK:       aie.dma_start(MM2S
// CHECK:       aie.dma_bd({{.*}} : memref<32xi32>, 0, 32, [<size = 4, stride = 8>, <size = 8, stride = 1>])

// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @link_source_producer_dims {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    // Source objectfifo: compute -> MemTile with dimensionsToStream.
    aie.objectfifo @of_src (%tile_0_2 dimensionsToStream [<size = 4, stride = 8>, <size = 8, stride = 1>],
                            {%tile_0_1}, 2 : i32)
        : !aie.objectfifo<memref<32xi32>>

    // Destination objectfifo: MemTile -> shim (linear).
    aie.objectfifo @of_dst (%tile_0_1, {%tile_0_0}, 2 : i32)
        : !aie.objectfifo<memref<32xi32>>

    // Link through MemTile(0,1).
    aie.objectfifo.link [@of_src] -> [@of_dst] ([] [0])

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of_src(Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
      aie.objectfifo.release @of_src(Produce, 1)
      aie.end
    }
  }
}
