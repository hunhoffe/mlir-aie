// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test: consumer_dimensions (dimensionsFromStream) must be
// preserved on the MemTile relay S2MM BD.
//
// Bug: ConduitToDMALink.cpp was dropping consumer_dimensions when emitting
// relay S2MM BDs, so the MemTile DMA ingested data with linear layout
// instead of the requested transposed layout.
//
// Topology:
//   shim(0,0) → MemTile(0,1) → compute(0,2)
//   of_in: shim→MemTile, depth=2, consumer dimensionsFromStream=[4x8 transpose]
//   of_out: MemTile→compute, depth=2
//   objectfifo.link chains of_in→of_out through MemTile(0,1)
//
// Expected:
//   The MemTile S2MM BD (ingest side) must carry the consumer dimensions
//   [<size = 4, stride = 8>, <size = 8, stride = 1>].

// CHECK-LABEL: module @relay_consumer_dims
// CHECK:   aie.device(npu1_1col) {

// MemTile DMA: S2MM ingest must carry consumer_dimensions
// CHECK:     aie.memtile_dma
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd({{.*}} : memref<32xi32>, 0, 32, [<size = 4, stride = 8>, <size = 8, stride = 1>])

// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @relay_consumer_dims {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    // Ingress: shim→MemTile with dimensionsFromStream (transpose layout).
    aie.objectfifo @of_in(%tile_0_0,
                          {%tile_0_1 dimensionsFromStream [<size = 4, stride = 8>, <size = 8, stride = 1>]},
                          2 : i32)
        : !aie.objectfifo<memref<32xi32>>

    // Egress: MemTile→compute (linear).
    aie.objectfifo @of_out(%tile_0_1, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<32xi32>>

    // Link through MemTile(0,1).
    aie.objectfifo.link [@of_in] -> [@of_out]([] [0])

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of_out(Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
      aie.objectfifo.release @of_out(Consume, 1)
      aie.end
    }
  }
}
