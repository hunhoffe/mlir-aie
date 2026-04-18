// RUN: aie-opt --objectfifo-to-conduit --conduit-depth-promote --conduit-to-dma --aie-assign-buffer-addresses %s | FileCheck %s
//
// Regression test (A-1): when two join sources share the same compute producer
// tile, the join path must allocate distinct MM2S channels (0 and 1) on that
// tile instead of hardcoding channel 0 for both.
//
// The A-1 fix (ConduitToDMALink.cpp lines 827-830) uses dynamic
// tileNextMM2SChannel allocation per source producer tile.
//
// Topology:
//   tile(0,2) [compute producer — TWO objectfifos from same tile]
//     @src_a (64 bytes) → MemTile(0,1) [join hub]  (MM2S channel 0)
//     @src_b (64 bytes) → MemTile(0,1) [join hub]  (MM2S channel 1)
//   MemTile(0,1) → shim(0,0) via @join_out (128 bytes joined)
//
// Expected: tile(0,2)'s aie.mem block has two aie.dma_start(MM2S, ...) with
//   channel indices 0 and 1 (not both 0).
//
// CHECK-LABEL: module @multi_scatter_same_producer
// CHECK: aie.device(npu2)

// Verify: both source flows from tile(0,2) use distinct MM2S channels.
// CHECK: aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA :
// CHECK: aie.flow(%tile_0_2, DMA : 1, %mem_tile_0_1, DMA :

// Verify: tile(0,2) aie.mem has two MM2S DMA starts with distinct channels.
// CHECK:     aie.mem(%tile_0_2) {
// CHECK:       aie.dma_start(MM2S, 0,
// CHECK:       aie.dma_start(MM2S, 1,
// CHECK:     }

// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.link

module @multi_scatter_same_producer {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %memtile = aie.tile(0, 1)
    %tile02 = aie.tile(0, 2)

    // Two objectfifos from the SAME compute tile(0,2) to the MemTile.
    aie.objectfifo @src_a (%tile02, {%memtile}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @src_b (%tile02, {%memtile}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    // Joined output: MemTile(0,1) → shim(0,0).
    aie.objectfifo @join_out (%memtile, {%shim}, 2 : i32) : !aie.objectfifo<memref<128xi8>>

    // Join link: combine src_a + src_b → join_out with byte offsets.
    aie.objectfifo.link [@src_a, @src_b] -> [@join_out] ([0, 64][])

    // Producer core: acquires from both fifos and produces data.
    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %sv_a = aie.objectfifo.acquire @src_a(Produce, 1) :
            !aie.objectfifosubview<memref<64xi8>>
        %elem_a = aie.objectfifo.subview.access %sv_a[0] :
            !aie.objectfifosubview<memref<64xi8>> -> memref<64xi8>
        aie.objectfifo.release @src_a(Produce, 1)

        %sv_b = aie.objectfifo.acquire @src_b(Produce, 1) :
            !aie.objectfifosubview<memref<64xi8>>
        %elem_b = aie.objectfifo.subview.access %sv_b[0] :
            !aie.objectfifosubview<memref<64xi8>> -> memref<64xi8>
        aie.objectfifo.release @src_b(Produce, 1)
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
