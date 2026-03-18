// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test for Task #54: broadcast link distribute flow emission.
//
// Bug: conduit.link distribute mode with a destination conduit that has N>1
// consumer tiles (broadcast) previously emitted only 1 aie.flow (to
// consumerTileCoords[0]).  Consumers 2..N received no flow and would deadlock
// waiting for a lock grant that never fires.
//
// Fix: ConduitToDMALink.cpp now loops over all consumerTileCoords for each
// distribute destination, emitting one aie.flow per consumer tile.
//
// This test has 1 link:
//   @src: shim(0,0) → memtile(0,1)     [shim producer → memtile consumer]
//   @dst: memtile(0,1) → {tile(0,2), tile(0,3), tile(0,4)}  [3 consumers]
//
// Expected: 4 aie.flow ops total
//   1. shim(0,0) → memtile(0,1)    [Phase 4a: shim producer flow]
//   2. memtile(0,1) → tile(0,2)    [Phase 5 distribute: consumer 0]
//   3. memtile(0,1) → tile(0,3)    [Phase 5 distribute: consumer 1]
//   4. memtile(0,1) → tile(0,4)    [Phase 5 distribute: consumer 2]
//
// The regression is: without the fix, only flows 1 and 2 were emitted.

// Verify each specific flow is present (DAG order independent).
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_3, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_4, DMA : 0)

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %shim_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)

    aie.objectfifo @src(%shim_0_0, {%mem_tile_0_1}, 2 : i32)
        : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo @dst(%mem_tile_0_1, {%tile_0_2, %tile_0_3, %tile_0_4}, 2 : i32)
        : !aie.objectfifo<memref<64xi32>>

    aie.objectfifo.link [@src] -> [@dst]([] [])
  }
}
