// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
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

    conduit.create @src {slot_elems = 128 : i64, element_type = memref<64xi32>, depth = 2 : i64}
    conduit.create @dst {slot_elems = 128 : i64, element_type = memref<64xi32>, depth = 2 : i64}

    conduit.scatter{src = @src, dsts = [@dst] {memtile = "tile(0,1)"}}

    aie.shim_dma_allocation @src_shim_alloc(%shim_0_0, MM2S, 0) {conduit_channel = @src}

    // Consumer cores — structural info for tile inference.
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = conduit.acquire {count = 1 : i64, name = @dst,
                            port = #conduit.port<Consume>} : <memref<64xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Consume>} : <memref<64xi32>>
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %0 = conduit.acquire {count = 1 : i64, name = @dst,
                            port = #conduit.port<Consume>} : <memref<64xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Consume>} : <memref<64xi32>>
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %0 = conduit.acquire {count = 1 : i64, name = @dst,
                            port = #conduit.port<Consume>} : <memref<64xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Consume>} : <memref<64xi32>>
      aie.end
    }
  }
}
