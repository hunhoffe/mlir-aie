// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Resource parity test: join lowering must produce the same lock/buffer/flow
// counts as the oracle (--aie-objectFifo-stateful-transform).
//
// Expected resource counts:
//   aie.buffer:   8  (link1: 2; link2: 2; link3: 2; link4: 2 on memtile)
//   aie.lock:    14  (link4: 2 on shim + 6 on memtile; link1/2/3: 2 each)
//   aie.flow:     4  (tile_2_2→memtile, tile_2_3→memtile, tile_3_3→memtile,
//                     mem_tile_2_1→shim_2_0)

// CHECK-LABEL: module @link_join_parity
// CHECK: aie.device(xcve2302)
//
// Verify shim alloc and memtile→shim flow (Phase 4b).
// CHECK: aie.shim_dma_allocation @link4_shim_alloc
// CHECK: aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
//
// Verify exactly 6 locks on memtile (3 pairs for 3 sources — no extra pair).
// CHECK: aie.lock(%mem_tile_2_1, 0) {init = 2
// CHECK: aie.lock(%mem_tile_2_1, 1) {init = 0
// CHECK: aie.lock(%mem_tile_2_1, 2) {init = 2
// CHECK: aie.lock(%mem_tile_2_1, 3) {init = 0
// CHECK: aie.lock(%mem_tile_2_1, 4) {init = 2
// CHECK: aie.lock(%mem_tile_2_1, 5) {init = 0
//
// Verify the 3 per-source flows.
// CHECK: aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 0)
// CHECK: aie.flow(%tile_2_3, DMA : 0, %mem_tile_2_1, DMA : 1)
// CHECK: aie.flow(%tile_3_3, DMA : 0, %mem_tile_2_1, DMA : 2)
//
// MemTile DMA: 3 S2MM + 1 MM2S.
// CHECK: aie.memtile_dma(%mem_tile_2_1)
// CHECK:   aie.dma_start(S2MM, 0,
// CHECK:   aie.dma_start(S2MM, 1,
// CHECK:   aie.dma_start(S2MM, 2,
// CHECK:   aie.dma_start(MM2S, 0,
// CHECK:   aie.end

// No leftover conduit ops.
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.gather

module @link_join_parity {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    // Three source conduits: compute tiles → MemTile
    conduit.create @link1 {element_type = memref<4x4xi32>, depth = 2 : i64}
    conduit.create @link2 {element_type = memref<20xi32>, depth = 2 : i64}
    conduit.create @link3 {element_type = memref<12xi32>, depth = 2 : i64}
    // Join destination: MemTile → shim
    conduit.create @link4 {element_type = memref<48xi32>, depth = 2 : i64}

    // Join link at MemTile(2,1)
    conduit.gather{srcs = [@link1, @link2, @link3], dst = @link4 {memtile = "tile(2,1)", offsets = array<i64: 0, 16, 36>}}

    // Shim consumer allocation for join output.
    aie.shim_dma_allocation @link4_shim_alloc(%tile20, S2MM, 0) {conduit_channel = @link4}

    // Producer cores — structural info for tile inference.
    %core_2_2 = aie.core(%tile22) {
      %0 = conduit.acquire {count = 1 : i64, name = @link1,
                            port = #conduit.port<Produce>} : <memref<4x4xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<4x4xi32>>
      aie.end
    }
    %core_2_3 = aie.core(%tile23) {
      %0 = conduit.acquire {count = 1 : i64, name = @link2,
                            port = #conduit.port<Produce>} : <memref<20xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<20xi32>>
      aie.end
    }
    %core_3_3 = aie.core(%tile33) {
      %0 = conduit.acquire {count = 1 : i64, name = @link3,
                            port = #conduit.port<Produce>} : <memref<12xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<12xi32>>
      aie.end
    }
  }
}
