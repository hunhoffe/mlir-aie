// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression test: MemTile relay channel feeding into a multi-source join.
//
// Topology:
//   tile(3,2) → MemTile(3,1) ──relay──→ MemTile(5,1) ─┐
//   tile(5,2) → MemTile(5,1) ─────────────────────────────┼── join ──→ shim(5,0)

// CHECK-LABEL: module @memtile_relay_join_source
// CHECK:   aie.device(npu2) {

// --- Shim DMA allocation for join output ---
// CHECK: aie.shim_dma_allocation @join_out_shim_alloc

// --- MemTile(5,1)→shim output flow ---
// CHECK: aie.flow(%mem_tile_5_1, DMA : 0, %shim_noc_tile_5_0, DMA : 0)

// --- Relay flow: MemTile(3,1)→MemTile(5,1) ---
// CHECK: aie.flow(%mem_tile_3_1, DMA : 0, %mem_tile_5_1, DMA : 0)

// --- Compute→MemTile flow for source A ---
// CHECK: aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 0)

// --- MemTile(3,1) relay DMA: 1 S2MM + 1 MM2S ---
// CHECK:     aie.memtile_dma(%mem_tile_3_1) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.dma_start(MM2S, 0,
// CHECK:       aie.end
// CHECK:     }

// --- Compute→MemTile flow for source B ---
// CHECK: aie.flow(%tile_5_2, DMA : 0, %mem_tile_5_1, DMA : 1)

// --- MemTile(5,1) join DMA: 2 S2MM + 1 MM2S ---
// CHECK:     aie.memtile_dma(%mem_tile_5_1) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.dma_start(S2MM, 1,
// CHECK:       aie.dma_start(MM2S, 0,
// CHECK:       aie.end
// CHECK:     }

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.scatter
// CHECK-NOT: conduit.gather

module @memtile_relay_join_source {
  aie.device(npu2) {
    %shim_5_0 = aie.tile(5, 0)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %tile_3_2 = aie.tile(3, 2)
    %tile_5_2 = aie.tile(5, 2)

    // Source A: compute tile(3,2) → local MemTile(3,1)
    conduit.create @prod_a {element_type = memref<64xi8>, depth = 2 : i64}
    // Relay: MemTile(3,1) → MemTile(5,1)
    conduit.create @relay {element_type = memref<64xi8>, depth = 2 : i64}
    // Source B: compute tile(5,2) → local MemTile(5,1)
    conduit.create @prod_b {element_type = memref<64xi8>, depth = 2 : i64}
    // Joined output: MemTile(5,1) → shim(5,0)
    conduit.create @join_out {element_type = memref<128xi8>, depth = 2 : i64}

    // Relay link at MemTile(3,1): forward prod_a → relay
    conduit.scatter{src = @prod_a, dsts = [@relay] {memtile = "tile(3,1)"}}

    // Join link at MemTile(5,1): combine relay + prod_b → join_out
    conduit.gather{srcs = [@relay, @prod_b], dst = @join_out {memtile = "tile(5,1)", offsets = array<i64: 0, 64>}}

    // Shim consumer allocation for join output.
    aie.shim_dma_allocation @join_out_shim_alloc(%shim_5_0, S2MM, 0) {conduit_channel = @join_out}

    // Producer cores — structural info for tile inference.
    %core_3_2 = aie.core(%tile_3_2) {
      %0 = conduit.acquire {count = 1 : i64, name = @prod_a,
                            port = #conduit.port<Produce>} : <memref<64xi8>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<64xi8>>
      aie.end
    }
    %core_5_2 = aie.core(%tile_5_2) {
      %0 = conduit.acquire {count = 1 : i64, name = @prod_b,
                            port = #conduit.port<Produce>} : <memref<64xi8>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<64xi8>>
      aie.end
    }
  }
}
