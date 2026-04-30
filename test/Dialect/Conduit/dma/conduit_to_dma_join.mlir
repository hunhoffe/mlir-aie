// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C test: 3 producers -> 1 consumer (join link).
//
// Fix 2: Pass C now generates N independent S2MM channels (one per source),
// each with its own BD ring using the source conduit's lock pair.
// The MM2S channel outputs the joined destination buffer.
// Flows: source compute tiles → memtile S2MM channels i, memtile → shim.

// CHECK-LABEL: module @link_join_offsets
// CHECK:   aie.device(xcve2302) {
// --- Exactly one shim_dma_allocation for the join destination (link4),
//     direction S2MM channel 0 ---
// CHECK:     aie.shim_dma_allocation @link4_shim_alloc(%{{.*}}shim{{.*}}2_0, S2MM, 0)
// CHECK-NOT: aie.shim_dma_allocation @link4_shim_alloc(
// --- Shim-side locks for link4 consumer endpoint (init=0, host programs these) ---
// CHECK:     aie.lock(%{{.*}}shim{{.*}}2_0
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "link4_cons_prod_lock_0"
// CHECK:     aie.lock(%{{.*}}shim{{.*}}2_0
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "link4_cons_cons_lock_0"
// --- memtile→shim flow emitted by Phase 4b immediately after shim locks ---
// CHECK:     aie.flow(%{{.*}}mem_tile_2_1, DMA : 0, %{{.*}}shim{{.*}}2_0, DMA : 0)
// --- Join destination buffers (2 buffers on memtile) ---
// CHECK:     aie.buffer(%{{.*}}mem_tile_2_1) {{.*}} memref<48xi32>
// CHECK:     aie.buffer(%{{.*}}mem_tile_2_1) {{.*}} memref<48xi32>
// --- 6 per-source lock pairs on memtile (3 sources × 2 locks) ---
// CHECK:     aie.lock(%{{.*}}mem_tile_2_1, 0) {init = 2
// CHECK:     aie.lock(%{{.*}}mem_tile_2_1, 1) {init = 0
// CHECK:     aie.lock(%{{.*}}mem_tile_2_1, 2) {init = 2
// CHECK:     aie.lock(%{{.*}}mem_tile_2_1, 3) {init = 0
// CHECK:     aie.lock(%{{.*}}mem_tile_2_1, 4) {init = 2
// CHECK:     aie.lock(%{{.*}}mem_tile_2_1, 5) {init = 0
// --- 3 per-source flows: compute tiles → memtile S2MM channels 0,1,2 ---
// CHECK:     aie.flow(%{{.*}}tile_2_2, DMA : 0, %{{.*}}mem_tile_2_1, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_2_3, DMA : 0, %{{.*}}mem_tile_2_1, DMA : 1)
// CHECK:     aie.flow(%{{.*}}tile_3_3, DMA : 0, %{{.*}}mem_tile_2_1, DMA : 2)
// --- MemTile DMA for join: 3 S2MM channels ingesting slices, 1 MM2S channel output ---
// CHECK:     aie.memtile_dma(%{{.*}}mem_tile_2_1) {
// Three S2MM starts (channels 0, 1, 2 — one per source)
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.dma_start(S2MM, 1,
// CHECK:       aie.dma_start(S2MM, 2,
// One MM2S start (channel 0 — output the joined buffer)
// CHECK:       aie.dma_start(MM2S, 0,
// CHECK:       aie.end
// CHECK:     }
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.gather

module @link_join_offsets {
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

    // Join link: 3 sources → 1 destination with byte offsets at MemTile(2,1)
    conduit.gather{srcs = [@link1, @link2, @link3], dst = @link4, memtile = %tile21, offsets = [0, 16, 36]}

    // Shim consumer allocation for output channel.
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
