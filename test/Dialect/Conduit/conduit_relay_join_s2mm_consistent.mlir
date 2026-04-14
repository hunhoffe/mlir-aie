// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression test: B-2 — S2MM channel consistency when a MemTile is both a
// relay source (for a join) and a broadcast consumer.
//
// Topology:
//   tile(2,2) [join source A] → MemTile(2,1) [join hub] → shim(2,0)
//   tile(3,2) [join source B] → MemTile(2,1) [join hub]

// CHECK-LABEL: module @conduit_relay_join_s2mm_consistent
// CHECK: aie.device

// Source A flow: tile(2,2) → MemTile(2,1) S2MM 0
// CHECK: aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 0)
// Source B flow: tile(3,2) → MemTile(2,1) S2MM 1
// CHECK: aie.flow(%tile_3_2, DMA : 0, %mem_tile_2_1, DMA : 1)

// MemTile DMA: 2 S2MM channels (0 for src_a, 1 for src_b) + 1 MM2S
// CHECK: aie.memtile_dma(%mem_tile_2_1)
// CHECK:   aie.dma_start(S2MM, 0,
// CHECK:   aie.dma_start(S2MM, 1,
// CHECK:   aie.dma_start(MM2S, 0,
// CHECK:   aie.end

// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.gather

module @conduit_relay_join_s2mm_consistent {
  aie.device(npu2) {
    %shim = aie.tile(2, 0)
    %mem_tile = aie.tile(2, 1)
    %tile_a = aie.tile(2, 2)
    %tile_b = aie.tile(3, 2)

    // Source A: tile(2,2) → MemTile(2,1)
    conduit.create @src_a {slot_elems = 128 : i64, element_type = memref<64xi8>, depth = 2 : i64}
    // Source B: tile(3,2) → MemTile(2,1)
    conduit.create @src_b {slot_elems = 128 : i64, element_type = memref<64xi8>, depth = 2 : i64}
    // Joined output: MemTile(2,1) → shim(2,0)
    conduit.create @join_out {slot_elems = 256 : i64, element_type = memref<128xi8>, depth = 2 : i64}

    // Join link: combine src_a + src_b → join_out
    conduit.gather{srcs = [@src_a, @src_b], dst = @join_out {memtile = "tile(2,1)", offsets = array<i64: 0, 64>}}

    // Shim consumer allocation for join output.
    aie.shim_dma_allocation @join_out_shim_alloc(%shim, S2MM, 0) {conduit_channel = @join_out}

    // Producer cores — structural info for tile inference.
    %core_a = aie.core(%tile_a) {
      %0 = conduit.acquire {count = 1 : i64, name = @src_a,
                            port = #conduit.port<Produce>} : <memref<64xi8>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<64xi8>>
      aie.end
    }
    %core_b = aie.core(%tile_b) {
      %0 = conduit.acquire {count = 1 : i64, name = @src_b,
                            port = #conduit.port<Produce>} : <memref<64xi8>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<64xi8>>
      aie.end
    }
  }
}
