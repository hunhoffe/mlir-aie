// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --conduit-to-dma %s | aie-opt -o /dev/null
//
// 4-way output-channel join test (Conv7-like pattern).
//
// 4 compute tiles each produce a partial output-channel slice.
// A MemTile joins them into a single contiguous buffer for DMA out.
//
// Tile layout (npu2, 4 columns):
//   tile(0,2), tile(1,2), tile(2,2), tile(3,2) → MemTile(0,1) join → shim(0,0)

// CHECK-LABEL: module @oc_merge_join
// CHECK:   aie.device(npu2) {

// --- Verify 4 compute-to-memtile flows (one per source) ---
// CHECK-DAG: aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA :
// CHECK-DAG: aie.flow(%tile_1_2, DMA : 0, %mem_tile_0_1, DMA :
// CHECK-DAG: aie.flow(%tile_2_2, DMA : 0, %mem_tile_0_1, DMA :
// CHECK-DAG: aie.flow(%tile_3_2, DMA : 0, %mem_tile_0_1, DMA :

// --- MemTile DMA: 4 S2MM ingests + 1 MM2S output ---
// CHECK:     aie.memtile_dma(%mem_tile_0_1) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.dma_start(S2MM, 1,
// CHECK:       aie.dma_start(S2MM, 2,
// CHECK:       aie.dma_start(S2MM, 3,
// CHECK:       aie.dma_start(MM2S, 0,
// CHECK:       aie.end
// CHECK:     }

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.gather

module @oc_merge_join {
  aie.device(npu2) {
    %shim_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)

    // 4 source conduits: compute tiles → MemTile
    conduit.create @oc_src0 {slot_elems = 128 : i64, element_type = memref<64xi8>, depth = 2 : i64}
    conduit.create @oc_src1 {slot_elems = 128 : i64, element_type = memref<64xi8>, depth = 2 : i64}
    conduit.create @oc_src2 {slot_elems = 128 : i64, element_type = memref<64xi8>, depth = 2 : i64}
    conduit.create @oc_src3 {slot_elems = 128 : i64, element_type = memref<64xi8>, depth = 2 : i64}
    // Join destination: MemTile → shim
    conduit.create @oc_dst {slot_elems = 512 : i64, element_type = memref<256xi8>, depth = 2 : i64}

    // Join link: 4 sources at byte offsets 0, 64, 128, 192 at MemTile(0,1)
    conduit.gather{srcs = [@oc_src0, @oc_src1, @oc_src2, @oc_src3], dst = @oc_dst {memtile = "tile(0,1)", offsets = array<i64: 0, 64, 128, 192>}}

    // Shim consumer allocation for join output.
    aie.shim_dma_allocation @oc_dst_shim_alloc(%shim_0_0, S2MM, 0) {conduit_channel = @oc_dst}

    // Producer cores — structural info for tile inference.
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = conduit.acquire {count = 1 : i64, name = @oc_src0,
                            port = #conduit.port<Produce>} : <memref<64xi8>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<64xi8>>
      aie.end
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %0 = conduit.acquire {count = 1 : i64, name = @oc_src1,
                            port = #conduit.port<Produce>} : <memref<64xi8>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<64xi8>>
      aie.end
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %0 = conduit.acquire {count = 1 : i64, name = @oc_src2,
                            port = #conduit.port<Produce>} : <memref<64xi8>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<64xi8>>
      aie.end
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %0 = conduit.acquire {count = 1 : i64, name = @oc_src3,
                            port = #conduit.port<Produce>} : <memref<64xi8>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<64xi8>>
      aie.end
    }
  }
}
