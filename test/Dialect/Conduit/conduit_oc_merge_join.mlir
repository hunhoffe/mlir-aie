// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | aie-opt -o /dev/null
//
// 4-way output-channel join test (Conv7-like pattern).
//
// 4 compute tiles each produce a partial output-channel slice.
// A MemTile joins them into a single contiguous buffer for DMA out.
//
// Tile layout (npu2, 4 columns):
//   tile(0,2), tile(1,2), tile(2,2), tile(3,2) → MemTile(0,1) join → shim(0,0)
//
// Each source produces memref<64xi8> (256B total).
// Join destination: memref<256xi8>.
//
// Expected resources on MemTile:
//   aie.lock: 8  (4 sources × 2 locks per source)
//   aie.flow: 5  (4 compute→memtile + 1 memtile→shim)
//   aie.dma_start(S2MM): 4  (one per source)
//   aie.dma_start(MM2S): 1  (joined output)

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
// CHECK-NOT: conduit.link

module @oc_merge_join {
  aie.device(npu2) {
    %shim_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)

    // 4 source objectfifos: each compute tile → MemTile
    aie.objectfifo @oc_src0 (%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @oc_src1 (%tile_1_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @oc_src2 (%tile_2_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xi8>>
    aie.objectfifo @oc_src3 (%tile_3_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xi8>>

    // Joined destination: MemTile → shim
    aie.objectfifo @oc_dst (%mem_tile_0_1, {%shim_0_0}, 2 : i32) : !aie.objectfifo<memref<256xi8>>

    // Join link: 4 sources at byte offsets 0, 64, 128, 192
    aie.objectfifo.link [@oc_src0, @oc_src1, @oc_src2, @oc_src3] -> [@oc_dst] ([0, 64, 128, 192][])
  }
}
