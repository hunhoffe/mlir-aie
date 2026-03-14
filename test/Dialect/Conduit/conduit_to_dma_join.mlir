// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: 3 producers -> 1 consumer (join link).
//
// Fix 2: Pass C now generates N independent S2MM channels (one per source),
// each with its own BD ring using the source conduit's lock pair.
// The MM2S channel outputs the joined destination buffer.
// Flows: source compute tiles → memtile S2MM channels i, memtile → shim.
//
// Ground truth (from --aie-objectFifo-stateful-transform on the same input):
//   Total aie.buffer:  8  (link1: 2 on tile_2_2; link2: 2 on tile_2_3;
//                          link3: 2 on tile_3_3; link4: 2 on memtile_2_1)
//   Total aie.lock:   14  (link4: 2 on shim + 6 on memtile + link3: 2 + link2: 2 + link1: 2)
//   Total aie.flow:    4  (tile_2_2→memtile, tile_2_3→memtile, tile_3_3→memtile,
//                          memtile→shim_2_0)
//   aie.dma_start(S2MM): 3  (channels 0,1,2 in memtile_dma — one per src fifo)
//   aie.dma_start(MM2S): 4  (1 in memtile_dma + 3 in compute tile aie.mem blocks)
//   aie.dma_bd: 18 total
//   aie.next_bd: 18 total
//
// The join lowering pattern in aie.memtile_dma:
//   - S2MM channel 0 ingests link4_buff slices for link1 (offset 0, len 16)
//   - S2MM channel 1 ingests link4_buff slices for link2 (offset 16, len 20)
//   - S2MM channel 2 ingests link4_buff slices for link3 (offset 36, len 12)
//   - MM2S channel 0 outputs the joined link4 buffer (full 48 elements)
//
// This test also verifies the compute tile aie.mem blocks (MM2S from each src tile).

// CHECK-LABEL: module @link_join_offsets
// CHECK:   aie.device(xcve2302) {
// --- Four flows: 3 compute tiles to memtile (S2MM channels 0,1,2),
//     and memtile MM2S channel 0 to shim ---
// CHECK:     aie.flow(%{{.*}}tile_2_2, DMA : 0, %{{.*}}mem_tile_2_1, DMA : 0)
// CHECK:     aie.flow(%{{.*}}tile_2_3, DMA : 0, %{{.*}}mem_tile_2_1, DMA : 1)
// CHECK:     aie.flow(%{{.*}}tile_3_3, DMA : 0, %{{.*}}mem_tile_2_1, DMA : 2)
// CHECK:     aie.flow(%{{.*}}mem_tile_2_1, DMA : 0, %{{.*}}shim{{.*}}2_0, DMA : 0)
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
// CHECK-NOT: conduit.link

module @link_join_offsets {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    // Three sources join into one destination at the MemTile (tile21)
    aie.objectfifo @link1 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link2 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link3 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
    aie.objectfifo @link4 (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    // Join: 3 sources -> 1 destination with byte offsets
    aie.objectfifo.link [@link1, @link2, @link3] -> [@link4] ([0, 16, 36][])
  }
}
