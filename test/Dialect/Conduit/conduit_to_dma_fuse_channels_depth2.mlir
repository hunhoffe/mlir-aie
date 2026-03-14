// RUN: aie-opt --conduit-to-dma %s | FileCheck %s --check-prefix=FUSED2
// RUN: aie-opt --conduit-fuse-channels --conduit-to-dma %s | FileCheck %s --check-prefix=FUSED2ANNO
//
// C3/C12: depth-2 fused channel test.
//
// Tests C3 (depth > 1 BD chain linking) and C12 (pre-annotated input bypassing
// the annotation pass).
//
// Both conduit.create ops have fuse_mode = "static" and
// fused_dma_channel_group = "group0" pre-annotated in the source.
// Pass C must chain the two depth-2 BD rings into one circular list:
//
//   dma_start(MM2S, 0) → chan_a_BD0 → chan_a_BD1 → chan_b_BD0 → chan_b_BD1 → chan_a_BD0
//
// No aie.core bodies are present to avoid the rotation-counter memref.load
// crash that occurs for depth>1 conduits with consumer acquire ops.
//
// Hardware topology:
//   tile[0,2] = producer tile (compute)
//   tile[0,4] = consumer of chan_a (non-adjacent)
//   tile[0,5] = consumer of chan_b (non-adjacent)
//   npu1_1col device, column 0

// FUSED2-LABEL: module @fuse_depth2_nocore
// FUSED2: aie.device(npu1_1col)

// C9: Independent lock pairs — depth=2 uses init=2 for producer locks.
// FUSED2-DAG: aie.lock(%{{.*}}tile_0_2, {{.*}}) {init = 2 : i32, sym_name = "chan_a_prod_lock_0"}
// FUSED2-DAG: aie.lock(%{{.*}}tile_0_2, {{.*}}) {init = 0 : i32, sym_name = "chan_a_cons_lock_0"}
// FUSED2-DAG: aie.lock(%{{.*}}tile_0_2, {{.*}}) {init = 2 : i32, sym_name = "chan_b_prod_lock_0"}
// FUSED2-DAG: aie.lock(%{{.*}}tile_0_2, {{.*}}) {init = 0 : i32, sym_name = "chan_b_cons_lock_0"}

// C2: Both flows use shared MM2S channel 0.
// FUSED2: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_4, DMA : 0)
// FUSED2: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_5, DMA : 0)

// C3: depth-2 fused chain — verify the 4-BD circular ring structure.
// dma_start → chan_a_BD0 → chan_a_BD1 → chan_b_BD0 → chan_b_BD1 → chan_a_BD0 (wrap)
// FUSED2:     aie.mem(%{{.*}}tile_0_2)
// FUSED2:       aie.dma_start(MM2S, 0, [[CHAN_A_BD0:\^bb[0-9]+]],
// FUSED2:       aie.dma_bd(%chan_a_buff_0
// FUSED2:       aie.next_bd [[CHAN_A_BD1:\^bb[0-9]+]]
// FUSED2:       aie.dma_bd(%chan_a_buff_1
// FUSED2:       aie.next_bd [[CHAN_B_BD0:\^bb[0-9]+]]
// (chan_a BD1 now chains to chan_b's first BD — NOT back to chan_a_BD0)
// FUSED2:       aie.dma_bd(%chan_b_buff_0
// FUSED2:       aie.next_bd [[CHAN_B_BD1:\^bb[0-9]+]]
// FUSED2:       aie.dma_bd(%chan_b_buff_1
// FUSED2:       aie.next_bd [[CHAN_A_BD0]]
// (chan_b BD1 wraps back to chan_a's first BD — completing the inter-ring chain)

// C12: pre-annotated input works identically without the annotation pass.
// FUSED2ANNO-LABEL: module @fuse_depth2_nocore
// FUSED2ANNO:   aie.dma_start(MM2S, 0,
// FUSED2ANNO-NOT: aie.dma_start(MM2S, 1,

// No residual conduit ops.
// FUSED2-NOT: conduit.create
// FUSED2-NOT: conduit.acquire
// FUSED2-NOT: conduit.release
// FUSED2ANNO-NOT: conduit.create

module @fuse_depth2_nocore {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)

    conduit.create {name = "chan_a", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 4>,
                    element_type = memref<4xi32>,
                    depth = 2 : i64,
                    fuse_mode = "static",
                    fused_dma_channel_group = "group0"}
    conduit.create {name = "chan_b", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 5>,
                    element_type = memref<4xi32>,
                    depth = 2 : i64,
                    fuse_mode = "static",
                    fused_dma_channel_group = "group0"}
  }
}
