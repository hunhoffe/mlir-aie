// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// C6: Two fused groups on the same producer tile (interleaved pairs scenario).
//
// Four conduits share producer tile [0,2]:
//   group0 = {chan_a, chan_c}  → assigned MM2S channel 0
//   group1 = {chan_b, chan_d}  → assigned MM2S channel 1
//
// Pass C must emit TWO dma_start ops on tile[0,2] — one per group — each
// heading a circular BD chain covering its two member conduits:
//
//   MM2S 0: dma_start → chan_a_BD → chan_c_BD → chan_a_BD
//   MM2S 1: dma_start → chan_b_BD → chan_d_BD → chan_b_BD
//
// Hardware topology:
//   tile[0,2] = producer (compute)
//   tile[0,4] = consumer of chan_a (group0)
//   tile[0,5] = consumer of chan_b (group1)
//   tile[1,4] = consumer of chan_c (group0)
//   tile[1,5] = consumer of chan_d (group1)
//   npu1 device (multi-column)

// CHECK-LABEL: module @fuse_2groups_test

// C9: All four conduits have independent lock pairs.
// CHECK-DAG: sym_name = "chan_a_prod_lock_0"
// CHECK-DAG: sym_name = "chan_b_prod_lock_0"
// CHECK-DAG: sym_name = "chan_c_prod_lock_0"
// CHECK-DAG: sym_name = "chan_d_prod_lock_0"

// C2: Four flows — group 0 on DMA:0, group 1 on DMA:1.
// CHECK-DAG: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_4, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_1_4, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}tile_0_2, DMA : 1, %{{.*}}tile_0_5, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}tile_0_2, DMA : 1, %{{.*}}tile_1_5, DMA : 0)

// C6: Two dma_start ops on the producer tile — one per group.
// C1: The mem body structure (from verified aie-opt output):
//   ^bb0: dma_start(MM2S, 0, ^bb1, ^bb2)   ← group0 on channel 0
//   ^bb1: [chan_a BD]  next_bd → chan_c     ← group0 chain
//   ^bb2: dma_start(MM2S, 1, ^bb3, ^bb4)   ← group1 on channel 1 (inside ^bb2)
//   ^bb3: [chan_b BD]  next_bd → chan_d     ← group1 chain
//   ^bb5: [chan_c BD]  next_bd → chan_a     ← group0 wrap
//   ^bb7: [chan_d BD]  next_bd → chan_b     ← group1 wrap
// CHECK:     aie.mem(%{{.*}}tile_0_2)
// CHECK:       aie.dma_start(MM2S, 0, [[BD_A:\^bb[0-9]+]],
// CHECK:       aie.dma_bd(%chan_a_buff_0
// CHECK:       aie.next_bd [[BD_C:\^bb[0-9]+]]
// CHECK:       aie.dma_start(MM2S, 1, [[BD_B:\^bb[0-9]+]],
// CHECK:       aie.dma_bd(%chan_b_buff_0
// CHECK:       aie.next_bd [[BD_D:\^bb[0-9]+]]
// CHECK:       aie.dma_bd(%chan_c_buff_0
// CHECK:       aie.next_bd [[BD_A]]
// CHECK:       aie.dma_bd(%chan_d_buff_0
// CHECK:       aie.next_bd [[BD_B]]

// No residual conduit ops.
// CHECK-NOT: conduit.create

module @fuse_2groups_test {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)

    conduit.create {name = "chan_a", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 4>,
                    element_type = memref<8xi32>, depth = 1 : i64,
                    fuse_mode = "static",
                    fused_dma_channel_group = "group0"}
    conduit.create {name = "chan_b", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 5>,
                    element_type = memref<8xi32>, depth = 1 : i64,
                    fuse_mode = "static",
                    fused_dma_channel_group = "group1"}
    conduit.create {name = "chan_c", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 4>,
                    element_type = memref<8xi32>, depth = 1 : i64,
                    fuse_mode = "static",
                    fused_dma_channel_group = "group0"}
    conduit.create {name = "chan_d", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 5>,
                    element_type = memref<8xi32>, depth = 1 : i64,
                    fuse_mode = "static",
                    fused_dma_channel_group = "group1"}
  }
}
