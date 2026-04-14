// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// C5: Three-member fused chain ordering test.
//
// Three conduits (chan_a, chan_b, chan_c) all share producer tile [0,2].
// All are pre-annotated with fused_dma_channel_group = "group0".
// Pass C must emit exactly one dma_start(MM2S, 0) and chain all three
// depth-1 BD rings into a single circular list:
//
//   dma_start(MM2S, 0) → chan_a_BD → chan_b_BD → chan_c_BD → chan_a_BD
//
// Hardware topology:
//   tile[0,2] = producer (compute)
//   tile[0,4] = consumer of chan_a (non-adjacent)
//   tile[0,5] = consumer of chan_b (non-adjacent)
//   tile[1,4] = consumer of chan_c (different column, non-adjacent)
//   npu1 device (multi-column)

// CHECK-LABEL: module @fuse_3member_test

// C2: Three flows all on the same MM2S channel 0 (single fused group).
// CHECK-DAG: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_4, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_5, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_1_4, DMA : 0)

// C5: Three-member chain — A→B→C→A ordering verified.
// One dma_start(MM2S, 0) followed by three BD blocks chained in order.
// CHECK:     aie.mem(%{{.*}}tile_0_2)
// CHECK:       aie.dma_start(MM2S, 0, [[BD_A:\^bb[0-9]+]],
// CHECK:       aie.dma_bd(%chan_a_buff_0
// CHECK:       aie.next_bd [[BD_B:\^bb[0-9]+]]
// CHECK:       aie.dma_bd(%chan_b_buff_0
// CHECK:       aie.next_bd [[BD_C:\^bb[0-9]+]]
// CHECK:       aie.dma_bd(%chan_c_buff_0
// CHECK:       aie.next_bd [[BD_A]]
// CHECK-NOT:   aie.dma_start(MM2S, 1,

// No residual conduit ops.
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire

module @fuse_3member_test {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_4 = aie.tile(1, 4)

    conduit.create @chan_a {slot_elems = 8 : i64,
                    element_type = memref<8xi32>, depth = 1 : i64,
                    fuse_mode = "static",
                    fused_dma_channel_group = "group0"}
    conduit.create @chan_b {slot_elems = 8 : i64,
                    element_type = memref<8xi32>, depth = 1 : i64,
                    fuse_mode = "static",
                    fused_dma_channel_group = "group0"}
    conduit.create @chan_c {slot_elems = 8 : i64,
                    element_type = memref<8xi32>, depth = 1 : i64,
                    fuse_mode = "static",
                    fused_dma_channel_group = "group0"}

    // Minimal aie.core blocks for inferAllTiles() Source 1.
    aie.core(%tile_0_2) {
      %w_a = conduit.acquire {name = @chan_a, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<8xi32>>
      conduit.release %w_a {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<8xi32>>
      %w_b = conduit.acquire {name = @chan_b, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<8xi32>>
      conduit.release %w_b {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<8xi32>>
      %w_c = conduit.acquire {name = @chan_c, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<8xi32>>
      conduit.release %w_c {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<8xi32>>
      aie.end
    }
    aie.core(%tile_0_4) {
      %w = conduit.acquire {name = @chan_a, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<8xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<8xi32>>
      aie.end
    }
    aie.core(%tile_0_5) {
      %w = conduit.acquire {name = @chan_b, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<8xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<8xi32>>
      aie.end
    }
    aie.core(%tile_1_4) {
      %w = conduit.acquire {name = @chan_c, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<8xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<8xi32>>
      aie.end
    }
  }
}
