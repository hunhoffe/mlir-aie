// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// C8: MemTile producer fused through Pass C.
//
// Verifies that when a fused group has a MemTile producer (row=1), Pass C:
//   - Emits aie.memtile_dma (not aie.mem) for the producer tile.
//   - Emits ONE dma_start(MM2S, 0, ...) inside memtile_dma (channel reuse).
//   - Chains mt_a's BD block to mt_b's BD block, which wraps back to mt_a.
//   - Emits two aie.flow ops both sourced from DMA : 0 (shared channel).
//   - Allocates independent lock pairs for each conduit on the MemTile.
//   - Consumer tiles get aie.mem S2MM rings (not memtile_dma).
//   - No residual conduit ops remain.
//
// Topology (npu1_1col):
//   MemTile producer: [0, 1]  (row=1 — isMemTile on AIE2/npu1)
//   Consumer A:       [0, 4]  (non-adjacent to row 1 — uses DMA path)
//   Consumer B:       [0, 5]  (non-adjacent to row 1 — uses DMA path)
//
// Pre-annotated: fused_dma_channel_group = "group0", fuse_mode = "static".
// This simulates the output of --conduit-fuse-channels for a MemTile producer.

// CHECK-LABEL: module @fuse_channels_memtile_test

// C9: Independent lock pairs — each conduit allocates its own producer lock on MemTile.
// CHECK-DAG: aie.lock(%{{.*}}mem_tile_0_1, {{[0-9]+}}) {init = 1 : i32, sym_name = "mt_a_prod_lock_0"}
// CHECK-DAG: aie.lock(%{{.*}}mem_tile_0_1, {{[0-9]+}}) {init = 1 : i32, sym_name = "mt_b_prod_lock_0"}

// C2: Two aie.flow ops both with DMA : 0 source channel (shared MM2S).
// CHECK-DAG: aie.flow(%{{.*}}mem_tile_0_1, DMA : 0, %{{.*}}tile_0_4, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}mem_tile_0_1, DMA : 0, %{{.*}}tile_0_5, DMA : 0)

// C8: aie.memtile_dma is emitted for the producer MemTile (not aie.mem).
// C6/C1: ONE dma_start(MM2S, 0, ...) inside memtile_dma body.
// CHECK: aie.memtile_dma(%{{.*}}mem_tile_0_1)
// CHECK:   aie.dma_start(MM2S, 0,
// CHECK-NOT: aie.dma_start(MM2S, 1,
// CHECK:   aie.dma_bd(%mt_a_buff_0
// CHECK:   aie.next_bd
// CHECK:   aie.dma_bd(%mt_b_buff_0
// CHECK:   aie.next_bd

// C10: Consumer tiles get aie.mem S2MM rings (not memtile_dma).
// CHECK: aie.mem(%{{.*}}tile_0_4)
// CHECK:   aie.dma_start(S2MM, 0,
// CHECK: aie.mem(%{{.*}}tile_0_5)
// CHECK:   aie.dma_start(S2MM, 0,

// No residual Conduit ops.
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @fuse_channels_memtile_test {
  aie.device(npu1_1col) {
    func.func @process_a(%buf: memref<8xi32>) -> () {
      return
    }
    func.func @process_b(%buf: memref<8xi32>) -> () {
      return
    }

    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)

    // Two conduits sharing MemTile producer [0,1].
    // mt_a: producer=[0,1] (MemTile row=1), consumer=[0,4]
    // mt_b: producer=[0,1] (MemTile row=1), consumer=[0,5]
    // Pre-annotated with fused_dma_channel_group = "group0", fuse_mode = "static".
    conduit.create {name = "mt_a", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 1>,
                    consumer_tiles = array<i64: 0, 4>,
                    element_type = memref<8xi32>,
                    depth = 1 : i64,
                    fuse_mode = "static",
                    fused_dma_channel_group = "group0"}
    conduit.create {name = "mt_b", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 1>,
                    consumer_tiles = array<i64: 0, 5>,
                    element_type = memref<8xi32>,
                    depth = 1 : i64,
                    fuse_mode = "static",
                    fused_dma_channel_group = "group0"}

    // Consumer core for mt_a.
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = conduit.acquire {name = "mt_a", count = 1 : i64, port = #conduit.port<Consume>}
                : !conduit.window<memref<8xi32>>
        %buf = conduit.subview_access %w {index = 0 : i64}
                   : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @process_a(%buf) : (memref<8xi32>) -> ()
        conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    // Consumer core for mt_b.
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = conduit.acquire {name = "mt_b", count = 1 : i64, port = #conduit.port<Consume>}
                : !conduit.window<memref<8xi32>>
        %buf = conduit.subview_access %w {index = 0 : i64}
                   : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @process_b(%buf) : (memref<8xi32>) -> ()
        conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
