// RUN: aie-opt --conduit-fuse-channels --conduit-to-dma %s | FileCheck %s --check-prefix=FUSED
// RUN: aie-opt --conduit-to-dma %s | FileCheck %s --check-prefix=NOFUSE
//
// Pass C integration test for --conduit-fuse-channels.
//
// Two conduits (chan_a, chan_b) share the same producer tile [0,2].
// Their consumer-side acquire/release ops are strictly non-overlapping:
//   chan_a: [acquire, release]  followed by
//   chan_b: [acquire, release]
//
// Without --conduit-fuse-channels:
//   Pass C assigns each conduit a distinct MM2S channel.
//   chan_a → MM2S 0, chan_b → MM2S 1 (two dma_start ops on tile[0,2]).
//
// With --conduit-fuse-channels:
//   The pass annotates both conduit.create ops with fused_dma_channel_group = "group0".
//   Pass C detects the shared group and assigns both to MM2S 0.
//   Only ONE dma_start(MM2S, 0) is emitted.
//   The BD rings are linked: chan_a's last BD → chan_b's first BD → chan_a's first BD.
//
// This validates:
//   Phase 4.5a: channel reuse for fused groups (fuseGroupMM2SChannel map).
//   Phase 5.5:  isFusedNonFirst suppresses second dma_start for chan_b.
//   Post-loop:  NextBDOp successor rewriting chains the two BD rings.
//
// Hardware topology:
//   tile[0,2] = producer tile (compute, row >= 2)
//   tile[0,4] = consumer of chan_a  (2 rows above producer — non-adjacent)
//   tile[0,5] = consumer of chan_b  (3 rows above producer — non-adjacent)
//   (npu1_1col: column 0, rows 0..5 available; rows 2..5 are compute tiles)

// C7 NOTE: fuse_mode = "runtime" is now rejected by Pass C with emitError + signalPassFailure.
// See conduit_to_dma_fuse_channels_runtime_err.mlir for the error test.
// The control-packet BD reprogramming path (Phase 3) is not yet implemented.

// FUSED-LABEL: module @fuse_channels_test
// FUSED: aie.device(npu1_1col)
// FUSED: aie.tile(0, 2)
// FUSED: aie.tile(0, 4)
// FUSED: aie.tile(0, 5)

// C9: Independent lock pairs — both conduits allocate their own locks on tile_0_2.
// FUSED-DAG: aie.lock(%{{.*}}tile_0_2, {{[0-9]+}}) {init = 1 : i32, sym_name = "chan_a_prod_lock_0"}
// FUSED-DAG: aie.lock(%{{.*}}tile_0_2, {{[0-9]+}}) {init = 0 : i32, sym_name = "chan_a_cons_lock_0"}
// FUSED-DAG: aie.lock(%{{.*}}tile_0_2, {{[0-9]+}}) {init = 1 : i32, sym_name = "chan_b_prod_lock_0"}
// FUSED-DAG: aie.lock(%{{.*}}tile_0_2, {{[0-9]+}}) {init = 0 : i32, sym_name = "chan_b_cons_lock_0"}

// C2: Both flows use DMA channel 0 (shared MM2S) in FUSED mode.
// FUSED: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_4, DMA : 0)
// FUSED: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_5, DMA : 0)

// NOFUSE-LABEL: module @fuse_channels_test
// C11: Unfused baseline — two producer buffers, four producer locks.
// NOFUSE-DAG: aie.buffer(%{{.*}}tile_0_2) {sym_name = "chan_a_buff_0"}
// NOFUSE-DAG: aie.buffer(%{{.*}}tile_0_2) {sym_name = "chan_b_buff_0"}
// NOFUSE-DAG: aie.lock(%{{.*}}tile_0_2, {{.*}}) {init = 1 : i32, sym_name = "chan_a_prod_lock_0"}
// NOFUSE-DAG: aie.lock(%{{.*}}tile_0_2, {{.*}}) {init = 0 : i32, sym_name = "chan_a_cons_lock_0"}
// NOFUSE-DAG: aie.lock(%{{.*}}tile_0_2, {{.*}}) {init = 1 : i32, sym_name = "chan_b_prod_lock_0"}
// NOFUSE-DAG: aie.lock(%{{.*}}tile_0_2, {{.*}}) {init = 0 : i32, sym_name = "chan_b_cons_lock_0"}
// C2: In NOFUSE mode chan_a uses MM2S 0, chan_b uses MM2S 1.
// NOFUSE: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_4, DMA : 0)
// NOFUSE: aie.flow(%{{.*}}tile_0_2, DMA : 1, %{{.*}}tile_0_5, DMA : 0)

// Producer tile DMA: only ONE MM2S channel allocated (channel reuse).
// C1: BD chain linking — dma_start points to chan_a's first BD; chan_a's last BD
//     chains to chan_b's first BD; chan_b's last BD wraps back to chan_a's first BD.
// FUSED:     aie.mem(%{{.*}}tile_0_2)
// FUSED:       aie.dma_start(MM2S, 0, [[FIRST_BD:\^bb[0-9]+]],
// FUSED-NOT:   aie.dma_start(MM2S, 1,
// FUSED:       aie.dma_bd(%chan_a_buff_0
// FUSED:       aie.next_bd [[CHAIN_BD:\^bb[0-9]+]]
// FUSED:       aie.dma_bd(%chan_b_buff_0
// FUSED:       aie.next_bd [[FIRST_BD]]

// Without fusion: two distinct MM2S channels (0 and 1).
// NOFUSE:     aie.mem(%{{.*}}tile_0_2)
// NOFUSE:       aie.dma_start(MM2S, 0,
// NOFUSE:       aie.dma_start(MM2S, 1,

// C10: Consumer S2MM rings are independent with correct buffers.
// FUSED:     aie.mem(%{{.*}}tile_0_4)
// FUSED:       aie.dma_start(S2MM, 0,
// FUSED:       aie.dma_bd(%chan_a_cons_buff_0
// FUSED:     aie.mem(%{{.*}}tile_0_5)
// FUSED:       aie.dma_start(S2MM, 0,
// FUSED:       aie.dma_bd(%chan_b_cons_buff_0

// No residual Conduit ops in either case.
// FUSED-NOT: conduit.create
// FUSED-NOT: conduit.acquire
// FUSED-NOT: conduit.release
// NOFUSE-NOT: conduit.create
// NOFUSE-NOT: conduit.acquire
// NOFUSE-NOT: conduit.release

module @fuse_channels_test {
  aie.device(npu1_1col) {
    func.func @process_a(%buf: memref<8xi32>) -> () {
      return
    }
    func.func @process_b(%buf: memref<8xi32>) -> () {
      return
    }

    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)

    // Two conduits sharing producer tile [0,2].
    // chan_a: producer=[0,2], consumer=[0,4]  (non-adjacent: 2 rows apart)
    // chan_b: producer=[0,2], consumer=[0,5]  (non-adjacent: 3 rows apart)
    // Both use depth=1, 8 i32 elements (capacity=8).
    conduit.create {name = "chan_a", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 4>,
                    element_type = memref<8xi32>,
                    depth = 1 : i64}
    conduit.create {name = "chan_b", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 5>,
                    element_type = memref<8xi32>,
                    depth = 1 : i64}

    // Producer core: produce chan_a then chan_b — strictly sequential.
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        // chan_a window: acquire, use, release.
        %wa = conduit.acquire {name = "chan_a", count = 1 : i64, port = #conduit.port<Produce>}
                 : !conduit.window<memref<8xi32>>
        %buf_a = conduit.subview_access %wa {index = 0 : i64}
                    : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @process_a(%buf_a) : (memref<8xi32>) -> ()
        conduit.release %wa {count = 1 : i64, port = #conduit.port<Produce>}
            : !conduit.window<memref<8xi32>>

        // chan_b window: acquire, use, release.
        %wb = conduit.acquire {name = "chan_b", count = 1 : i64, port = #conduit.port<Produce>}
                 : !conduit.window<memref<8xi32>>
        %buf_b = conduit.subview_access %wb {index = 0 : i64}
                    : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @process_b(%buf_b) : (memref<8xi32>) -> ()
        conduit.release %wb {count = 1 : i64, port = #conduit.port<Produce>}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    // Consumer core for chan_a.
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = conduit.acquire {name = "chan_a", count = 1 : i64, port = #conduit.port<Consume>}
                : !conduit.window<memref<8xi32>>
        %buf = conduit.subview_access %w {index = 0 : i64}
                   : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @process_a(%buf) : (memref<8xi32>) -> ()
        conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    // Consumer core for chan_b.
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = conduit.acquire {name = "chan_b", count = 1 : i64, port = #conduit.port<Consume>}
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
