// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression test for: Pass C allocating only `depth` buffers for channels with
// partial-release (sliding-window) semantics, causing hardware deadlock.
//
// Pattern: conduit.create{depth=2} + acquire{count=3}/release{count=1}
//   - Consumer acquires 3 buffer slots simultaneously (sliding window over 3 rows).
//   - Consumer releases 1 slot per step.
//   - Formula: n_buffers = depth + max(0, maxAcquire - minRelease)
//              n_buffers = 2 + max(0, 3 - 1) = 4
//
// Bug: Conduit was allocating only `depth` (= 2) buffers and setting prod_lock
// init = 2. Deadlock at iteration 2: consumer holds 3 slots, releases 1, DMA
// fills 1, but prod_lock never reaches 3 again — starvation.
//
// Fix: allocate 4 buffers, prod_lock init = 4.
//
// Topology: shim(0,0) → compute(0,2), depth=2, element=memref<32xi32>.

// CHECK-LABEL: module @conduit_partial_release_buffers
//
// Four buffers must be allocated on the consumer tile (not two).
// CHECK: aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME: sym_name = "fifo_cons_buff_0"
// CHECK: aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME: sym_name = "fifo_cons_buff_1"
// CHECK: aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME: sym_name = "fifo_cons_buff_2"
// CHECK: aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME: sym_name = "fifo_cons_buff_3"
//
// prod_lock init must be 4 (not 2).
// CHECK: aie.lock(%{{.*}}tile_0_2
// CHECK-SAME: init = 4
// CHECK-SAME: sym_name = "fifo_cons_prod_lock_0"
//
// cons_lock init must be 0.
// CHECK: aie.lock(%{{.*}}tile_0_2
// CHECK-SAME: init = 0
// CHECK-SAME: sym_name = "fifo_cons_cons_lock_0"
//
// S2MM BD ring must have 4 BD blocks (one per physical buffer).
// CHECK: aie.mem(%{{.*}}tile_0_2)
// CHECK-NEXT: aie.dma_start(S2MM
// CHECK: aie.dma_bd(%fifo_cons_buff_0
// CHECK: aie.next_bd
// CHECK: aie.dma_bd(%fifo_cons_buff_1
// CHECK: aie.next_bd
// CHECK: aie.dma_bd(%fifo_cons_buff_2
// CHECK: aie.next_bd
// CHECK: aie.dma_bd(%fifo_cons_buff_3
// CHECK: aie.next_bd

module @conduit_partial_release_buffers {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // depth=2: user-specified ring depth. With acquire=3/release=1,
    // Pass C must allocate 2 + (3-1) = 4 physical buffers.
    // capacity = 64 = 32 elements * depth(2); perBufLen = 64/2 = 32.
    conduit.create {name = "fifo", capacity = 64 : i64, depth = 2 : i64,
                    element_type = memref<32xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>}

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      // Preamble: acquire 2 rows (fresh start).
      %win_pre = conduit.acquire {name = "fifo", count = 2 : i64,
                                   port = #conduit.port<Consume>}
                     : !conduit.window<memref<32xi32>>
      %pre0 = conduit.subview_access %win_pre {index = 0 : i64}
                  : !conduit.window<memref<32xi32>> -> memref<32xi32>
      %pre1 = conduit.subview_access %win_pre {index = 1 : i64}
                  : !conduit.window<memref<32xi32>> -> memref<32xi32>
      conduit.release %win_pre {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<32xi32>>

      // Middle: acquire 3 rows, release 1 (sliding window).
      scf.for %i = %c0 to %c4 step %c1 {
        %win_mid = conduit.acquire {name = "fifo", count = 3 : i64,
                                     port = #conduit.port<Consume>}
                       : !conduit.window<memref<32xi32>>
        %mid0 = conduit.subview_access %win_mid {index = 0 : i64}
                    : !conduit.window<memref<32xi32>> -> memref<32xi32>
        %mid1 = conduit.subview_access %win_mid {index = 1 : i64}
                    : !conduit.window<memref<32xi32>> -> memref<32xi32>
        %mid2 = conduit.subview_access %win_mid {index = 2 : i64}
                    : !conduit.window<memref<32xi32>> -> memref<32xi32>
        conduit.release %win_mid {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<32xi32>>
      }

      // Tail: acquire 2, release 2.
      %win_tail = conduit.acquire {name = "fifo", count = 2 : i64,
                                    port = #conduit.port<Consume>}
                      : !conduit.window<memref<32xi32>>
      %tail0 = conduit.subview_access %win_tail {index = 0 : i64}
                   : !conduit.window<memref<32xi32>> -> memref<32xi32>
      %tail1 = conduit.subview_access %win_tail {index = 1 : i64}
                   : !conduit.window<memref<32xi32>> -> memref<32xi32>
      conduit.release %win_tail {count = 2 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<32xi32>>

      aie.end
    }

    aie.runtime_sequence(%in: memref<128xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,128][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<128xi32>
    }
  }
}
