// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression test: depth=4, acquire=3 → max(4, 3+1) = 4 buffers (NOT 6).
//
// The previous wrong formula was: depth + max(0, maxAcquire - minRelease)
//   depth=4, acquire=3, release=1 → 4 + (3-1) = 6 (over-allocates by 2)
// The correct formula is: max(depth, maxAcquire + 1)
//   max(4, 3+1) = max(4, 4) = 4 (depth is already sufficient)
//
// Over-allocation wastes 2×element_size of tile SRAM (2×32×4 = 256 bytes here)
// and can exceed the 32KB tile memory budget for large element types.
//
// Topology: shim(0,0) → compute(0,2), depth=4, acquire=3, element=memref<32xi32>.

// CHECK-LABEL: module @conduit_partial_release_depth4
//
// Exactly four buffers (depth is already sufficient; no over-allocation).
// CHECK: aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME: sym_name = "fifo_cons_buff_0"
// CHECK: aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME: sym_name = "fifo_cons_buff_1"
// CHECK: aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME: sym_name = "fifo_cons_buff_2"
// CHECK: aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME: sym_name = "fifo_cons_buff_3"
// CHECK-NOT: sym_name = "fifo_cons_buff_4"
// CHECK-NOT: sym_name = "fifo_cons_buff_5"
//
// prod_lock init must be 4 (not 6).
// CHECK: aie.lock(%{{.*}}tile_0_2
// CHECK-SAME: init = 4
// CHECK-SAME: sym_name = "fifo_cons_prod_lock_0"
//
// cons_lock init must be 0.
// CHECK: aie.lock(%{{.*}}tile_0_2
// CHECK-SAME: init = 0
// CHECK-SAME: sym_name = "fifo_cons_cons_lock_0"
//
// S2MM BD ring must have exactly 4 BD blocks (not 6).
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

module @conduit_partial_release_depth4 {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // depth=4: user-specified ring depth. With acquire=3/release=1,
    // max(depth, maxAcquire+1) = max(4, 4) = 4 (no extra buffers needed).
    // slot_elems = 128 = 32 elements * depth(4); perBufLen = 128/4 = 32.
    conduit.create @fifo {slot_elems = 128 : i64, depth = 4 : i64,
                    element_type = memref<32xi32>
                    }

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      // Preamble: acquire 2.
      %win_pre = conduit.acquire {name = @fifo, count = 2 : i64,
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
        %win_mid = conduit.acquire {name = @fifo, count = 3 : i64,
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

      // Tail: acquire 2, release 2 (full release).
      %win_tail = conduit.acquire {name = @fifo, count = 2 : i64,
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

    aie.runtime_sequence(%in: memref<256xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,256][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<256xi32>
    }
  }
}
