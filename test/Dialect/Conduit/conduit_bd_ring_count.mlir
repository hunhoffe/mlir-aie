// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression test: S2MM BD ring size equals nConsumerBuffers(), not depth
// (commit 6c3fdb5: nConsumerBuffers = max(depth, maxConsumerAcquire+1)).
//
// Case: depth=3, acquire=3, release=1 (sliding window).
//   Old wrong formula: depth + (acq - rel) = 3 + (3-1) = 5 buffers (excess).
//   Correct formula:   max(depth, maxAcquire+1) = max(3, 4) = 4 buffers.
//
// The BD ring in aie.mem must cover all 4 physical buffers, not just depth=3.
// If the ring covered only 3 BDs, the 4th buffer would never receive DMA data,
// causing a hardware deadlock when the consumer tries to acquire it.
//
// Topology: shim(0,0) → compute(0,2), depth=3, element=memref<32xi32>.

// CHECK-LABEL: module @conduit_bd_ring_count

// Exactly 4 buffers allocated (max(3, 4) = 4, not 3 or 5).
// CHECK: aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME: sym_name = "fifo_cons_buff_0"
// CHECK: aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME: sym_name = "fifo_cons_buff_1"
// CHECK: aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME: sym_name = "fifo_cons_buff_2"
// CHECK: aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME: sym_name = "fifo_cons_buff_3"
// CHECK-NOT: sym_name = "fifo_cons_buff_4"

// prod_lock init = 4 (one token per physical buffer slot).
// CHECK: aie.lock(%{{.*}}tile_0_2
// CHECK-SAME: init = 4
// CHECK-SAME: sym_name = "fifo_cons_prod_lock_0"

// S2MM BD ring must contain exactly 4 aie.dma_bd blocks (covers all 4 buffers).
// A ring with only 3 BDs would never DMA into buff_3, causing deadlock.
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

module @conduit_bd_ring_count {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // depth=3, but maxConsumerAcquire=3 → max(3, 3+1) = 4 buffers needed.
    // slot_elems = 96 = 32 elements * depth(3); perBufLen = 96/3 = 32.
    conduit.create @fifo {slot_elems = 96 : i64, depth = 3 : i64,
                    element_type = memref<32xi32>
                    }

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      // Preamble: acquire 2, release 1.
      %win_pre = conduit.acquire {name = @fifo, count = 2 : i64,
                                   port = #conduit.port<Consume>}
                     : !conduit.window<memref<32xi32>>
      conduit.release %win_pre {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<32xi32>>

      // Middle: acquire 3, release 1 (sliding window, maxConsumerAcquire=3).
      scf.for %i = %c0 to %c4 step %c1 {
        %win_mid = conduit.acquire {name = @fifo, count = 3 : i64,
                                     port = #conduit.port<Consume>}
                       : !conduit.window<memref<32xi32>>
        conduit.release %win_mid {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<32xi32>>
      }

      // Tail: acquire 2, release 2 (full release).
      %win_tail = conduit.acquire {name = @fifo, count = 2 : i64,
                                    port = #conduit.port<Consume>}
                      : !conduit.window<memref<32xi32>>
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
