// RUN: aie-opt --conduit-to-dma --verify-diagnostics %s
//
// Regression test: Produce-port sliding window with acquireCount > depth
// must be rejected at Pass C (commit 6c3fdb5).
//
// Error condition: port=Produce, acquire=3, release=1, depth=2 → 3 > 2 → error.
// Safe condition:  port=Produce, acquire=2, release=1, depth=2 → 2 <= 2 → no error.
//
// The error fires on the conduit.release op (that's where Pass C Phase 1
// detects the partial-release pattern and checks acquireCount vs depth).
//
// Topology: compute(0,2) producer → shim(0,0) consumer.

module @conduit_produce_sliding_window_error {
  aie.device(npu1_1col) {
    %prod = aie.tile(0, 2)
    %shim = aie.tile(0, 0)

    // --- Error case: depth=2, acquire=3 (3 > 2 → hard error) ---
    //
    // capacity = 64 = 32 elements * depth(2); perBufLen = 64/2 = 32.
    conduit.create {name = "sliding_out_err", capacity = 64 : i64, depth = 2 : i64,
                    element_type = memref<32xi32>,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64>,
                    shim_consumer_tiles = array<i64: 0, 0>}

    aie.shim_dma_allocation @sliding_out_err_shim_alloc(%shim, S2MM, 0)

    %core_err = aie.core(%prod) {
      %c0 = arith.constant 0 : index
      %val = arith.constant 42 : i32

      %w0 = conduit.acquire {name = "sliding_out_err", count = 3 : i64,
                             port = #conduit.port<Produce>}
                : !conduit.window<memref<32xi32>>
      %buf = conduit.subview_access %w0 {index = 0 : i64}
                 : !conduit.window<memref<32xi32>> -> memref<32xi32>
      memref.store %val, %buf[%c0] : memref<32xi32>
      // expected-error @below {{Produce-port sliding windows (acquire > release on Produce port) are not yet supported in Pass C when maxProdAcquire > depth}}
      conduit.release %w0 {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<32xi32>>

      aie.end
    }

    aie.runtime_sequence(%out: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd (%out[0,0,0,0][1,1,1,64][0,0,0,1])
          {metadata = @sliding_out_err_shim_alloc, id = 0 : i64} : memref<64xi32>
    }
  }
}
