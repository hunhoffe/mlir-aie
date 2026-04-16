// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression test: Produce-port sliding window with acquireCount > depth
// is now supported. Pass C allocates max(depth, maxProduceAcquire+1) buffers.
//
// Previously hard-errored: depth=2, acquire=3, release=1 (3 > 2 → error).
// Now correctly allocates max(2, 3+1) = 4 producer buffers.
//
// Topology: compute(0,2) producer → shim(0,0) consumer.

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK: %[[PROD:.*]] = aie.tile(0, 2)
// Four buffers allocated (max(depth=2, acquire+1=4) = 4):
// CHECK-COUNT-4: aie.buffer(%[[PROD]]) {{.*}} : memref<32xi32>
// prod_lock init = 4:
// CHECK: aie.lock(%[[PROD]], {{.*}}) {init = 4 : i32

module @conduit_produce_sliding_window_error {
  aie.device(npu1_1col) {
    %prod = aie.tile(0, 2)
    %shim = aie.tile(0, 0)

    // depth=2, acquire=3 (3 > 2) → nProducerBuffers() = max(2, 3+1) = 4.
    conduit.create @sliding_out_err {depth = 2 : i64,
                    element_type = memref<32xi32>,
                    shim_consumer_tiles = array<i64: 0, 0>}

    aie.shim_dma_allocation @sliding_out_err_shim_alloc(%shim, S2MM, 0)

    %core_err = aie.core(%prod) {
      %c0 = arith.constant 0 : index
      %val = arith.constant 42 : i32

      %w0 = conduit.acquire {name = @sliding_out_err, count = 3 : i64,
                             port = #conduit.port<Produce>}
                : !conduit.window<memref<32xi32>>
      %buf = conduit.subview_access %w0 {index = 0 : i64}
                 : !conduit.window<memref<32xi32>> -> memref<32xi32>
      memref.store %val, %buf[%c0] : memref<32xi32>
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
