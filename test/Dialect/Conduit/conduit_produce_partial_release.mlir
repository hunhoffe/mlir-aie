// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression test: Produce-port partial-release buffer allocation.
// acquire=3, release=1, depth=2 → nProducerBuffers() = max(2, 3+1) = 4 buffers.
// prod_lock init must equal nProducerBuffers() = 4.
//
// Topology: compute(0,2) producer → shim(0,0) consumer.
// The producer acquires 3 output slots but releases only 1 per step.

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK: %[[PROD:.*]] = aie.tile(0, 2)
// Four buffers allocated (not 2 = depth, not 3 = max acquire):
// CHECK: aie.buffer(%[[PROD]]) {{.*}} : memref<32xi32>
// CHECK: aie.buffer(%[[PROD]]) {{.*}} : memref<32xi32>
// CHECK: aie.buffer(%[[PROD]]) {{.*}} : memref<32xi32>
// CHECK: aie.buffer(%[[PROD]]) {{.*}} : memref<32xi32>
// prod_lock init = 4 (number of producer buffers):
// CHECK: aie.lock(%[[PROD]], {{.*}}) {init = 4 : i32

module @conduit_produce_partial_release {
  aie.device(npu1_1col) {
    %prod = aie.tile(0, 2)
    %shim = aie.tile(0, 0)

    // depth=2, acquire=3 → nProducerBuffers() = max(2, 3+1) = 4 buffers
    conduit.create @sliding_out {slot_elems = 64 : i64, depth = 2 : i64,
                    element_type = memref<32xi32>,
                    shim_consumer_tiles = array<i64: 0, 0>}

    aie.shim_dma_allocation @sliding_out_shim_alloc(%shim, S2MM, 0)

    %core = aie.core(%prod) {
      %c0 = arith.constant 0 : index
      %val = arith.constant 42 : i32

      // Acquire 3 slots (partial-release pattern: depth=2, so 3 > depth)
      %w0 = conduit.acquire {name = @sliding_out, count = 3 : i64,
                             port = #conduit.port<Produce>}
                : !conduit.window<memref<32xi32>>
      %buf = conduit.subview_access %w0 {index = 0 : i64}
                 : !conduit.window<memref<32xi32>> -> memref<32xi32>
      memref.store %val, %buf[%c0] : memref<32xi32>
      // Release only 1: partial release (holds 2 more)
      conduit.release %w0 {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<32xi32>>

      aie.end
    }

    aie.runtime_sequence(%out: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd (%out[0,0,0,0][1,1,1,64][0,0,0,1])
          {metadata = @sliding_out_shim_alloc, id = 0 : i64} : memref<64xi32>
    }
  }
}
