// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #11 — Pass A dma_repeat inference under multi-emission per channel
// (gemv-multi-batch pattern, two Llama decode call sites, num_batches=32).
//
// Geometry (downscaled for the test):
//   * nv = num_invocations = 2
//   * N  = num_batches      = 2  (→ 2 aiex.dma_configure_task_for emissions)
//   * K  = per-batch inner walk = 4
//   * Core: outer scf.for(0, nv*N=4) wrapping inner scf.for(0, K=4)
//          → total_core_acquires = 16
//   * Each host BD: len = K = 4 elements, fifo elem = memref<1xbf16>
//          → acquires_per_BD = 4
//   * rt_emissions_per_channel = 2
//   * Three-factor formula: dma_repeat = (16 / 2) / 4 = 2 = nv ✓
//
// IRON gemv emits one `rt.fill` per batch via Python-side `for batch in
// range(num_batches)`, so the runtime_sequence holds N distinct
// `aiex.dma_configure_task_for` ops on the same channel name.  This pins
// that the inference COUNTS those emissions (not just one) for the divisor.

// CHECK-LABEL: module @infer_multi_emission_gemv
// CHECK: conduit.create @A
// CHECK-SAME: dma_repeat = 2

module @infer_multi_emission_gemv {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @A(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<1xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      // outer trip = nv*N = 4; inner trip = K = 4 → 16 acquires total.
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %sub = aie.objectfifo.acquire @A (Consume, 1)
              : !aie.objectfifosubview<memref<1xbf16>>
          %elem = aie.objectfifo.subview.access %sub[0]
              : !aie.objectfifosubview<memref<1xbf16>> -> memref<1xbf16>
          aie.objectfifo.release @A (Consume, 1)
        }
      }
      aie.end
    }

    aie.runtime_sequence(%batch0: memref<4xbf16>, %batch1: memref<4xbf16>) {
      // Two emissions on the same channel — one per batch.  Both BDs have
      // len = K = 4 elements; per_BD = 4 / 1 = 4.
      %t0 = aiex.dma_configure_task_for @A {
        aie.dma_bd(%batch0 : memref<4xbf16>, 0, 4,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 4, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_free_task(%t0)
      %t1 = aiex.dma_configure_task_for @A {
        aie.dma_bd(%batch1 : memref<4xbf16>, 0, 4,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 4, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t1)
    }
  }
}
