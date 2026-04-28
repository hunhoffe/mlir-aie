// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #42 (was Task #11) — Pass A dma_repeat inference under multi-emission
// per channel (gemv-multi-batch pattern, two Llama decode call sites,
// num_batches=32).
//
// Geometry (downscaled for the test):
//   * nv = num_invocations = 2  (host-side run() loop, INVISIBLE to IR)
//   * N  = num_batches      = 2  (→ 2 aiex.dma_configure_task_for emissions)
//   * K  = per-batch inner walk = 4
//   * Core: outer scf.for(0, nv*N=4) wrapping inner scf.for(0, K=4)
//          → total_core_acquires = 16
//   * Each host BD: len = K = 4 elements, fifo elem = memref<1xbf16>
//          → acquires_per_BD = 4
//   * rt_emissions_per_channel = 2
//
// HISTORY: this fixture originally pinned `dma_repeat = 2 = nv ✓` (the
// three-factor formula `(16 / 2) / 4 = 2`).  That encoded a BUG: Pass A's
// formula is only valid when the host dispatches the rt_seq exactly once,
// but IRON's `num_invocations = N` host-side `run()` loop is INVISIBLE
// in the IR.  The formula under-divides by num_invocations and inflates
// `dma_repeat` by exactly that factor.  Pass C surfaces the inflated value
// onto the shim's `repeat_count`, and firmware reads `repeat_count = N`
// as N+1 fires per call → over-fire of the shim BD.  Llama op7_GEMV +
// op11_GEMV (both num_batches=32, num_invocations=16) hit this and
// over-fired by 16-17×.
//
// FIX (Task #42, 2026-04-28): Pass A's `inferDmaRepeatForChannel` now
// extends its existing `emit.count == 1` shim skip to `emit.count >= 1`
// — i.e. NO inferred `dma_repeat` for any shim-bearing channel
// (regardless of multi-emission count).  Compute-to-compute channels
// (`emit.count == 0`) keep the legacy outer-loop-drives-dma_repeat path.
// This fixture now pins the FIX: no `dma_repeat` attribute should appear
// on the shim-bearing `conduit.create @A`.  See
// `.claude/plans/repeat-count-overfire-rootcause.md`.

// CHECK-LABEL: module @infer_multi_emission_gemv
// CHECK: conduit.create @A
// CHECK-NOT: dma_repeat

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
