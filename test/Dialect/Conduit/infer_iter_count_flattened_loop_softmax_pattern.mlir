// RUN: aie-opt --objectfifo-to-conduit %s -verify-diagnostics 2>&1 | FileCheck %s
//
// Task #11 + #74 — Pass A dma_repeat inference under a flattened single
// scf.for (Softmax pattern from IRON `iron/operators/softmax/design.py`,
// which uses `range_(num_invocations * N_div_n)` instead of nested loops).
//
// Geometry:
//   * Single core, single scf.for(0, N*M) where N = num_invocations = 4 and
//     M = N_div_n = 4 → 16 acquires total.
//   * Host runtime_sequence emits ONE BD on the channel
//     (aiex.dma_configure_task_for @softmax_in — emit.count = 1);
//     BD len = M = 4 elements; fifo elem-type is memref<1xbf16>
//     → acquires_per_BD = 4.
//
// EXPECTED BEHAVIOR (post Task #42, 2026-04-28): for any shim-bearing
// channel (emit.count >= 1) the runtime shape is ambiguous between (A)
// replay-per-dispatch and (C) host-side num_invocations loop with one BD
// fire each.  IRON Softmax lowers to (C); a stamped dma_repeat = 4 would
// over-fire the shim BD and stall the NPU (Bug C).  Pass A SKIPs the
// dma_repeat stamp and emits a remark; runtime defaults to dma_repeat = 1.
// See infer_iter_count_multi_emission_gemv_pattern.mlir for the
// multi-emission analogue (the inference path is now SKIPPED there too —
// see .claude/plans/repeat-count-overfire-rootcause.md).

// CHECK-LABEL: module @infer_flattened_loop_softmax
// CHECK: conduit.create @softmax_in
// CHECK-NOT: dma_repeat
// CHECK-NEXT: aie.core

module @infer_flattened_loop_softmax {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (shim-bearing channel); deferring dma_repeat to runtime}}
    aie.objectfifo @softmax_in(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<1xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      // Single flattened loop: trip count = N * M = 4 * 4 = 16.
      scf.for %i = %c0 to %c16 step %c1 {
        %sub = aie.objectfifo.acquire @softmax_in (Consume, 1)
            : !aie.objectfifosubview<memref<1xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<1xbf16>> -> memref<1xbf16>
        aie.objectfifo.release @softmax_in (Consume, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%a0: memref<4xbf16>) {
      // Single BD, len = M = 4 elements; fifo elem-type holds 1 → per-BD = 4.
      %tA = aiex.dma_configure_task_for @softmax_in {
        aie.dma_bd(%a0 : memref<4xbf16>, 0, 4,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 4, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tA)
      aiex.dma_await_task(%tA)
      aiex.dma_free_task(%tA)
    }
  }
}
