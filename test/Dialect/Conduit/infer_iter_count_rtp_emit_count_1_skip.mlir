// RUN: aie-opt --objectfifo-to-conduit %s -verify-diagnostics 2>&1 | FileCheck %s
//
// Pattern D (Task #16) × `dc792ebbd5` (single-shim-BD skip) cross-pass
// guard.
//
// Setup:
//   * shim → core S2MM channel @chan with a SINGLE
//     `aiex.dma_configure_task_for` emission in the runtime_sequence
//     (emit.count = 1).
//   * Consumer core upper bound RTP-folds to 8 via Pattern D.
//
// Expected behavior: even though Pattern D successfully resolves the
// trip count, the emit.count == 1 skip in `inferDmaRepeatForChannel`
// (post-`dc792ebbd5`, Bug C falsification) STILL wins.  Pass A cannot
// disambiguate the host-side `num_invocations` from the IR even with a
// known per-dispatch trip, so it defers `dma_repeat` to the runtime
// (default = 1, matching the IRON host-loop path).
//
// This pins the layering: Pattern D's RTP fold is plumbed in BEFORE the
// emit.count check inside the formula — a known trip is necessary but
// not sufficient to stamp dma_repeat on a single-BD shim channel.

// CHECK-LABEL: module @infer_rtp_emit_count_1_skip
// CHECK: conduit.create @chan
// CHECK-NOT: dma_repeat
// CHECK-NEXT: aie.core

module @infer_rtp_emit_count_1_skip {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %my_rtp = aie.buffer(%tile_0_2) {sym_name = "my_rtp", use_write_rtp = true} : memref<2xi32>

    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (shim-bearing channel); deferring dma_repeat to runtime}}
    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<8xbf16>>

    // Consumer core: RTP-folded trip = 8.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %i0 = arith.constant 0 : index
      %v = memref.load %my_rtp[%i0] : memref<2xi32>
      %ub = arith.index_cast %v : i32 to index
      scf.for %k = %c0 to %ub step %c1 {
        %s = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<8xbf16>>
        %b = aie.objectfifo.subview.access %s[0]
            : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
        aie.objectfifo.release @chan (Consume, 1)
      }
      aie.end
    }

    // Single shim BD def (emit.count = 1) + the RTP write.  Pattern D
    // folds the trip but the emit.count == 1 skip wins regardless.
    aie.runtime_sequence(%a0: memref<8xbf16>) {
      aiex.npu.rtp_write(@my_rtp, 0, 8)
      %t0 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%a0 : memref<8xbf16>, 0, 8,
            [<size = 1, stride = 0>, <size = 1, stride = 0>,
             <size = 1, stride = 0>, <size = 8, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_free_task(%t0)
    }
  }
}
