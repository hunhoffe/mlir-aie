// RUN: aie-opt --objectfifo-to-conduit %s -verify-diagnostics 2>&1 | FileCheck %s
//
// Task #74 (Bug C falsification) — Pass A dma_repeat inference SKIP path
// for the single-shim-BD pattern.
//
// This test pins the post-#74 behavior on the IR shape that reproduced
// Bug C end-to-end:
//
//   IRON ElementwiseAdd, sz=256, t=64, col=1, num_invocations=4, npu2
//   (lit reduction of /tmp/iter_count_smoke_*/build/...mlir)
//
// Geometry:
//   * Single shim BD def: ONE aiex.dma_configure_task_for per channel
//     in the runtime_sequence (emit.count = 1).
//   * Consumer core: two nested scf.for trips of 4 each → 16 acquires
//     total per channel.
//   * BD len = 256 elements, fifo element type = memref<64xbf16>
//     → naive acquires_per_BD = 256 / 64 = 4.
//
// Pre-#74 Pass A stamped dma_repeat = (16 / 1) / 4 = 4 on each shim
// channel.  IRON's `num_invocations = 4` actually lowers to a host-side
// `run()` loop with 4 separate dispatches (each dispatch fires the BD
// chain ONCE), so the stamped dma_repeat = 4 caused the shim DMA to
// fire 4× per dispatch — 16× total — and starved the cores after the
// first dispatch's worth of work.  This was the Bug C NPU stall.
//
// Post-#74 EXPECTED BEHAVIOR: with emit.count == 1 Pass A CANNOT
// disambiguate between (A) replay-per-dispatch, (B) host-loop with
// per-dispatch replay, and (C) host-loop with single fire per dispatch
// (the IRON path).  Pass A SKIPs the dma_repeat stamp and emits a
// remark on each conduit.create; runtime defaults to dma_repeat = 1
// (one BD fire per shim dispatch) which matches the IRON host-loop
// runtime and restores Bug C correctness.
//
// For the emit.count > 1 case (IRON gemv-style per-batch BD emission
// via Python looping in rt.sequence) Pass A CAN observe the host
// dispatch count directly as emit.count and the three-factor formula
// is sound — see infer_iter_count_multi_emission_gemv_pattern.mlir.

// CHECK-LABEL: module @infer_single_shim_bd_skip
// CHECK: conduit.create @in0
// CHECK-NOT: dma_repeat
// CHECK: conduit.create @in1
// CHECK-NOT: dma_repeat
// CHECK: conduit.create @out0
// CHECK-NOT: dma_repeat
// CHECK-NEXT: aie.core

module @infer_single_shim_bd_skip {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (shim-bearing channel); deferring dma_repeat to runtime}}
    aie.objectfifo @in0(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (shim-bearing channel); deferring dma_repeat to runtime}}
    aie.objectfifo @in1(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (shim-bearing channel); deferring dma_repeat to runtime}}
    aie.objectfifo @out0(%tile_0_2, {%tile_0_0}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Two nested scf.for trips of 4 each → 16 acquires per channel.
    // BD len = 256, fifo elem = 64 → pre-#74 acquires_per_BD = 4 →
    // pre-#74 dma_repeat = (16 / 1) / 4 = 4 (Bug C over-fire).
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %outer = %c0 to %c4 step %c1 {
        scf.for %inner = %c0 to %c4 step %c1 {
          %s0 = aie.objectfifo.acquire @in0 (Consume, 1)
              : !aie.objectfifosubview<memref<64xbf16>>
          %b0 = aie.objectfifo.subview.access %s0[0]
              : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
          aie.objectfifo.release @in0 (Consume, 1)
          %s1 = aie.objectfifo.acquire @in1 (Consume, 1)
              : !aie.objectfifosubview<memref<64xbf16>>
          %b1 = aie.objectfifo.subview.access %s1[0]
              : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
          aie.objectfifo.release @in1 (Consume, 1)
          %so = aie.objectfifo.acquire @out0 (Produce, 1)
              : !aie.objectfifosubview<memref<64xbf16>>
          %bo = aie.objectfifo.subview.access %so[0]
              : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
          aie.objectfifo.release @out0 (Produce, 1)
        }
      }
      aie.end
    }

    // Single shim BD def per channel — emit.count = 1 — IRON's
    // `num_invocations = 4` lowers to a host-side run() loop, NOT to
    // multiple BD defs in the runtime_sequence, so Pass A can't see
    // the dispatch count.
    aie.runtime_sequence(%a0: memref<256xbf16>,
                         %a1: memref<256xbf16>,
                         %a2: memref<256xbf16>) {
      %t0 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%a0 : memref<256xbf16>, 0, 256,
            [<size = 1, stride = 0>, <size = 1, stride = 0>,
             <size = 1, stride = 0>, <size = 256, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%a1 : memref<256xbf16>, 0, 256,
            [<size = 1, stride = 0>, <size = 1, stride = 0>,
             <size = 1, stride = 0>, <size = 256, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @out0 {
        aie.dma_bd(%a2 : memref<256xbf16>, 0, 256,
            [<size = 1, stride = 0>, <size = 1, stride = 0>,
             <size = 1, stride = 0>, <size = 256, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
    }
  }
}
