// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test: --conduit-to-dma must emit aiex.dma_await_task for ALL
// output (S2MM) channels, not just the last one.
//
// Bug: ConduitToDMALower Step 8g uses a scalar `awaitTask` variable that
// gets overwritten when multiple S2MM (get_memref) ops exist. Only the
// last output channel gets dma_await_task; earlier ones are silently dropped,
// creating race conditions on hardware (host reads before DMA completes).
//
// Setup: one input (MM2S) and two outputs (S2MM). After full lowering, the
// runtime_sequence must contain dma_await_task for BOTH output channels.

// CHECK-LABEL: module @await_all_outputs

// The runtime_sequence must contain exactly 3 dma_configure_task_for ops
// (1 input + 2 outputs):
// CHECK:       aie.runtime_sequence
// CHECK-COUNT-3: aiex.dma_configure_task_for

// CRITICAL: Both output channels must have dma_await_task, not just one.
// CHECK-COUNT-2: aiex.dma_await_task

module @await_all_outputs {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // Input: LPDDR5 → compute tile (MM2S).
    aie.objectfifo @in_data(%shim, {%tile}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    // Output A: compute tile → LPDDR5 (S2MM).
    aie.objectfifo @out_a(%tile, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    // Output B: compute tile → LPDDR5 (S2MM).
    aie.objectfifo @out_b(%tile, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @compute_kernel(memref<128xbf16>, memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @in_data(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %oa = aie.objectfifo.acquire @out_a(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %oa_buf = aie.objectfifo.subview.access %oa[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %ob = aie.objectfifo.acquire @out_b(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %ob_buf = aie.objectfifo.subview.access %ob[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @compute_kernel(%in_buf, %oa_buf, %ob_buf)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @out_a(Produce, 1)
        aie.objectfifo.release @out_b(Produce, 1)
        aie.objectfifo.release @in_data(Consume, 1)
      }
      aie.end
    } {link_with = "compute.a"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>, %arg2: memref<128xbf16>) {
      // Input DMA (MM2S)
      %t0 = aiex.dma_configure_task_for @in_data {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      // Output A DMA (S2MM)
      %t1 = aiex.dma_configure_task_for @out_a {
        aie.dma_bd(%arg1 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      // Output B DMA (S2MM)
      %t2 = aiex.dma_configure_task_for @out_b {
        aie.dma_bd(%arg2 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      // Await BOTH outputs
      aiex.dma_await_task(%t1)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
    }
  }
}
