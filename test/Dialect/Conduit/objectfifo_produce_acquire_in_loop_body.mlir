// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s --check-prefix=CONDUIT
// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s --check-prefix=DMA
//
// Regression test: Produce acquire inside a loop body must NOT be subsumed
// by a released preamble acquire in a dominating scope.
//
// Pattern (producer core — @outRows):
//   acquire @outRows(Produce, 1)   // preamble
//   call conv(...)
//   release @outRows(Produce, 1)   // preamble release — window now closed
//   scf.for {
//     acquire @outRows(Produce, 1) // inner loop: NEW acquire, NOT subsumed
//     call conv(...)
//     release @outRows(Produce, 1)
//   }
//
// Bug: findWindowInDominatingBlock found the preamble's (released) window in
// the dominating scope and reused it for the loop body's acquire.  Pass C
// then generated no AcquireGreaterEqual inside the loop, causing the core to
// write output without owning the lock → hardware deadlock.
//
// Fix: released windows are tracked; findWindowInDominatingBlock skips them.
//
// Expected (CONDUIT): two separate conduit.acquire ops for @outRows —
//   one before the scf.for (preamble) and one inside the scf.for body (loop).
// Expected (DMA): two AcquireGreaterEqual use_lock ops for outRows_prod_lock —
//   one in the preamble and one inside the scf.for body.

// CONDUIT-LABEL: module @produce_acquire_loop_regression
// CONDUIT:       aie.core
// CONDUIT:         conduit.acquire {{{.*}}name = @outRows{{.*}}port = #conduit.port<Produce>
// CONDUIT:         conduit.release
// CONDUIT:         scf.for
// CONDUIT:           conduit.acquire {{{.*}}name = @outRows{{.*}}port = #conduit.port<Produce>
// CONDUIT:           conduit.release

// DMA-LABEL: module @produce_acquire_loop_regression
// DMA:       aie.core
// DMA:         aie.use_lock({{.*}}, AcquireGreaterEqual
// DMA:         aie.use_lock({{.*}}, Release
// DMA:         scf.for
// DMA:           aie.use_lock({{.*}}, AcquireGreaterEqual
// DMA:           aie.use_lock({{.*}}, Release

module @produce_acquire_loop_regression {
  aie.device(npu1_1col) {

    func.func private @dummy_compute(%in: memref<32xi32>,
                                     %out: memref<32xi32>) -> ()

    %shim00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)

    // @inRows: shim → compute tile (depth=2)
    aie.objectfifo @inRows(%shim00, {%tile02}, 2 : i32)
        : !aie.objectfifo<memref<32xi32>>

    // @outRows: compute tile → shim (depth=2)
    aie.objectfifo @outRows(%tile02, {%shim00}, 2 : i32)
        : !aie.objectfifo<memref<32xi32>>

    // Consumer core: preamble produce(1)/release(1); loop: produce(1)/release(1) x4
    %core02 = aie.core(%tile02) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c4   = arith.constant 4 : index

      // Preamble: acquire one input row and one output row, compute, release both.
      %sv_in_pre = aie.objectfifo.acquire @inRows(Consume, 1)
                       : !aie.objectfifosubview<memref<32xi32>>
      %in_pre = aie.objectfifo.subview.access %sv_in_pre[0]
                    : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>

      %sv_out_pre = aie.objectfifo.acquire @outRows(Produce, 1)
                        : !aie.objectfifosubview<memref<32xi32>>
      %out_pre = aie.objectfifo.subview.access %sv_out_pre[0]
                     : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>

      func.call @dummy_compute(%in_pre, %out_pre) : (memref<32xi32>, memref<32xi32>) -> ()

      // Release both — preamble window closed.
      aie.objectfifo.release @outRows(Produce, 1)
      aie.objectfifo.release @inRows(Consume, 1)

      // Inner loop: each iteration acquires a fresh @outRows(Produce, 1).
      // This acquire must NOT be subsumed by the (released) preamble acquire.
      scf.for %i = %c0 to %c4 step %c1 {
        %sv_in = aie.objectfifo.acquire @inRows(Consume, 1)
                     : !aie.objectfifosubview<memref<32xi32>>
        %in_buf = aie.objectfifo.subview.access %sv_in[0]
                      : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>

        %sv_out = aie.objectfifo.acquire @outRows(Produce, 1)
                      : !aie.objectfifosubview<memref<32xi32>>
        %out_buf = aie.objectfifo.subview.access %sv_out[0]
                       : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>

        func.call @dummy_compute(%in_buf, %out_buf)
            : (memref<32xi32>, memref<32xi32>) -> ()

        aie.objectfifo.release @inRows(Consume, 1)
        aie.objectfifo.release @outRows(Produce, 1)
      }

      aie.end
    }

    aie.runtime_sequence(%arg0: memref<160xi32>, %arg1: memref<160xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 160][0, 0, 0, 1])
          {id = 0 : i64, metadata = @inRows} : memref<160xi32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 160][0, 0, 0, 1])
          {id = 1 : i64, metadata = @outRows} : memref<160xi32>
      aiex.npu.dma_wait {symbol = @outRows}
    }
  }
}
