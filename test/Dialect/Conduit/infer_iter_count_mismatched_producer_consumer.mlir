// RUN: aie-opt --objectfifo-to-conduit %s -verify-diagnostics 2>&1 | FileCheck %s
//
// Task #11 — Pass A dma_repeat inference, mismatched producer/consumer
// trip-count rejection.
//
// A compute-to-compute objectfifo where the producer core's outer loop
// runs 16 iters but the consumer core's outer loop runs 32.  A real
// conduit channel would deadlock under this geometry; the inference
// helper detects the mismatch, emits a remark, and skips the dma_repeat
// annotation rather than guess.

// CHECK-LABEL: module @infer_mismatch_skipped
// CHECK: conduit.create @chan
// CHECK-NOT: dma_repeat
// CHECK: }

module @infer_mismatch_skipped {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: producer trip 16 differs from consumer trip 32}}
    aie.objectfifo @chan(%tile_0_2, {%tile_0_3}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    // Producer core: 16 iterations.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c16 step %c1 {
        %sub = aie.objectfifo.acquire @chan (Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        aie.objectfifo.release @chan (Produce, 1)
      }
      aie.end
    }

    // Consumer core: 32 iterations (mismatch).
    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c32 step %c1 {
        %sub = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        aie.objectfifo.release @chan (Consume, 1)
      }
      aie.end
    }
  }
}
