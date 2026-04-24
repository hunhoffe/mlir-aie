// RUN: aie-opt --objectfifo-to-conduit %s -verify-diagnostics 2>&1 | FileCheck %s
//
// Task #11 — Pass A dma_repeat inference, dynamic-bound rejection.
//
// When the enclosing scf.for has a dynamically-computed upper bound,
// inference cannot derive a static dma_repeat.  Behaviour: emit a
// 'dma_repeat inference skipped' remark on the conduit.create and leave
// the conduit.create without a dma_repeat attr (default = 1).

// CHECK-LABEL: module @infer_dynamic_skipped
// CHECK: conduit.create @chan
// CHECK-NOT: dma_repeat
// CHECK: }

module @infer_dynamic_skipped {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: dynamic loop bounds in producer or consumer core}}
    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      // Dynamic upper bound: a runtime mul masks the constant value.
      %dyn = arith.muli %c4, %c4 : index
      scf.for %i = %c0 to %dyn step %c1 {
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
