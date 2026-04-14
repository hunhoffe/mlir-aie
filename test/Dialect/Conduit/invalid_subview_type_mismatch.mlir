// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative test for conduit.subview_access result type mismatch.

aie.device(npu1) {
conduit.create @typed_fifo {slot_elems = 10 : i64,
                element_type = memref<10xi32>,
                depth = 1 : i64}
func.func @subview_type_mismatch() {
  %win = conduit.acquire {name = @typed_fifo, count = 1 : i64,
                          port = #conduit.port<Consume>}
             : !conduit.window<memref<10xi32>>
  // expected-error @+1 {{'conduit.subview_access' op result type 'memref<10xi16>' does not match window element type 'memref<10xi32>'}}
  %buf = conduit.subview_access %win {index = 0 : i64}
             : !conduit.window<memref<10xi32>> -> memref<10xi16>
  conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<10xi32>>
  return
}
}
