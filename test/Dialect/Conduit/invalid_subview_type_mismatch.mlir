// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative test for conduit.subview_access result type mismatch.
//
// ConduitOps.cpp SubviewAccess::verify() checks:
//   if (getResult().getType() != winTy.getElementType())
//     return emitOpError("result type ... does not match window element type ...")
//
// This is a distinct error from the M2 index-out-of-bounds check (also in
// SubviewAccess::verify()). The index check is covered by invalid.mlir; this
// test covers the type mismatch path — accessing a window<memref<10xi32>>
// but declaring the result as memref<10xi16>.

func.func @subview_type_mismatch() {
  conduit.create {name = "typed_fifo", capacity = 10 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<10xi32>,
                  depth = 1 : i64}
  %win = conduit.acquire {name = "typed_fifo", count = 1 : i64,
                          port = #conduit.port<Consume>}
             : !conduit.window<memref<10xi32>>
  // expected-error @+1 {{'conduit.subview_access' op result type 'memref<10xi16>' does not match window element type 'memref<10xi32>'}}
  %buf = conduit.subview_access %win {index = 0 : i64}
             : !conduit.window<memref<10xi32>> -> memref<10xi16>
  conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<10xi32>>
  return
}
