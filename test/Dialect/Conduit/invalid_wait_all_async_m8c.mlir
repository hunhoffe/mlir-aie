// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative test: M8c operand type check on conduit.wait_all_async.
//
// WaitAllAsync::verify() calls checkTokenOperandTypes() on its inputs.
// The TableGen AnyType variadic would normally accept anything — M8c fills
// that gap by explicitly checking that every operand is a conduit token type.
//
// This supplements invalid.mlir which covers the conduit.wait_all path.
// Here we cover the wait_all_async input path.

// -----

// wait_all_async with an i32 non-token input must be rejected.
func.func @wait_all_async_non_token(%bad : i32) {
  conduit.create @ch_wa {slot_elems = 64 : i64}
  %tok = conduit.put_memref_async {name = @ch_wa, num_elems = 64 : i64,
             offsets = array<i64: 0>, sizes = array<i64: 64>,
             strides = array<i64: 1>} : !conduit.dma.token
  // expected-error @+1 {{'conduit.wait_all_async' op operand #1 must be variadic of conduit token type, but got 'i32'}}
  %merged = conduit.wait_all_async %tok, %bad :
      (!conduit.dma.token, i32) -> !conduit.dma.token
  conduit.wait %merged : !conduit.dma.token
  return
}

// -----

// wait_all_async result (dma.token) escapes via call — M10 via wait_all_async.
// WaitAllAsync::verify() calls checkTokenDoesNotEscape() on its result.
func.func private @consumer(%tok : !conduit.dma.token)
func.func @wait_all_async_escape_call() {
  conduit.create @ch_wa2 {slot_elems = 64 : i64}
  %tok = conduit.put_memref_async {name = @ch_wa2, num_elems = 64 : i64,
             offsets = array<i64: 0>, sizes = array<i64: 64>,
             strides = array<i64: 1>} : !conduit.dma.token
  // expected-error @+1 {{'conduit.wait_all_async' op M10: token escapes function scope via call argument}}
  %merged = conduit.wait_all_async %tok :
      (!conduit.dma.token) -> !conduit.dma.token
  func.call @consumer(%merged) : (!conduit.dma.token) -> ()
  return
}
