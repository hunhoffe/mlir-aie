// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// Regression test (A-9): wait_window channel name must match the acquire_async
// token's channel name.

aie.device(npu1) {
conduit.create @foo {slot_elems = 8 : i64,
                element_type = memref<8xi32>,
                depth = 1 : i64}
conduit.create @bar {slot_elems = 8 : i64,
                element_type = memref<8xi32>,
                depth = 1 : i64}
func.func @bad_wait_window_name_mismatch() {
  // acquire_async for @foo produces a token.
  %tok = conduit.acquire_async {name = @foo, count = 1 : i64,
             port = #conduit.port<Consume>}
             : !conduit.window.token

  // wait_window claims the token is for @bar — name mismatch.
  // expected-error@+1 {{'conduit.wait_window' op wait_window channel name 'bar' does not match the channel name 'foo' of the acquire_async token operand}}
  %win = conduit.wait_window %tok for @bar
             : !conduit.window.token -> !conduit.window<memref<8xi32>>

  return
}
}
