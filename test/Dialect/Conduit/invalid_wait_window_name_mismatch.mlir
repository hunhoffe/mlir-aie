// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// Regression test (A-9): wait_window channel name must match the acquire_async
// token's channel name.
//
// A mismatch means the caller is presenting a lock-grant token from channel
// "foo" to a wait_window that expects channel "bar". Pass C would then emit
// use_lock on the wrong lock, silently corrupting synchronization.
//
// The verifier must detect and reject this at IR parse / verification time.

func.func @bad_wait_window_name_mismatch() {
  conduit.create @foo {capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  conduit.create @bar {capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}

  // acquire_async for "foo" produces a token.
  %tok = conduit.acquire_async {name = "foo", count = 1 : i64,
             port = #conduit.port<Consume>}
             : !conduit.window.token

  // wait_window claims the token is for "bar" — name mismatch.
  // expected-error@+1 {{'conduit.wait_window' op wait_window channel name 'bar' does not match the channel name 'foo' of the acquire_async token operand}}
  %win = conduit.wait_window %tok for "bar"
             : !conduit.window.token -> !conduit.window<memref<8xi32>>

  return
}
