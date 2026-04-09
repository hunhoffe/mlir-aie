// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative test: M10 token escape checks on conduit.release_async.
//
// ReleaseAsync::verify() calls checkTokenDoesNotEscape() on its result token.
// A conduit.window.token from release_async that escapes via return or call
// must be rejected (hardware state is not portable across function boundaries).
//
// This is a distinct M10 path from the acquire_async escape tests in invalid.mlir.

// -----

// release_async token escapes via return.
func.func @release_async_escape_return() -> !conduit.window.token {
  conduit.create @ch_rel {slot_elems = 1 : i64, depth = 0 : i64}
  // expected-error @+1 {{'conduit.release_async' op M10: token escapes function scope via return}}
  %tok = conduit.release_async {name = @ch_rel, count = 1 : i64,
                                 port = #conduit.port<Consume>}
             : !conduit.window.token
  return %tok : !conduit.window.token
}

// -----

// release_async token escapes via call argument.
func.func private @downstream(%tok : !conduit.window.token)
func.func @release_async_escape_call() {
  conduit.create @ch_rel2 {slot_elems = 1 : i64, depth = 0 : i64}
  // expected-error @+1 {{'conduit.release_async' op M10: token escapes function scope via call argument}}
  %tok = conduit.release_async {name = @ch_rel2, count = 1 : i64,
                                  port = #conduit.port<Consume>}
             : !conduit.window.token
  func.call @downstream(%tok) : (!conduit.window.token) -> ()
  return
}
