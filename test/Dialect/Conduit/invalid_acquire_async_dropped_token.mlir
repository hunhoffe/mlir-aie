// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative test: M8-drop — acquire_async with a dropped (zero-use) token.
//
// AcquireAsync::verify() checks use_empty() on the produced window.token.
// A dropped token means the lock is permanently acquired and can never be
// released, causing hardware deadlock.  This is a hard error.

// -----

// acquire_async token with no uses must be rejected (M8-drop).
func.func @acquire_async_dropped_token() {
  conduit.create {name = "ch", capacity = 128 : i64}
  // expected-error @+1 {{'conduit.acquire_async' op (M8-drop) window.token has no uses}}
  %tok = conduit.acquire_async {name = "ch", count = 1 : i64}
             : !conduit.window.token
  // %tok is never used — lock permanently acquired, hardware deadlock.
  return
}
