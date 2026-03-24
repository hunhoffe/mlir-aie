// RUN: aie-opt -split-input-file -verify-diagnostics %s | FileCheck %s
//
// C-6 regression test: explicit port on acquire_async; optional SSA window
// operand on release_async for type-checked release.
//
// Tests:
//   (1) acquire_async with explicit port = #conduit.port<Consume> — roundtrip
//   (2) acquire_async with explicit port = #conduit.port<Produce> — roundtrip
//   (3) release_async with optional $window SSA operand — type-checked release
//   (4) release_async without $window (name-only) — still valid
//   (5) Invalid: release_async $window from wrong channel — verifier rejects

// -----

// (1) acquire_async with port=Consume roundtrips correctly.

// CHECK-LABEL: func.func @acquire_async_consume_port
func.func @acquire_async_consume_port() {
  conduit.create @fifo {capacity = 8 : i64,
                  producer_tile = array<i64: 0, 0>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  // CHECK: conduit.acquire_async
  // CHECK-SAME: name = @fifo
  // CHECK-SAME: port = #conduit.port<Consume>
  %tok = conduit.acquire_async {name = @fifo, count = 1 : i64,
             port = #conduit.port<Consume>}
             : !conduit.window.token
  %win = conduit.wait_window %tok for @fifo
             : !conduit.window.token -> !conduit.window<memref<8xi32>>
  conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>
  return
}

// -----

// (2) acquire_async with port=Produce roundtrips correctly.

// CHECK-LABEL: func.func @acquire_async_produce_port
func.func @acquire_async_produce_port() {
  conduit.create @out {capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  // CHECK: conduit.acquire_async
  // CHECK-SAME: port = #conduit.port<Produce>
  %tok = conduit.acquire_async {name = @out, count = 1 : i64,
             port = #conduit.port<Produce>}
             : !conduit.window.token
  %win = conduit.wait_window %tok for @out
             : !conduit.window.token -> !conduit.window<memref<8xi32>>
  conduit.release %win {count = 1 : i64, port = #conduit.port<Produce>}
      : !conduit.window<memref<8xi32>>
  return
}

// -----

// (3) release_async with optional SSA $window operand (type-checked release).
// The verifier confirms the window comes from a matching conduit.acquire.

// CHECK-LABEL: func.func @release_async_with_window_operand
func.func @release_async_with_window_operand() {
  conduit.create @ch {capacity = 8 : i64,
                  producer_tile = array<i64: 0, 0>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  %win = conduit.acquire {name = @ch, count = 1 : i64,
                          port = #conduit.port<Consume>}
             : !conduit.window<memref<8xi32>>
  // CHECK: conduit.release_async
  // CHECK-SAME: name = @ch
  // CHECK-SAME: !conduit.window.token
  %rel_tok = conduit.release_async(%win : !conduit.window<memref<8xi32>>) {
                 name = @ch, count = 1 : i64, port = #conduit.port<Consume>}
                 : !conduit.window.token
  conduit.wait_all %rel_tok : !conduit.window.token
  return
}

// -----

// (4) release_async without $window (name-only path) — valid for producer-side
// standalone async release where no prior acquire exists in this scope.

// CHECK-LABEL: func.func @release_async_name_only
func.func @release_async_name_only() {
  conduit.create @fifo {capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  // CHECK: conduit.release_async
  // CHECK-SAME: name = @fifo
  // CHECK-SAME: port = #conduit.port<Produce>
  %rel_tok = conduit.release_async {name = @fifo, count = 1 : i64,
                 port = #conduit.port<Produce>}
                 : !conduit.window.token
  conduit.wait_all %rel_tok : !conduit.window.token
  return
}

// -----

// (5) Invalid: release_async $window names channel "other" but $name="ch".
// Verifier must reject: window from "other" cannot satisfy "ch"'s lock.

func.func @release_async_window_name_mismatch() {
  conduit.create @ch {capacity = 8 : i64,
                  producer_tile = array<i64: 0, 0>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  conduit.create @other {capacity = 8 : i64,
                  producer_tile = array<i64: 0, 0>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  %win_other = conduit.acquire {name = @other, count = 1 : i64,
                                port = #conduit.port<Consume>}
                   : !conduit.window<memref<8xi32>>
  // expected-error@+1 {{'conduit.release_async' op $window is from channel 'other' but $name is 'ch'}}
  %rel_tok = conduit.release_async(%win_other : !conduit.window<memref<8xi32>>) {
                 name = @ch, count = 1 : i64, port = #conduit.port<Consume>}
                 : !conduit.window.token
  conduit.wait_all %rel_tok : !conduit.window.token
  conduit.release %win_other {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>
  return
}
