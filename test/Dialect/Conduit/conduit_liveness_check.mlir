// RUN: aie-opt --conduit-check-liveness %s -split-input-file -verify-diagnostics
//
// M11 window liveness check tests.
//
// The pass errors when a conduit.acquire or conduit.wait_window result has no
// conduit.release or conduit.release_async in the enclosing function.
//
// Three correctness tiers verified here:
//   (a) Passing cases  — no error emitted
//   (b) Failing cases  — expected-error annotations consumed by -verify-diagnostics
//
// Theory reference:
//   A !conduit.window<T> is a hardware lock grant.  The lock counter on the
//   producing tile is decremented when the consumer acquires and must be
//   incremented by a matching release.  Without the release, the producer tile
//   spins forever waiting for the counter to recover — hardware deadlock.
//
//   M11 is the "zero-release" dual of M8a ("over-release"):
//     M8a: totalReleased > acquiredCount → double-release error
//     M11: totalReleased == 0            → lock-leak error
//   Together they bound the release count to [1, acquiredCount].
//
//===----------------------------------------------------------------------===//
// PASSING: acquire + matching sync release (same block)
//===----------------------------------------------------------------------===//

aie.device(npu1) {
conduit.create @c {slot_elems = 8 : i64,
                producer_tile = array<i64: 0, 2>,
                consumer_tiles = array<i64: 0, 3>,
                element_type = memref<8xi32>,
                depth = 1 : i64}
func.func @ok_sync_release() {
  %win = conduit.acquire {name = @c, count = 1 : i64, port = #conduit.port<Consume>}
             : !conduit.window<memref<8xi32>>
  conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>
  return
}
} // aie.device

// -----

//===----------------------------------------------------------------------===//
// PASSING: async acquire path (acquire_async + wait_window + release)
//===----------------------------------------------------------------------===//

aie.device(npu1) {
conduit.create @c {slot_elems = 8 : i64,
                producer_tile = array<i64: 0, 2>,
                consumer_tiles = array<i64: 0, 3>,
                element_type = memref<8xi32>,
                depth = 1 : i64}
func.func @ok_async_acquire() {
  %tok = conduit.acquire_async {name = @c, count = 1 : i64,
             port = #conduit.port<Consume>}
             : !conduit.window.token
  %win = conduit.wait_window %tok for @c
             : !conduit.window.token -> !conduit.window<memref<8xi32>>
  conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>
  return
}
} // aie.device

// -----

//===----------------------------------------------------------------------===//
// PASSING: release_async by name (async release path; no SSA window operand)
//===----------------------------------------------------------------------===//

aie.device(npu1) {
conduit.create @c {slot_elems = 8 : i64,
                producer_tile = array<i64: 0, 2>,
                consumer_tiles = array<i64: 0, 3>,
                element_type = memref<8xi32>,
                depth = 1 : i64}
func.func @ok_release_async() {
  %win = conduit.acquire {name = @c, count = 1 : i64, port = #conduit.port<Consume>}
             : !conduit.window<memref<8xi32>>
  // release_async references the channel by name, not by SSA window value.
  // M11 accepts this as a valid release path.
  conduit.release_async {name = @c, count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window.token
  return
}
} // aie.device

// -----

//===----------------------------------------------------------------------===//
// PASSING: sliding window — partial release satisfies M11
//   acquire(3) → release(1) → release(2)  — totalReleased == 3 == acquiredCount
//   M11 fires on ZERO releases; one release is sufficient regardless of count.
//===----------------------------------------------------------------------===//

aie.device(npu1) {
conduit.create @c {slot_elems = 24 : i64,
                producer_tile = array<i64: 0, 2>,
                consumer_tiles = array<i64: 0, 3>,
                element_type = memref<24xi32>,
                depth = 3 : i64}
func.func @ok_partial_release() {
  %win = conduit.acquire {name = @c, count = 3 : i64, port = #conduit.port<Consume>}
             : !conduit.window<memref<24xi32>>
  conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<24xi32>>
  return
}
} // aie.device

// -----

//===----------------------------------------------------------------------===//
// FAILING: acquire with no release anywhere in the function
//===----------------------------------------------------------------------===//

aie.device(npu1) {
conduit.create @c {slot_elems = 8 : i64,
                producer_tile = array<i64: 0, 2>,
                consumer_tiles = array<i64: 0, 3>,
                element_type = memref<8xi32>,
                depth = 1 : i64}
// expected-note@+1 {{in function '@fail_no_release'}}
func.func @fail_no_release() {
  // expected-error@+1 {{M11: window lock grant from conduit.acquire on channel 'c' is never released}}
  %win = conduit.acquire {name = @c, count = 1 : i64, port = #conduit.port<Consume>}
             : !conduit.window<memref<8xi32>>
  return
}
} // aie.device

// -----

//===----------------------------------------------------------------------===//
// FAILING: async acquire (wait_window) with no release
//===----------------------------------------------------------------------===//

aie.device(npu1) {
conduit.create @c {slot_elems = 8 : i64,
                producer_tile = array<i64: 0, 2>,
                consumer_tiles = array<i64: 0, 3>,
                element_type = memref<8xi32>,
                depth = 1 : i64}
// expected-note@+1 {{in function '@fail_wait_window_no_release'}}
func.func @fail_wait_window_no_release() {
  %tok = conduit.acquire_async {name = @c, count = 1 : i64,
             port = #conduit.port<Consume>}
             : !conduit.window.token
  // expected-error@+1 {{M11: window lock grant from conduit.wait_window on channel 'c' is never released}}
  %win = conduit.wait_window %tok for @c
             : !conduit.window.token -> !conduit.window<memref<8xi32>>
  return
}
} // aie.device

// -----

//===----------------------------------------------------------------------===//
// FAILING: two acquires, only one released — second window leaks
//===----------------------------------------------------------------------===//

aie.device(npu1) {
conduit.create @c {slot_elems = 8 : i64,
                producer_tile = array<i64: 0, 2>,
                consumer_tiles = array<i64: 0, 3>,
                element_type = memref<8xi32>,
                depth = 2 : i64}
// expected-note@+1 {{in function '@fail_second_acquire_leaked'}}
func.func @fail_second_acquire_leaked() {
  // First acquire: correctly released.
  %w1 = conduit.acquire {name = @c, count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<8xi32>>
  conduit.release %w1 {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>
  // Second acquire: NOT released — leaked lock grant.
  // expected-error@+1 {{M11: window lock grant from conduit.acquire on channel 'c' is never released}}
  %w2 = conduit.acquire {name = @c, count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<8xi32>>
  return
}
} // aie.device

// -----

//===----------------------------------------------------------------------===//
// PASSING: release_async on a DIFFERENT channel does not suppress M11 on "c"
// (name scoping is per-channel, not global)
//===----------------------------------------------------------------------===//

aie.device(npu1) {
conduit.create @c {slot_elems = 8 : i64,
                producer_tile = array<i64: 0, 2>,
                consumer_tiles = array<i64: 0, 3>,
                element_type = memref<8xi32>,
                depth = 1 : i64}
// expected-note@+1 {{in function '@fail_wrong_channel_release_async'}}
func.func @fail_wrong_channel_release_async() {
  // expected-error@+1 {{M11: window lock grant from conduit.acquire on channel 'c' is never released}}
  %win = conduit.acquire {name = @c, count = 1 : i64, port = #conduit.port<Consume>}
             : !conduit.window<memref<8xi32>>
  // release_async names channel "other", not "c" — M11 on "c" still fires.
  conduit.release_async {name = @other, count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window.token
  return
}
} // aie.device
