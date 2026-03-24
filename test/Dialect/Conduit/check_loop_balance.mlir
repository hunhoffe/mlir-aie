// RUN: aie-opt --conduit-check-loop-balance %s 2>&1 | FileCheck %s
//
// MVE-1: conduit-check-loop-balance pass tests.
//
// Three cases:
//   1. Violation: repeat_count=4 but consumer scf.for has trip count 64.
//      The pass should emit a warning on the conduit.create.
//
//   2. Valid: repeat_count=64 and scf.for trip count=64. No warning.
//
//   3. No repeat_count / iter_count — channel skipped entirely. No warning.
//
// The check fires when:
//   - conduit.create has repeat_count or iter_count attribute, AND
//   - a conduit.acquire on the Consume port references that channel, AND
//   - the acquire is inside a statically bounded scf.for with T > N.
//
// This catches the Exp C class of deadlock: DMA fires N times, consumer
// loop iterates T > N times — consumer stalls after N iterations.

// CHECK:      warning: conduit-check-loop-balance: channel '@short_dma'
// CHECK-SAME: DMA count 4
// CHECK-SAME: trip count 64

// CHECK-NOT: warning: conduit-check-loop-balance: channel '@long_dma'
// CHECK-NOT: warning: conduit-check-loop-balance: channel '@no_repeat'

module {
  aie.device(npu2) {

    // -----------------------------------------------------------------------
    // CASE 1: Violation — repeat_count(4) < loop trip count(64).
    // Expected: warning emitted on @short_dma.
    // -----------------------------------------------------------------------
    conduit.create @short_dma {capacity = 16 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<16xi32>,
                    depth = 1 : i64,
                    repeat_count = 4 : i64}

    // -----------------------------------------------------------------------
    // CASE 2: Valid — repeat_count(64) == loop trip count(64).
    // Expected: no warning.
    // -----------------------------------------------------------------------
    conduit.create @long_dma {capacity = 16 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<16xi32>,
                    depth = 1 : i64,
                    repeat_count = 64 : i64}

    // -----------------------------------------------------------------------
    // CASE 3: No repeat_count / iter_count — channel skipped.
    // Expected: no warning (even though it's inside a loop).
    // -----------------------------------------------------------------------
    conduit.create @no_repeat {capacity = 16 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<16xi32>,
                    depth = 1 : i64}

    %tile02 = aie.tile(0, 2)

    %core02 = aie.core(%tile02) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index

      // CASE 1 + CASE 2 + CASE 3: all acquires inside a trip-count-64 loop.
      scf.for %i = %c0 to %c64 step %c1 {

        // Violation: short_dma has repeat_count=4, loop iterates 64 times.
        %w1 = conduit.acquire {name = @short_dma, count = 1 : i64,
                               port = #conduit.port<Consume>}
                : !conduit.window<memref<16xi32>>
        conduit.release %w1 {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<16xi32>>

        // Valid: long_dma has repeat_count=64, loop iterates 64 times.
        %w2 = conduit.acquire {name = @long_dma, count = 1 : i64,
                               port = #conduit.port<Consume>}
                : !conduit.window<memref<16xi32>>
        conduit.release %w2 {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<16xi32>>

        // No repeat: no_repeat has no repeat_count — skipped.
        %w3 = conduit.acquire {name = @no_repeat, count = 1 : i64,
                               port = #conduit.port<Consume>}
                : !conduit.window<memref<16xi32>>
        conduit.release %w3 {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<16xi32>>
      }

      aie.end
    }
  }
}
