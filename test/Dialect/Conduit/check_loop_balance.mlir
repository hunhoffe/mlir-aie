// RUN: aie-opt --conduit-check-loop-balance %s 2>&1 | FileCheck %s
//
// MVE-1: conduit-check-loop-balance pass tests.
//
// Three cases:
//   1. Violation: dma_repeat=4 (total DMA sends) but consumer scf.for has
//      trip count 64 — deadlock after 4 iterations.
//      The pass should emit a warning on the conduit.create.
//
//   2. Valid: dma_repeat=64 and scf.for trip count=64. Sends match loop.
//      No warning.
//
//   3. No dma_repeat — channel skipped entirely. bd_repeat alone is not
//      the total send count and is not checked. No warning.
//
// The check fires when:
//   - conduit.create has dma_repeat=N attribute, AND
//   - a conduit.acquire on the Consume port references that channel, AND
//   - the acquire is inside a statically bounded scf.for with T > N.
//
// This catches the Exp C class of deadlock: DMA fires N times (dma_repeat=N),
// consumer loop iterates T > N times — consumer stalls after N iterations.

// CHECK:      warning: conduit-check-loop-balance: channel '@short_dma'
// CHECK-SAME: dma_repeat 4
// CHECK-SAME: trip count 64

// CHECK-NOT: warning: conduit-check-loop-balance: channel '@long_dma'
// CHECK-NOT: warning: conduit-check-loop-balance: channel '@no_iter'

module {
  aie.device(npu2) {

    // -----------------------------------------------------------------------
    // CASE 1: Violation — dma_repeat(4) < loop trip count(64).
    // Expected: warning emitted on @short_dma.
    // -----------------------------------------------------------------------
    conduit.create @short_dma {capacity = 16 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<16xi32>,
                    depth = 1 : i64,
                    dma_repeat = 4 : i64}

    // -----------------------------------------------------------------------
    // CASE 2: Valid — dma_repeat(64) == loop trip count(64). No warning.
    // -----------------------------------------------------------------------
    conduit.create @long_dma {capacity = 16 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<16xi32>,
                    depth = 1 : i64,
                    dma_repeat = 64 : i64}

    // -----------------------------------------------------------------------
    // CASE 3: No dma_repeat — channel skipped.
    // bd_repeat alone is not the total send count; not checked.
    // Expected: no warning even though the acquire is inside the loop.
    // -----------------------------------------------------------------------
    conduit.create @no_iter {capacity = 16 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<16xi32>,
                    depth = 1 : i64,
                    bd_repeat = 4 : i64}

    %tile02 = aie.tile(0, 2)

    %core02 = aie.core(%tile02) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index

      // All three acquires inside a trip-count-64 loop.
      scf.for %i = %c0 to %c64 step %c1 {

        // CASE 1 violation: dma_repeat=4, loop=64 → warning.
        %w1 = conduit.acquire {name = @short_dma, count = 1 : i64,
                               port = #conduit.port<Consume>}
                : !conduit.window<memref<16xi32>>
        conduit.release %w1 {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<16xi32>>

        // CASE 2 valid: dma_repeat=64, loop=64 → no warning.
        %w2 = conduit.acquire {name = @long_dma, count = 1 : i64,
                               port = #conduit.port<Consume>}
                : !conduit.window<memref<16xi32>>
        conduit.release %w2 {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<16xi32>>

        // CASE 3 no dma_repeat: channel skipped → no warning.
        %w3 = conduit.acquire {name = @no_iter, count = 1 : i64,
                               port = #conduit.port<Consume>}
                : !conduit.window<memref<16xi32>>
        conduit.release %w3 {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<16xi32>>
      }

      aie.end
    }
  }
}
