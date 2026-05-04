// RUN: aie-opt --conduit-check-loop-balance %s 2>&1 | FileCheck %s
//
// MVE-1: conduit-check-loop-balance pass tests.
//
// dma_repeat is 0-INDEXED ("additional fires beyond the initial one";
// total fires = dma_repeat + 1).  Convention matches IRON's
// `aiex.dma_configure_task_for.repeat_count` (aiex.py:289-291) and
// is documented at `CanonicalizeChannelPutsUtils.h::getDmaRepeatOr0`.
// See Bug #98 / Task #39.
//
// Three cases:
//   1. Violation: dma_repeat=3 (= 4 total DMA sends) but consumer scf.for
//      has trip count 64 — deadlock after 4 iterations.
//      The pass should emit a warning on the conduit.create.
//
//   2. Valid: dma_repeat=63 (= 64 total DMA sends) and scf.for trip
//      count=64.  Sends match loop exactly (boundary case).  No warning.
//
//   3. No dma_repeat — channel skipped entirely. bd_repeat alone is not
//      the total send count and is not checked. No warning.
//
// The check fires when:
//   - conduit.create has dma_repeat=N attribute (= N+1 total fires), AND
//   - a conduit.acquire on the Consume port references that channel, AND
//   - the acquire is inside a statically bounded scf.for with T > N+1.
//
// This catches the Exp C class of deadlock: DMA fires N+1 times
// (dma_repeat=N), consumer loop iterates T > N+1 times — consumer stalls
// after N+1 iterations.

// CHECK:      warning: conduit-check-loop-balance: channel '@short_dma'
// CHECK-SAME: fires 4 total DMA sends
// CHECK-SAME: dma_repeat = 3, 0-indexed
// CHECK-SAME: trip count 64

// CHECK-NOT: warning: conduit-check-loop-balance: channel '@long_dma'
// CHECK-NOT: warning: conduit-check-loop-balance: channel '@no_iter'

module {
  aie.device(npu2) {

    // -----------------------------------------------------------------------
    // CASE 1: Violation — dma_repeat=3 (= 4 total fires) < loop trip
    // count(64).  Expected: warning emitted on @short_dma.
    // -----------------------------------------------------------------------
    conduit.create @short_dma {                    element_type = memref<16xi32>,
                    depth = 1 : i64,
                    dma_repeat = 3 : i64}

    // -----------------------------------------------------------------------
    // CASE 2: Valid — dma_repeat=63 (= 64 total fires) == loop trip
    // count(64).  Boundary case: tripCount == totalFires → no warning.
    // -----------------------------------------------------------------------
    conduit.create @long_dma {                    element_type = memref<16xi32>,
                    depth = 1 : i64,
                    dma_repeat = 63 : i64}

    // -----------------------------------------------------------------------
    // CASE 3: No dma_repeat — channel skipped.
    // bd_repeat alone is not the total send count; not checked.
    // Expected: no warning even though the acquire is inside the loop.
    // -----------------------------------------------------------------------
    conduit.create @no_iter {                    element_type = memref<16xi32>,
                    depth = 1 : i64,
                    bd_repeat = 4 : i64}

    %tile02 = aie.tile(0, 2)

    %core02 = aie.core(%tile02) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index

      // All three acquires inside a trip-count-64 loop.
      scf.for %i = %c0 to %c64 step %c1 {

        // CASE 1 violation: dma_repeat=3 (= 4 fires), loop=64 → warning.
        %w1 = conduit.acquire {name = @short_dma, count = 1 : i64,
                               port = #conduit.port<Consume>}
                : !conduit.window<memref<16xi32>>
        conduit.release %w1 {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<16xi32>>

        // CASE 2 valid: dma_repeat=63 (= 64 fires), loop=64 → no warning.
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
