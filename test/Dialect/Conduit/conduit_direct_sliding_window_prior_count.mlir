// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// BUG: Hand-authored Conduit IR for a sliding window pattern generates wrong
// AcquireGreaterEqual values when prior_count is omitted on middle and tail
// acquires.
//
// In the preamble/middle/tail sliding window:
//   - Preamble: conduit.acquire{count=2}        → AcquireGreaterEqual(2) ✓
//   - Middle:   conduit.acquire{count=3}        → AcquireGreaterEqual(3) ✗ (bug)
//               correct: conduit.acquire{count=3, prior_count=2}
//                        → AcquireGreaterEqual(1) ✓
//   - Tail:     conduit.acquire{count=2}        → AcquireGreaterEqual(2) ✗ (bug)
//               correct: conduit.acquire{count=2, prior_count=1}
//                        → AcquireGreaterEqual(1) ✓
//
// Without prior_count, Pass C treats each acquire as starting from zero held
// elements, so it emits the full count as the AcquireGreaterEqual value.
// The middle then asks for 3 new rows when only 1 is actually available
// (2 are already in the buffer from the preamble), stalling immediately.
//
// This test XFAIL until either:
//   (a) Pass C infers prior_count from SSA dominance, or
//   (b) The conduit_direct source is updated with explicit prior_count attrs.
//
// XFAIL: *
//
// CHECK-LABEL: module @conduit_direct_sliding_window_prior_count

// Preamble: acquire 2 — correct in both buggy and fixed versions.
// CHECK: aie.core(%tile_0_2)
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 2)

// Middle: must acquire only 1 (2 rows already held from preamble).
// BUG: currently emits AcquireGreaterEqual(3).
// CHECK: scf.for
// CHECK-NEXT: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 1)

// Tail: must acquire only 1 (1 row held from last middle window).
// BUG: currently emits AcquireGreaterEqual(2).
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 1)

// Regression guard: middle must never acquire 3 — that deadlocks.
// CHECK-NOT: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 3)

module @conduit_direct_sliding_window_prior_count {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // Input: shim → tile, depth=4.
    conduit.create {name = "fifo", capacity = 512 : i64, depth = 4 : i64,
                    element_type = memref<128xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>}

    // Output: tile → shim, depth=2.
    conduit.create {name = "out", capacity = 256 : i64, depth = 2 : i64,
                    element_type = memref<64xi32>,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64>,
                    shim_consumer_tiles = array<i64: 0, 0>}

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)
    aie.shim_dma_allocation @out_shim_alloc(%shim, S2MM, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      // Preamble: acquire(2), no prior held elements.
      %win_pre = conduit.acquire {name = "fifo", count = 2 : i64,
                                   port = #conduit.port<Consume>}
                     : !conduit.window<memref<128xi32>>
      %pre0 = conduit.subview_access %win_pre {index = 0 : i64}
                  : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %pre1 = conduit.subview_access %win_pre {index = 1 : i64}
                  : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %win_out_pre = conduit.acquire {name = "out", count = 1 : i64,
                                       port = #conduit.port<Produce>}
                         : !conduit.window<memref<64xi32>>
      %out_pre = conduit.subview_access %win_out_pre {index = 0 : i64}
                     : !conduit.window<memref<64xi32>> -> memref<64xi32>
      conduit.release %win_out_pre {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<64xi32>>
      conduit.release %win_pre {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xi32>>

      // Middle: acquire(3) — BUG: missing prior_count=2.
      // Should be: conduit.acquire {count=3, prior_count=2} → AcquireGreaterEqual(1).
      // Currently:  conduit.acquire {count=3}                → AcquireGreaterEqual(3).
      scf.for %i = %c0 to %c4 step %c1 {
        %win_mid = conduit.acquire {name = "fifo", count = 3 : i64,
                                     port = #conduit.port<Consume>}
                       : !conduit.window<memref<128xi32>>
        %mid0 = conduit.subview_access %win_mid {index = 0 : i64}
                    : !conduit.window<memref<128xi32>> -> memref<128xi32>
        %mid1 = conduit.subview_access %win_mid {index = 1 : i64}
                    : !conduit.window<memref<128xi32>> -> memref<128xi32>
        %mid2 = conduit.subview_access %win_mid {index = 2 : i64}
                    : !conduit.window<memref<128xi32>> -> memref<128xi32>
        %win_out_mid = conduit.acquire {name = "out", count = 1 : i64,
                                         port = #conduit.port<Produce>}
                           : !conduit.window<memref<64xi32>>
        %out_mid = conduit.subview_access %win_out_mid {index = 0 : i64}
                       : !conduit.window<memref<64xi32>> -> memref<64xi32>
        conduit.release %win_mid {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<128xi32>>
        conduit.release %win_out_mid {count = 1 : i64, port = #conduit.port<Produce>}
            : !conduit.window<memref<64xi32>>
      }

      // Tail: acquire(2) — BUG: missing prior_count=1.
      // Should be: conduit.acquire {count=2, prior_count=1} → AcquireGreaterEqual(1).
      // Currently:  conduit.acquire {count=2}                → AcquireGreaterEqual(2).
      %win_tail = conduit.acquire {name = "fifo", count = 2 : i64,
                                    port = #conduit.port<Consume>}
                      : !conduit.window<memref<128xi32>>
      %tail0 = conduit.subview_access %win_tail {index = 0 : i64}
                   : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %tail1 = conduit.subview_access %win_tail {index = 1 : i64}
                   : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %win_out_tail = conduit.acquire {name = "out", count = 1 : i64,
                                        port = #conduit.port<Produce>}
                          : !conduit.window<memref<64xi32>>
      %out_tail = conduit.subview_access %win_out_tail {index = 0 : i64}
                      : !conduit.window<memref<64xi32>> -> memref<64xi32>
      conduit.release %win_out_tail {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<64xi32>>
      conduit.release %win_tail {count = 2 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xi32>>

      aie.end
    }

    aie.runtime_sequence(%in: memref<896xi32>, %out: memref<384xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,896][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<896xi32>
      aiex.npu.dma_memcpy_nd (%out[0,0,0,0][1,1,1,384][0,0,0,1])
          {metadata = @out_shim_alloc, id = 1 : i64} : memref<384xi32>
      aiex.npu.dma_wait {symbol = @out_shim_alloc}
    }
  }
}
