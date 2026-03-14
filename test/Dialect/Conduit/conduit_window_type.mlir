// RUN: aie-opt %s | FileCheck %s
//
// Roundtrip test for the !conduit.window<T> type and associated ops.
//
// All five gaps from the original XFAIL are now resolved (Task #15):
//   (a) !conduit.window<T> exists as a parametric TypeDef constrained to MemRefType.
//   (b) conduit.acquire returns !conduit.window<T>.
//   (c) conduit.subview_access takes a !conduit.window<T> SSA operand.
//   (d) conduit.release takes a !conduit.window<T> SSA operand.
//   (e) conduit.wait_window exists (16-op dialect; wait_window is the window materialization op).
//
// Syntax notes:
//   - The window type prints without the conduit.window prefix (MLIR mnemonic
//     omission): <memref<N x T>> rather than !conduit.window<memref<N x T>>.
//   - conduit.release: window SSA value first, then attr-dict:
//       conduit.release %win {count = K : i64, port = #conduit.port<Consume>} : <memref<...>>
//   - conduit.subview_access uses {index = i : i64}, not bracket syntax.
//   - conduit.acquire requires port = #conduit.port<Produce>|"Consume".

// CHECK-LABEL: func.func @blocking_acquire_roundtrip
func.func @blocking_acquire_roundtrip() {
  conduit.create {name = "input", capacity = 10 : i64}

  // CHECK: conduit.acquire
  // CHECK-SAME: count = 1
  // CHECK-SAME: name = "input"
  // CHECK-SAME: port = #conduit.port<Consume>
  %win = conduit.acquire {name = "input", count = 1 : i64, port = #conduit.port<Consume>}
             : !conduit.window<memref<10xi32>>

  // CHECK: conduit.subview_access
  // CHECK-SAME: index = 0
  %elem = conduit.subview_access %win {index = 0 : i64}
              : !conduit.window<memref<10xi32>> -> memref<10xi32>

  // CHECK: conduit.release
  // CHECK-SAME: count = 1
  conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<10xi32>>

  return
}

// CHECK-LABEL: func.func @sliding_window_partial_release
func.func @sliding_window_partial_release() {
  conduit.create {name = "weights", capacity = 4 : i64}

  // CHECK: conduit.acquire
  // CHECK-SAME: count = 4
  // CHECK-SAME: name = "weights"
  %win = conduit.acquire {name = "weights", count = 4 : i64, port = #conduit.port<Consume>}
             : !conduit.window<memref<4xi32>>

  // CHECK: conduit.subview_access
  // CHECK-SAME: index = 0
  %e0 = conduit.subview_access %win {index = 0 : i64}
             : !conduit.window<memref<4xi32>> -> memref<4xi32>
  // CHECK: conduit.subview_access
  // CHECK-SAME: index = 1
  %e1 = conduit.subview_access %win {index = 1 : i64}
             : !conduit.window<memref<4xi32>> -> memref<4xi32>

  // Sliding release: release only 2 of 4 acquired slots.
  // CHECK: conduit.release
  // CHECK-SAME: count = 2
  conduit.release %win {count = 2 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<4xi32>>

  return
}

// CHECK-LABEL: func.func @async_acquire_wait_window
// conduit.wait_window: dedicated op for acquire tokens; conduit.wait is void.
func.func @async_acquire_wait_window() {
  conduit.create {name = "async_input", capacity = 16 : i64}

  // CHECK: conduit.acquire_async
  // CHECK-SAME: name = "async_input"
  // CHECK-SAME: !conduit.window.token
  %tok = conduit.acquire_async {name = "async_input", count = 1 : i64}
             : !conduit.window.token

  // CHECK: conduit.wait_window
  // CHECK-SAME: for "async_input"
  // CHECK-SAME: !conduit.window.token
  // CHECK-SAME: <memref<16xi32>>
  %win = conduit.wait_window %tok for "async_input"
             : !conduit.window.token -> !conduit.window<memref<16xi32>>

  // CHECK: conduit.subview_access
  %elem = conduit.subview_access %win {index = 0 : i64}
              : !conduit.window<memref<16xi32>> -> memref<16xi32>

  // CHECK: conduit.release
  conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<16xi32>>

  return
}

// CHECK-LABEL: func.func @async_acquire_with_overlap
// Overlapping DMA and acquire — the canonical double-buffer usage.
func.func @async_acquire_with_overlap() {
  conduit.create {name = "in",  capacity = 9 : i64}
  conduit.create {name = "out", capacity = 1 : i64}

  // CHECK: conduit.put_memref_async
  %dma_tok = conduit.put_memref_async {name = "in", num_elems = 9 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 9>,
                 strides = array<i64: 1>} : !conduit.dma.token

  // CHECK: conduit.acquire_async
  %acq_tok = conduit.acquire_async {name = "out", count = 1 : i64}
                 : !conduit.window.token

  // DMA wait is void — does NOT return a window.
  // CHECK: conduit.wait
  conduit.wait %dma_tok : !conduit.dma.token

  // Acquire wait returns the window handle via the dedicated op.
  // CHECK: conduit.wait_window
  // CHECK-SAME: for "out"
  %win_out = conduit.wait_window %acq_tok for "out"
                 : !conduit.window.token -> !conduit.window<memref<1xi32>>

  // CHECK: conduit.subview_access
  %result = conduit.subview_access %win_out {index = 0 : i64}
                : !conduit.window<memref<1xi32>> -> memref<1xi32>

  // CHECK: conduit.release
  conduit.release %win_out {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<1xi32>>

  return
}
