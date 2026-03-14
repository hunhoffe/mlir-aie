// RUN: aie-opt %s | FileCheck %s

module {

// CHECK-LABEL: func.func @window_ops
// Tests the Tier 2 buffer-window workflow with typed conduit.create.
func.func @window_ops() {
  // CHECK: conduit.create
  // CHECK-SAME: capacity = 8 : i64
  // CHECK-SAME: name = "w1"
  conduit.create {name = "w1", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 0>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  // CHECK: conduit.acquire
  // CHECK-SAME: count = 2 : i64
  // CHECK-SAME: name = "w1"
  // CHECK-SAME: port = "Consume"
  %win = conduit.acquire {name = "w1", count = 2 : i64, port = "Consume"}
             : !conduit.window<memref<8xi32>>
  // CHECK: conduit.subview_access
  // CHECK-SAME: index = 0 : i64
  %elem = conduit.subview_access %win {index = 0 : i64}
             : !conduit.window<memref<8xi32>> -> memref<8xi32>
  // CHECK: conduit.release
  // CHECK-SAME: count = 2 : i64
  // CHECK-SAME: port = "Consume"
  conduit.release %win {count = 2 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  return
}

// CHECK-LABEL: func.func @link_op
func.func @link_op() {
  // CHECK: conduit.objectfifo_link
  // CHECK-SAME: memtile = "tile(0,1)"
  // CHECK-SAME: mode = "distribute"
  conduit.objectfifo_link {srcs = ["in"], dsts = ["out0", "out1"],
                           mode = "distribute", memtile = "tile(0,1)",
                           offsets = array<i64: 0, 1024>}
  return
}

// CHECK-LABEL: func.func @memref_ops
func.func @memref_ops() {
  // CHECK: conduit.put_memref
  conduit.put_memref {name = "ch", num_elems = 256 : i64,
                      offsets = array<i64: 0>, sizes = array<i64: 256>,
                      strides = array<i64: 1>}
  // CHECK: conduit.get_memref
  conduit.get_memref {name = "ch", num_elems = 256 : i64,
                      offsets = array<i64: 0>, sizes = array<i64: 256>,
                      strides = array<i64: 1>}
  return
}

// CHECK-LABEL: func.func @async_ops
// Tests the token synchronization ops using the split token types.
// put_memref_async → !conduit.dma.token
// acquire_async    → !conduit.window.token
// wait accepts dma.token; wait_all accepts AnyType (mixed); wait_all_async result is dma.token
func.func @async_ops() {
  conduit.create {name = "ch_a", capacity = 64 : i64}
  conduit.create {name = "ch_b", capacity = 1 : i64}
  // CHECK: conduit.put_memref_async
  // CHECK-SAME: !conduit.dma.token
  %tok0 = conduit.put_memref_async {name = "ch_a", num_elems = 64 : i64,
               offsets = array<i64: 0>, sizes = array<i64: 64>,
               strides = array<i64: 1>} : !conduit.dma.token
  // CHECK: conduit.acquire_async
  // CHECK-SAME: !conduit.window.token
  %tok1 = conduit.acquire_async {name = "ch_b", count = 1 : i64}
               : !conduit.window.token
  // CHECK: conduit.wait
  // CHECK-SAME: !conduit.dma.token
  conduit.wait %tok0 : !conduit.dma.token
  // CHECK: conduit.wait_all
  conduit.wait_all %tok0, %tok1 : !conduit.dma.token, !conduit.window.token
  // CHECK: conduit.wait_all_async
  // The result of wait_all_async is always !conduit.dma.token (merged completion).
  %merged = conduit.wait_all_async %tok0, %tok1 :
      (!conduit.dma.token, !conduit.window.token) -> !conduit.dma.token
  return
}

// CHECK-LABEL: func.func @subview_op
func.func @subview_op() {
  conduit.create {name = "buf", capacity = 8 : i64}
  %win = conduit.acquire {name = "buf", count = 2 : i64, port = "Consume"}
             : !conduit.window<memref<8xi32>>
  // CHECK: conduit.subview_access
  %elem = conduit.subview_access %win {index = 0 : i64}
             : !conduit.window<memref<8xi32>> -> memref<8xi32>
  conduit.release %win {count = 2 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  return
}

// CHECK-LABEL: func.func @acquire_async_op
// Tests the Option B design: conduit.wait_window returns !conduit.window<T>.
// acquire_async returns !conduit.window.token; conduit.wait_window consumes it
// and produces the window when the buffer is ready.
// Cross-tier: mix the window.token with a dma.token in wait_all.
func.func @acquire_async_op() {
  conduit.create {name = "output", capacity = 1 : i64}
  conduit.create {name = "input", capacity = 64 : i64}
  // Non-blocking window acquisition (Tier 2 bridge) — returns !conduit.window.token
  // CHECK: conduit.acquire_async
  // CHECK-SAME: !conduit.window.token
  %acq_tok = conduit.acquire_async {name = "output", count = 1 : i64}
                 : !conduit.window.token
  // Non-blocking DMA send — returns !conduit.dma.token
  // CHECK: conduit.put_memref_async
  // CHECK-SAME: !conduit.dma.token
  %dma_tok = conduit.put_memref_async {name = "input", num_elems = 9 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 9>,
                 strides = array<i64: 1>} : !conduit.dma.token
  // Cross-tier wait: hardware satisfies DMA fill and lock grant in parallel.
  // wait_all accepts AnyType variadic — mixed dma.token + window.token.
  // CHECK: conduit.wait_all
  conduit.wait_all %dma_tok, %acq_tok : !conduit.dma.token, !conduit.window.token
  // wait_window accepts only !conduit.window.token; produces !conduit.window<T>.
  // CHECK: conduit.wait_window
  // CHECK-SAME: for "output"
  // CHECK-SAME: -> <memref
  %window = conduit.wait_window %acq_tok for "output"
                : !conduit.window.token -> !conduit.window<memref<9xi32>>
  // CHECK: conduit.subview_access
  %out = conduit.subview_access %window {index = 0 : i64}
             : !conduit.window<memref<9xi32>> -> memref<9xi32>
  conduit.release %window {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<9xi32>>
  return
}

// CHECK-LABEL: func.func @release_async_op
// release_async returns !conduit.window.token (it is a lock op, not a DMA op).
// conduit.wait accepts !conduit.dma.token only; use wait_all for window tokens.
func.func @release_async_op() {
  conduit.create {name = "out", capacity = 2 : i64}
  %win = conduit.acquire {name = "out", count = 1 : i64, port = "Consume"}
             : !conduit.window<memref<2xi32>>
  // CHECK: conduit.release_async
  // CHECK-SAME: !conduit.window.token
  %rel_tok = conduit.release_async {name = "out", count = 1 : i64}
                 : !conduit.window.token
  // wait_all accepts AnyType variadic — can wait on a window.token here.
  // CHECK: conduit.wait_all
  conduit.wait_all %rel_tok : !conduit.window.token
  return
}

// CHECK-LABEL: func.func @register_ext_bufs
func.func @register_ext_bufs() {
  // CHECK: conduit.register_external_buffers
  // CHECK-SAME: base_addr = 0 : i64
  // CHECK-SAME: name = "shim_chan"
  // CHECK-SAME: num_buffers = 2 : i64
  conduit.register_external_buffers {name = "shim_chan",
                                     num_buffers = 2 : i64,
                                     base_addr = 0 : i64}
  return
}

// CHECK-LABEL: func.func @csdf_create
// Tests that conduit.create with an access_pattern (CSDF) attribute
// roundtrips correctly through the parser and printer.
func.func @csdf_create() {
  // CHECK: conduit.create
  // CHECK-SAME: access_pattern = array<i64: 1, 2, 1>
  // CHECK-SAME: capacity = 4 : i64
  // CHECK-SAME: depth = 4 : i64
  // CHECK-SAME: name = "csdf_ch"
  conduit.create {name = "csdf_ch", capacity = 4 : i64,
                  producer_tile = array<i64: 2, 2>,
                  consumer_tiles = array<i64: 2, 3>,
                  element_type = memref<i32>,
                  depth = 4 : i64,
                  access_pattern = array<i64: 1, 2, 1>}
  return
}

// CHECK-LABEL: func.func @routing_mode_packet
// Tests that conduit.create with routing_mode = "packet" roundtrips correctly.
func.func @routing_mode_packet() {
  // CHECK: conduit.create
  // CHECK-SAME: capacity = 10 : i64
  // CHECK-SAME: name = "pkt_ch"
  // CHECK-SAME: routing_mode = "packet"
  conduit.create {name = "pkt_ch", capacity = 10 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<10xi32>,
                  depth = 1 : i64,
                  routing_mode = "packet"}
  return
}

// CHECK-LABEL: func.func @csdf_balanced_rates
// Tests that conduit.create with explicit producer_rates and consumer_rates
// passes the CSDF balance check: sum(P)*len(C) == sum(C)*len(P).
//
// Case 1: Same period (q=r=2), same sum (3=3).
//   P=[1,2] sum=3 len=2, C=[1,2] sum=3 len=2
//   Check: 3*2 == 3*2 → 6 == 6 ✓
func.func @csdf_balanced_rates() {
  // CHECK: conduit.create
  // CHECK-SAME: consumer_rates = array<i64: 1, 2>
  // CHECK-SAME: name = "csdf_full"
  // CHECK-SAME: producer_rates = array<i64: 1, 2>
  conduit.create {name = "csdf_full", capacity = 6 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<i32>,
                  depth = 6 : i64,
                  producer_rates = array<i64: 1, 2>,
                  consumer_rates = array<i64: 1, 2>}
  return
}

// CHECK-LABEL: func.func @csdf_balanced_different_periods
// Case 2: Different periods (q=1, r=2) but CSDF-balanced.
//   P=[6] sum=6 len=1, C=[2,4] sum=6 len=2
//   Check: 6*2 == 6*1 → 12 == 6? NO — this is NOT balanced.
//   Correct balanced example: P=[3] sum=3 len=1, C=[1,1,1] sum=3 len=3
//   Check: 3*3 == 3*1 → 9 == 3? NO.
//   Actually: P=[2] sum=2 len=1, C=[1,1] sum=2 len=2
//   Check: 2*2 == 2*1 → 4 == 2? NO.
//   The key insight: sum(P)/len(P) == sum(C)/len(C) [equal average rates].
//   P=[4] sum=4 len=1, C=[2,2] sum=4 len=2
//   Check: 4*2 == 4*1 → 8 == 4? NO.
//   P=[2] sum=2 len=2, C=[1] sum=1 len=1 → 2*1==1*2 → 2==2 ✓
//   Producer fires 2 phases of 1 token each; consumer fires 1 phase of 1 token
//   but producer delivers 2 tokens while consumer expects 1 — this deadlocks.
//   Real example: P=[2,2] sum=4 len=2, C=[4] sum=4 len=1 → 4*1==4*2 → 4==8? NO.
//   Correct: P=[2,2] sum=4 len=2, C=[2,2] sum=4 len=2 → 4*2==4*2 ✓ (same period).
//   Or: P=[1,1] sum=2 len=2, C=[2] sum=2 len=1 → 2*1==2*2 → 2==4? NO.
//   In fact sum(P)*len(C)==sum(C)*len(P) means sum(P)/len(P)==sum(C)/len(C) (equal AVERAGE rates).
//   P=[3,1] sum=4 len=2, C=[2] sum=2 len=1 → 4*1==2*2 → 4==4 ✓
func.func @csdf_balanced_different_periods() {
  // P=[3,1] sum=4 len=2, C=[2] sum=2 len=1
  // CSDF check: sum(P)*len(C) = 4*1 = 4 == sum(C)*len(P) = 2*2 = 4 ✓
  // Interpretation: producer fires a 2-phase cycle delivering 3 then 1 token;
  // consumer fires every phase consuming 2 tokens.  Over 2 producer phases / 2
  // consumer phases: producer delivers 4, consumer expects 4 — balanced.
  // CHECK: conduit.create
  // CHECK-SAME: consumer_rates = array<i64: 2>
  // CHECK-SAME: name = "csdf_diff_period"
  // CHECK-SAME: producer_rates = array<i64: 3, 1>
  conduit.create {name = "csdf_diff_period", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<i32>,
                  depth = 4 : i64,
                  producer_rates = array<i64: 3, 1>,
                  consumer_rates = array<i64: 2>}
  return
}

} // module
