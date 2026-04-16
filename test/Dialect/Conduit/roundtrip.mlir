// RUN: aie-opt --split-input-file %s | FileCheck %s --check-prefixes=CHECK

module {

aie.device(npu1) {

// CHECK: conduit.create @w1
// CHECK-SAME: element_type = memref<8xi32>
conduit.create @w1 {                element_type = memref<8xi32>,
                depth = 1 : i64}
conduit.create @ch_a {depth = 0 : i64, element_type = memref<64xi32>}
conduit.create @ch_b {depth = 0 : i64, element_type = memref<64xi32>}
conduit.create @buf {depth = 0 : i64, element_type = memref<8xi32>}
conduit.create @output {depth = 0 : i64, element_type = memref<9xi32>}
conduit.create @input {depth = 0 : i64, element_type = memref<9xi32>}
conduit.create @out {depth = 0 : i64, element_type = memref<2xi32>}
// CHECK: conduit.create @pkt_ch
// CHECK-SAME: routing_mode = #conduit.routing_mode<packet>
conduit.create @pkt_ch {                element_type = memref<10xi32>,
                depth = 1 : i64,
                routing_mode = #conduit.routing_mode<packet>}
// CHECK: conduit.create @csdf_full
// CHECK-SAME: consumer_rates = array<i64: 1, 2>
// CHECK-SAME: producer_rates = array<i64: 1, 2>
conduit.create @csdf_full {                element_type = memref<i32>,
                depth = 6 : i64,
                producer_rates = array<i64: 1, 2>,
                consumer_rates = array<i64: 1, 2>}
// CHECK: conduit.create @csdf_diff_period
// CHECK-SAME: consumer_rates = array<i64: 2>
// CHECK-SAME: producer_rates = array<i64: 3, 1>
conduit.create @csdf_diff_period {                element_type = memref<i32>,
                depth = 4 : i64,
                producer_rates = array<i64: 3, 1>,
                consumer_rates = array<i64: 2>}

// CHECK-LABEL: func.func @window_ops
// Tests the Tier 2 buffer-window workflow with typed conduit.create.
func.func @window_ops() {
  // CHECK: conduit.acquire
  // CHECK-SAME: count = 2 : i64
  // CHECK-SAME: name = @w1
  // CHECK-SAME: port = #conduit.port<Consume>
  %win = conduit.acquire {name = @w1, count = 2 : i64, port = #conduit.port<Consume>}
             : !conduit.window<memref<8xi32>>
  // CHECK: conduit.subview_access
  // CHECK-SAME: index = 0 : i64
  %elem = conduit.subview_access %win {index = 0 : i64}
             : !conduit.window<memref<8xi32>> -> memref<8xi32>
  // CHECK: conduit.release
  // CHECK-SAME: count = 2 : i64
  // CHECK-SAME: port = #conduit.port<Consume>
  conduit.release %win {count = 2 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>
  return
}

// CHECK-LABEL: func.func @scatter_op
func.func @scatter_op() {
  // CHECK: conduit.scatter{src = @in, dsts = [@out0, @out1]
  // CHECK-SAME: memtile = "tile(0,1)"
  // CHECK-SAME: offsets = array<i64: 0, 1024>
  conduit.scatter{src = @in, dsts = [@out0, @out1] {memtile = "tile(0,1)",
                   offsets = array<i64: 0, 1024>}}
  return
}

// CHECK-LABEL: func.func @memref_ops
func.func @memref_ops() {
  // CHECK: conduit.put_memref
  conduit.put_memref {name = @ch, num_elems = 256 : i64,
                      offsets = array<i64: 0>, sizes = array<i64: 256>,
                      strides = array<i64: 1>}
  // CHECK: conduit.get_memref
  conduit.get_memref {name = @ch, num_elems = 256 : i64,
                      offsets = array<i64: 0>, sizes = array<i64: 256>,
                      strides = array<i64: 1>}
  return
}

// CHECK-LABEL: func.func @async_ops
// Tests the token synchronization ops using the split token types.
// put_memref_async → !conduit.dma.token
// acquire_async    → !conduit.window.token
// wait_all accepts AnyType variadic (dma or window tokens); wait_all_async result is dma.token
func.func @async_ops() {
  // CHECK: conduit.put_memref_async
  // CHECK-SAME: !conduit.dma.token
  %tok0 = conduit.put_memref_async {name = @ch_a, num_elems = 64 : i64,
               offsets = array<i64: 0>, sizes = array<i64: 64>,
               strides = array<i64: 1>} : !conduit.dma.token
  // CHECK: conduit.acquire_async
  // CHECK-SAME: !conduit.window.token
  %tok1 = conduit.acquire_async {name = @ch_b, count = 1 : i64,
               port = #conduit.port<Consume>}
               : !conduit.window.token
  // CHECK: conduit.wait_all
  // CHECK-SAME: !conduit.dma.token
  conduit.wait_all %tok0 : !conduit.dma.token
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
  %win = conduit.acquire {name = @buf, count = 2 : i64, port = #conduit.port<Consume>}
             : !conduit.window<memref<8xi32>>
  // CHECK: conduit.subview_access
  %elem = conduit.subview_access %win {index = 0 : i64}
             : !conduit.window<memref<8xi32>> -> memref<8xi32>
  conduit.release %win {count = 2 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>
  return
}

// CHECK-LABEL: func.func @acquire_async_op
// Tests the Option B design: conduit.wait_window returns !conduit.window<T>.
// acquire_async returns !conduit.window.token; conduit.wait_window consumes it
// and produces the window when the buffer is ready.
// Cross-tier: mix the window.token with a dma.token in wait_all.
func.func @acquire_async_op() {
  // Non-blocking window acquisition (Tier 2 bridge) — returns !conduit.window.token
  // CHECK: conduit.acquire_async
  // CHECK-SAME: !conduit.window.token
  %acq_tok = conduit.acquire_async {name = @output, count = 1 : i64,
                 port = #conduit.port<Consume>}
                 : !conduit.window.token
  // Non-blocking DMA send — returns !conduit.dma.token
  // CHECK: conduit.put_memref_async
  // CHECK-SAME: !conduit.dma.token
  %dma_tok = conduit.put_memref_async {name = @input, num_elems = 9 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 9>,
                 strides = array<i64: 1>} : !conduit.dma.token
  // Cross-tier wait: hardware satisfies DMA fill and lock grant in parallel.
  // wait_all accepts AnyType variadic — mixed dma.token + window.token.
  // CHECK: conduit.wait_all
  conduit.wait_all %dma_tok, %acq_tok : !conduit.dma.token, !conduit.window.token
  // wait_window accepts only !conduit.window.token; produces !conduit.window<T>.
  // CHECK: conduit.wait_window
  // CHECK-SAME: for @output
  // CHECK-SAME: -> <memref
  %window = conduit.wait_window %acq_tok for @output
                : !conduit.window.token -> !conduit.window<memref<9xi32>>
  // CHECK: conduit.subview_access
  %out = conduit.subview_access %window {index = 0 : i64}
             : !conduit.window<memref<9xi32>> -> memref<9xi32>
  conduit.release %window {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<9xi32>>
  return
}

// CHECK-LABEL: func.func @release_async_op
// release_async returns !conduit.window.token (it is a lock op, not a DMA op).
// conduit.wait_all accepts both !conduit.dma.token and !conduit.window.token.
func.func @release_async_op() {
  %win = conduit.acquire {name = @out, count = 1 : i64, port = #conduit.port<Consume>}
             : !conduit.window<memref<2xi32>>
  // CHECK: conduit.release_async
  // CHECK-SAME: !conduit.window.token
  %rel_tok = conduit.release_async(%win : !conduit.window<memref<2xi32>>) {name = @out, count = 1 : i64, port = #conduit.port<Consume>}
                 : !conduit.window.token
  // wait_all accepts AnyType variadic — can wait on a window.token here.
  // CHECK: conduit.wait_all
  conduit.wait_all %rel_tok : !conduit.window.token
  return
}

func.func @routing_mode_packet() {
  return
}

func.func @csdf_balanced_rates() {
  return
}

func.func @csdf_balanced_different_periods() {
  return
}

} // aie.device(npu1)

} // module

// -----

// Tests that conduit.create with routing_mode=cascade roundtrips correctly.
// After cascade migration (#27), core-body cascade ops are aie.put_cascade /
// aie.get_cascade directly.
// Must wrap in aie.device(npu2) so the AIE verifier knows cascade width=512
// (vector<16xi32> = 512 bits, valid for AIE2 / npu2).
// CHECK-LABEL: aie.device
aie.device(npu2) {
  %t03 = aie.tile(0, 3)
  %t13 = aie.tile(1, 3)
  // CHECK: conduit.create @cas
  conduit.create @cas {                  depth = 1 : i64,
                  element_type = memref<16xi32>,
                  routing_mode = #conduit.routing_mode<cascade>}
  aie.core(%t03) {
    %v = arith.constant dense<42> : vector<16xi32>
    // CHECK: aie.put_cascade
    aie.put_cascade(%v : vector<16xi32>)
    aie.end
  }
  aie.core(%t13) {
    // CHECK: aie.get_cascade
    %r = aie.get_cascade() : vector<16xi32>
    aie.end
  }
}
