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
// Tests the token synchronization ops using surviving async producers.
func.func @async_ops() {
  conduit.create {name = "ch_a", capacity = 64 : i64}
  conduit.create {name = "ch_b", capacity = 1 : i64}
  // CHECK: conduit.put_memref_async
  %tok0 = conduit.put_memref_async {name = "ch_a", num_elems = 64 : i64,
               offsets = array<i64: 0>, sizes = array<i64: 64>,
               strides = array<i64: 1>} : !conduit.async.token
  // CHECK: conduit.acquire_async
  %tok1 = conduit.acquire_async {name = "ch_b", count = 1 : i64}
               : !conduit.async.token
  // CHECK: conduit.wait
  conduit.wait %tok0 : !conduit.async.token
  // CHECK: conduit.wait_all
  conduit.wait_all %tok0, %tok1
  // CHECK: conduit.wait_all_async
  %merged = conduit.wait_all_async %tok0, %tok1 :
      (!conduit.async.token, !conduit.async.token) -> !conduit.async.token
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
// acquire_async returns a token; conduit.wait_window consumes it and produces
// the window when the buffer is ready.
func.func @acquire_async_op() {
  conduit.create {name = "output", capacity = 1 : i64}
  // Non-blocking window acquisition (Tier 2 bridge)
  // CHECK: conduit.acquire_async
  %acq_tok = conduit.acquire_async {name = "output", count = 1 : i64}
                 : !conduit.async.token
  // CHECK: conduit.wait_all
  conduit.wait_all %acq_tok
  // Option B: conduit.wait_window produces !conduit.window<T>
  // The type prints as <memref<...>> (mnemonic prefix omitted by MLIR printer).
  // CHECK: conduit.wait_window
  // CHECK-SAME: for "output"
  // CHECK-SAME: -> <memref
  %window = conduit.wait_window %acq_tok for "output"
                : !conduit.async.token -> !conduit.window<memref<9xi32>>
  // CHECK: conduit.subview_access
  %out = conduit.subview_access %window {index = 0 : i64}
             : !conduit.window<memref<9xi32>> -> memref<9xi32>
  conduit.release %window {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<9xi32>>
  return
}

// CHECK-LABEL: func.func @release_async_op
func.func @release_async_op() {
  conduit.create {name = "out", capacity = 2 : i64}
  %win = conduit.acquire {name = "out", count = 1 : i64, port = "Consume"}
             : !conduit.window<memref<2xi32>>
  // CHECK: conduit.release_async
  %rel_tok = conduit.release_async {name = "out", count = 1 : i64}
                 : !conduit.async.token
  conduit.wait %rel_tok : !conduit.async.token
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

} // module
