// RUN: aie-opt %s | FileCheck %s

module {

// CHECK-LABEL: func.func @core_ops
func.func @core_ops() {
  // CHECK: conduit.create
  // CHECK-SAME: capacity = 8 : i64
  // CHECK-SAME: name = "c1"
  conduit.create {name = "c1", capacity = 8 : i64}
  // CHECK: conduit.put
  // CHECK-SAME: name = "c1"
  // CHECK-SAME: value = 42 : i64
  conduit.put {name = "c1", value = 42 : i64}
  // CHECK: conduit.get
  %v = conduit.get {name = "c1"} : i64
  // CHECK: conduit.prefill
  conduit.prefill {name = "c1", values = array<i64: 0, 1, 2>}
  // CHECK: conduit.advance
  conduit.advance {name = "c1", count = 1 : i64}
  // CHECK: conduit.acquire
  conduit.acquire {name = "c1", count = 2 : i64}
  // CHECK: conduit.release
  conduit.release {name = "c1", count = 2 : i64}
  return
}

// CHECK-LABEL: func.func @fan_ops
func.func @fan_ops() {
  conduit.create {name = "a", capacity = 8 : i64}
  conduit.create {name = "b", capacity = 8 : i64}
  conduit.create {name = "c", capacity = 16 : i64}
  // CHECK: conduit.merge
  conduit.merge {srcs = ["a", "b"], dst = "c"}
  // CHECK: conduit.fork
  conduit.fork {src = "c", dsts = ["a", "b"]}
  // CHECK: conduit.annotate
  conduit.annotate {name = "c", key = "lower_to", value = "objectfifo"}
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
func.func @async_ops() {
  // CHECK: conduit.put_async
  %tok0 = conduit.put_async {name = "ch", value = 1 : i64} : !conduit.async.token
  // CHECK: conduit.get_async
  %tok1 = conduit.get_async {name = "ch"} : !conduit.async.token
  // CHECK: conduit.wait
  conduit.wait %tok0 : !conduit.async.token
  // CHECK: conduit.wait_all
  conduit.wait_all %tok0, %tok1
  // CHECK: conduit.wait_all_async
  %merged = conduit.wait_all_async %tok0, %tok1 :
      (!conduit.async.token, !conduit.async.token) -> !conduit.async.token
  // CHECK: conduit.chain
  conduit.chain [%tok0] -> %tok1 : (!conduit.async.token) -> !conduit.async.token
  return
}

// CHECK-LABEL: func.func @subview_op
func.func @subview_op() {
  conduit.create {name = "buf", capacity = 8 : i64}
  conduit.acquire {name = "buf", count = 2 : i64}
  // CHECK: conduit.subview_access
  %elem = conduit.subview_access {name = "buf", index = 0 : i64} : i64
  conduit.release {name = "buf", count = 2 : i64}
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
