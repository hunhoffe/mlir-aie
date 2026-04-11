// RUN: aie-opt --conduit-fuse-relay -split-input-file %s | FileCheck %s
//
// Basic test for --conduit-fuse-relay: a gather→scatter pair on the same
// memtile fuses into conduit.transpose.
//
// Topology:
//   @src0, @src1  ──gather──►  @intermediate  ──scatter──►  @dst0, @dst1
//
// After fusion:
//   @src0, @src1  ──transpose──►  @dst0, @dst1
//   @intermediate conduit.create is erased.

aie.device(npu1) {
conduit.create @src0 {slot_elems = 64 : i64, depth = 0 : i64}
conduit.create @src1 {slot_elems = 64 : i64, depth = 0 : i64}
conduit.create @intermediate {slot_elems = 128 : i64, depth = 0 : i64}
conduit.create @dst0 {slot_elems = 64 : i64, depth = 0 : i64}
conduit.create @dst1 {slot_elems = 64 : i64, depth = 0 : i64}
// CHECK-NOT:   conduit.create @intermediate
// CHECK-LABEL: func.func @fuse_gather_scatter_basic
// CHECK:       conduit.transpose
// CHECK-SAME:  srcs = {{[[]}}[@src0, @src1]{{[]]}}
// CHECK-SAME:  dsts = {{[[]}}[@dst0, @dst1]{{[]]}}
// CHECK-SAME:  memtile = "tile(0,1)"
// CHECK-SAME:  offsets = array<i64: 0, 0, 0, 0>
func.func @fuse_gather_scatter_basic() {
  conduit.gather{srcs = [@src0, @src1], dst = @intermediate {memtile = "tile(0,1)"}}
  conduit.scatter{src = @intermediate, dsts = [@dst0, @dst1] {memtile = "tile(0,1)"}}
  return
}
}

// -----

// Negative test: gather and scatter on different memtiles should NOT fuse.

aie.device(npu1) {
conduit.create @a0 {slot_elems = 64 : i64, depth = 0 : i64}
conduit.create @relay {slot_elems = 128 : i64, depth = 0 : i64}
conduit.create @b0 {slot_elems = 64 : i64, depth = 0 : i64}
// CHECK-LABEL: func.func @no_fuse_different_memtile
// CHECK:       conduit.gather
// CHECK:       conduit.scatter
func.func @no_fuse_different_memtile() {
  conduit.gather{srcs = [@a0], dst = @relay {memtile = "tile(0,1)"}}
  conduit.scatter{src = @relay, dsts = [@b0] {memtile = "tile(1,1)"}}
  return
}
}

// -----

// Negative test: intermediate channel has an acquire user → should NOT fuse.

aie.device(npu1) {
conduit.create @x0 {slot_elems = 64 : i64, depth = 0 : i64}
conduit.create @relay_used {slot_elems = 128 : i64, depth = 1 : i64}
conduit.create @y0 {slot_elems = 64 : i64, depth = 0 : i64}
// CHECK-LABEL: func.func @no_fuse_intermediate_has_users
// CHECK:       conduit.gather
// CHECK:       conduit.scatter
func.func @no_fuse_intermediate_has_users() {
  conduit.gather{srcs = [@x0], dst = @relay_used {memtile = "tile(0,1)"}}
  conduit.scatter{src = @relay_used, dsts = [@y0] {memtile = "tile(0,1)"}}
  // This acquire on the intermediate channel should block fusion.
  %w = conduit.acquire {name = @relay_used, count = 1 : i64, port = #conduit.port<Consume>}
       : !conduit.window<memref<128xi32>>
  return
}
}
