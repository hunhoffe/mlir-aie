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
conduit.create @src0 {element_type = memref<64xi32>, depth = 0 : i64}
conduit.create @src1 {element_type = memref<64xi32>, depth = 0 : i64}
conduit.create @intermediate {element_type = memref<128xi32>, depth = 0 : i64}
conduit.create @dst0 {element_type = memref<64xi32>, depth = 0 : i64}
conduit.create @dst1 {element_type = memref<64xi32>, depth = 0 : i64}
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
conduit.create @a0 {element_type = memref<64xi32>, depth = 0 : i64}
conduit.create @relay {element_type = memref<128xi32>, depth = 0 : i64}
conduit.create @b0 {element_type = memref<64xi32>, depth = 0 : i64}
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
conduit.create @x0 {element_type = memref<64xi32>, depth = 0 : i64}
conduit.create @relay_used {element_type = memref<128xi32>, depth = 1 : i64}
conduit.create @y0 {element_type = memref<64xi32>, depth = 0 : i64}
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

// -----

// Negative test: budget overflow — 6 srcs * 6 dsts = 36 > 32 (packet ID limit).
// The pass should NOT fuse even though gather.dst == scatter.src and same memtile.

aie.device(npu1) {
conduit.create @s0 {element_type = memref<16xi32>, depth = 0 : i64}
conduit.create @s1 {element_type = memref<16xi32>, depth = 0 : i64}
conduit.create @s2 {element_type = memref<16xi32>, depth = 0 : i64}
conduit.create @s3 {element_type = memref<16xi32>, depth = 0 : i64}
conduit.create @s4 {element_type = memref<16xi32>, depth = 0 : i64}
conduit.create @s5 {element_type = memref<16xi32>, depth = 0 : i64}
conduit.create @mid {element_type = memref<96xi32>, depth = 0 : i64}
conduit.create @d0 {element_type = memref<16xi32>, depth = 0 : i64}
conduit.create @d1 {element_type = memref<16xi32>, depth = 0 : i64}
conduit.create @d2 {element_type = memref<16xi32>, depth = 0 : i64}
conduit.create @d3 {element_type = memref<16xi32>, depth = 0 : i64}
conduit.create @d4 {element_type = memref<16xi32>, depth = 0 : i64}
conduit.create @d5 {element_type = memref<16xi32>, depth = 0 : i64}
// CHECK-LABEL: func.func @no_fuse_budget_overflow
// CHECK:       conduit.gather
// CHECK:       conduit.scatter
func.func @no_fuse_budget_overflow() {
  conduit.gather{srcs = [@s0, @s1, @s2, @s3, @s4, @s5], dst = @mid {memtile = "tile(0,1)"}}
  conduit.scatter{src = @mid, dsts = [@d0, @d1, @d2, @d3, @d4, @d5] {memtile = "tile(0,1)"}}
  return
}
}
