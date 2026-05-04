// RUN: aie-opt --objectfifo-to-conduit %s 2>&1 | FileCheck %s
//
// Pass A test: an aie.objectfifo carrying a `fusion_group` string attribute
// must lower to a conduit.create that carries the same `fusion_group` attribute
// (so the spatial-fusion pass --conduit-fuse-operators can pair the channel
// with another channel sharing the same group string).
//
// IRON path that exercises this:
//   ObjectFifo(..., fusion_group="swiglu_a_to_b") in
//   mlir-aie/python/iron/dataflow/objectfifo.py attaches the StringAttr to
//   the aie.objectfifo.createOp; ObjectFifoToConduit.cpp then forwards
//   op->getAttrOfType<StringAttr>("fusion_group") into the conduit.create
//   builder's fusion_group operand.
//
// Expected behavior:
//   - Producer-side conduit.create carries fusion_group = "swiglu_a_to_b".
//   - Consumer-side conduit.create carries fusion_group = "swiglu_a_to_b".
//   - The aie.objectfifo ops are erased.

// CHECK-LABEL: module @fusion_group_convert
// CHECK:   aie.device(npu1_1col) {
// CHECK:     conduit.create @of_a
// CHECK-SAME:   fusion_group = "swiglu_a_to_b"
// CHECK:     conduit.create @of_b
// CHECK-SAME:   fusion_group = "swiglu_a_to_b"
// CHECK-NOT: aie.objectfifo

module @fusion_group_convert {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @of_a (%tile_0_0, {%tile_0_2}, 2 : i32) {fusion_group = "swiglu_a_to_b"} : !aie.objectfifo<memref<16xbf16>>
    aie.objectfifo @of_b (%tile_0_2, {%tile_0_3}, 2 : i32) {fusion_group = "swiglu_a_to_b"} : !aie.objectfifo<memref<16xbf16>>
  }
}
