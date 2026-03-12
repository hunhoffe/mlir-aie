// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Pass A test: N→1 join link with byte offsets.
// Source: adapted from test/objectFifo-stateful-transform/data_movement_patterns/
//         link/link_test_join_offsets.mlir

// CHECK-LABEL: module @link_distribute_offsets
// CHECK:   aie.device(xcve2302) {
// CHECK:     conduit.create
// CHECK-SAME:   name = "link1"
// CHECK:     conduit.create
// CHECK-SAME:   name = "link2"
// CHECK:     conduit.create
// CHECK-SAME:   name = "link3"
// CHECK:     conduit.create
// CHECK-SAME:   name = "link4"
// CHECK:     conduit.objectfifo_link
// CHECK-SAME:   dsts = ["link4"]
// CHECK-SAME:   memtile = "tile(2,1)"
// CHECK-SAME:   mode = "join"
// CHECK-SAME:   offsets = array<i64: 0, 16, 36>
// CHECK-SAME:   srcs = ["link1", "link2", "link3"]
// CHECK-NOT: aie.objectfifo
// CHECK-NOT: aie.objectfifo.link

module @link_distribute_offsets {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link2 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link3 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
    aie.objectfifo @link4 (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    aie.objectfifo.link [@link1, @link2, @link3] -> [@link4] ([0, 16, 36][])
  }
}
