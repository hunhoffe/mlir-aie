// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit %s 2>&1 | FileCheck %s
//
// Pass B broadcast_shape capacity propagation test.
//
// Verifies that broadcast_shape dimensions are multiplied to produce the
// correct capacity on conduit.create for several common fan-out shapes:
//
//   channel_1x2: broadcast_shape=[1,2] → capacity=2
//   channel_2x1: broadcast_shape=[2,1] → capacity=2
//   channel_2x2: broadcast_shape=[2,2] → capacity=4
//   channel_1x4: broadcast_shape=[1,4] → capacity=4
//   channel_scalar: no broadcast_shape → capacity=1 (default)
//
// Consumer tile coordinates are not available at Pass B time; conduit.create
// is emitted with correct capacity and empty consumer_tiles.
// Full wiring requires a tile-placement pre-pass to populate consumer_tiles.

// CHECK: remark{{.*}}channel_1x2{{.*}}capacity=2
// CHECK: remark{{.*}}channel_2x1{{.*}}capacity=2
// CHECK: remark{{.*}}channel_2x2{{.*}}capacity=4
// CHECK: remark{{.*}}channel_1x4{{.*}}capacity=4

// CHECK-LABEL: module

// channel_1x2: capacity=2
// CHECK: conduit.create @channel_1x2
// CHECK-SAME: capacity = 2

// channel_2x1: capacity=2
// CHECK: conduit.create @channel_2x1
// CHECK-SAME: capacity = 2

// channel_2x2: capacity=4
// CHECK: conduit.create @channel_2x2
// CHECK-SAME: capacity = 4

// channel_1x4: capacity=4
// CHECK: conduit.create @channel_1x4
// CHECK-SAME: capacity = 4

// channel_scalar: no broadcast_shape, capacity=1 default
// CHECK: conduit.create @channel_scalar
// CHECK-SAME: capacity = 1

// No residual air.channel ops.
// CHECK-NOT: air.channel

module {
  // 1×2 broadcast (column broadcast in a 2-row herd)
  "air.channel"() {sym_name = "channel_1x2", size = [1, 1],
                   broadcast_shape = array<i64: 1, 2>} : () -> ()

  // 2×1 broadcast (row broadcast in a 2-column herd)
  "air.channel"() {sym_name = "channel_2x1", size = [1, 1],
                   broadcast_shape = array<i64: 2, 1>} : () -> ()

  // 2×2 broadcast (full 2×2 herd broadcast)
  "air.channel"() {sym_name = "channel_2x2", size = [1, 1],
                   broadcast_shape = array<i64: 2, 2>} : () -> ()

  // 1×4 broadcast (4-wide column broadcast)
  "air.channel"() {sym_name = "channel_1x4", size = [1, 1],
                   broadcast_shape = array<i64: 1, 4>} : () -> ()

  // Scalar channel: no broadcast_shape, should produce capacity=1.
  "air.channel"() {sym_name = "channel_scalar", size = [1, 1]} : () -> ()

  func.func @test(%buf : memref<16xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // A put on each channel to populate element_type.
    "air.channel.put"(%buf, %c0, %c1, %c1)
        {chan_name = @channel_1x2,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<16xi32>, index, index, index) -> ()
    "air.channel.put"(%buf, %c0, %c1, %c1)
        {chan_name = @channel_2x1,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<16xi32>, index, index, index) -> ()
    "air.channel.put"(%buf, %c0, %c1, %c1)
        {chan_name = @channel_2x2,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<16xi32>, index, index, index) -> ()
    "air.channel.put"(%buf, %c0, %c1, %c1)
        {chan_name = @channel_1x4,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<16xi32>, index, index, index) -> ()
    "air.channel.put"(%buf, %c0, %c1, %c1)
        {chan_name = @channel_scalar,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<16xi32>, index, index, index) -> ()
    return
  }
}
