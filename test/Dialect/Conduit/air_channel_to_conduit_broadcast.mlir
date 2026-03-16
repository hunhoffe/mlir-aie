// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit %s 2>&1 | FileCheck %s
//
// Pass B (--air-channel-to-conduit) broadcast_shape diagnostic test.
//
// Verifies that air.channel declarations with a broadcast_shape attribute produce
// a diagnostic warning (not silently dropped) and still emit a conduit.create.
//
// Sprint item 5b: broadcast_shape → warning + conduit.create emission.
//
// Note: FileCheck is run on combined stdout+stderr (2>&1) so it can check
// the warning message emitted by mlir's diagnostic system.

// Output order (stderr+stdout combined):
//   line 1: warning: ... broadcast_shape ...
//   line 2: "air.channel"() ... (source echo from diagnostic)
//   line 3: ^
//   line 4: note: see current operation: ...
//   line 5: module {
//   line 6: conduit.create {...name = "bcast_chan"...routing_mode = "packet"}
//
// Checks must follow the output order.

// 1. Warning contains "broadcast_shape".
// CHECK: warning{{.*}}broadcast_shape

// 2. Module opens (comes before conduit.create in output).
// CHECK: module {

// 3. conduit.create with name and routing_mode on same line.
// CHECK: conduit.create{{.*}}name = "bcast_chan"{{.*}}routing_mode = "packet"

// 4. No further air.channel ops in module body (source echoes already passed).
// CHECK-NOT: air.channel

module {
  // air.channel with broadcast_shape = [1, 4] and channel_type = "dma_packet".
  // This is a typical 1-to-4 packet broadcast pattern from the mlir-air corpus
  // (e.g., L2ToL1 channels in matmul workloads).
  "air.channel"() {sym_name = "bcast_chan", size = [1, 1],
                   broadcast_shape = array<i64: 1, 4>,
                   channel_type = "dma_packet"} : () -> ()

  func.func @test_broadcast_channel(
      %src : memref<8x8xi32>,
      %dst : memref<8x8xi32>) {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index

    %tok0 = "air.channel.put"(%src, %c0, %c0, %c8, %c8, %c8, %c1)
        {chan_name = @bcast_chan,
         operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>}
        : (memref<8x8xi32>, index, index, index, index, index, index)
        -> !air.async.token

    return
  }
}
