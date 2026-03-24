// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s
//
// Pass B (--air-channel-to-conduit) channel_type propagation test.
//
// Verifies that air.channel declarations with channel_type = "dma_packet" produce
// a conduit.create with routing_mode = #conduit.routing_mode<packet>, while channels without channel_type
// (dma_stream default) produce a conduit.create with no routing_mode attribute.
//
// Sprint item 5a: channel_type → routing_mode propagation.

// CHECK-LABEL: module

// --- Packet channel: routing_mode = #conduit.routing_mode<packet> ---
// CHECK:   conduit.create @pkt_chan
// CHECK-SAME: routing_mode = #conduit.routing_mode<packet>

// --- Stream channel: no routing_mode attribute ---
// CHECK:   conduit.create @stream_chan
// CHECK-NOT: routing_mode

// CHECK-NOT: air.channel

module {
  // air.channel declaration with channel_type = "dma_packet".
  // Expected: conduit.create with routing_mode = #conduit.routing_mode<packet>.
  "air.channel"() {sym_name = "pkt_chan", size = [1, 1],
                   channel_type = "dma_packet"} : () -> ()

  // air.channel declaration with no channel_type (dma_stream default).
  // Expected: conduit.create with no routing_mode attr.
  "air.channel"() {sym_name = "stream_chan", size = [1, 1]} : () -> ()

  func.func @test_packet_channel(
      %src : memref<4x4xi32>,
      %dst : memref<4x4xi32>) {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // Put through the packet channel.
    %tok0 = "air.channel.put"(%src, %c0, %c0, %c4, %c4, %c4, %c1)
        {chan_name = @pkt_chan,
         operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>}
        : (memref<4x4xi32>, index, index, index, index, index, index)
        -> !air.async.token

    // Get through the stream channel.
    %tok1 = "air.channel.get"(%dst, %c0, %c0, %c4, %c4, %c4, %c1)
        {chan_name = @stream_chan,
         operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>}
        : (memref<4x4xi32>, index, index, index, index, index, index)
        -> !air.async.token

    return
  }
}
