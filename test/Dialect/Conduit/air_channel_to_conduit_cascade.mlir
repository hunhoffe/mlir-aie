// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s
//
// Pass B (--air-channel-to-conduit) cascade channel_type test.
//
// Verifies that air.channel declarations with channel_type = "cascade" emit
// a conduit.create with routing_mode = #conduit.routing_mode<cascade> (no longer a hard error),
// and that put/get ops are rewritten to conduit.put_cascade / conduit.get_cascade.
//
// Uses memref<1xvector<16xi32>>: element type vector<16xi32> = 512 bits (AIE2).
// The cascade verifier rejects types that are not 384 (AIE1) or 512 (AIE2) bits.
//
// Trivial (zero-offset, unit-size, unit-stride) operands do not produce
// any diagnostic. Non-trivial / dynamic operands are tested separately in
// air_channel_to_conduit_cascade_offset_warning.mlir.

// CHECK-LABEL: module

// --- Cascade channel: routing_mode = #conduit.routing_mode<cascade> ---
// CHECK:   conduit.create
// CHECK-SAME: name = "cas_chan"
// CHECK-SAME: routing_mode = #conduit.routing_mode<cascade>

// CHECK-NOT: air.channel

// conduit.put_cascade is emitted for the put path (load from memref[0]).
// CHECK: conduit.put_cascade "cas_chan"
// CHECK-SAME: vector<16xi32>

// conduit.get_cascade is emitted for the get path.
// CHECK: conduit.get_cascade "cas_chan"
// CHECK-SAME: vector<16xi32>

module {
  // air.channel declaration with channel_type = "cascade".
  "air.channel"() {sym_name = "cas_chan", size = [1, 1],
                   channel_type = "cascade"} : () -> ()

  // Static-offset put: straightforward load + put_cascade.
  func.func @test_cascade_put(%src : memref<1xvector<16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    "air.channel.put"(%src, %c0, %c1, %c1)
        {chan_name = @cas_chan,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<1xvector<16xi32>>, index, index, index)
        -> ()
    return
  }

  // Static-offset get: get_cascade + store.
  func.func @test_cascade_get(%dst : memref<1xvector<16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    "air.channel.get"(%dst, %c0, %c1, %c1)
        {chan_name = @cas_chan,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<1xvector<16xi32>>, index, index, index)
        -> ()
    return
  }
}
