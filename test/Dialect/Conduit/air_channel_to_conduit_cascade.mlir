// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit --verify-each=false %s | FileCheck %s
//
// Pass B (--air-channel-to-conduit) cascade channel_type test.
//
// Verifies that air.channel declarations with channel_type = "cascade" emit
// a conduit.create with routing_mode = #conduit.routing_mode<cascade> (no longer a hard error),
// and that put/get ops are rewritten to aie.put_cascade / aie.get_cascade.
//
// Uses memref<1xvector<16xi32>>: element type vector<16xi32> = 512 bits (AIE2).
//
// Note: --verify-each=false is required because the AIE verifier rejects
// aie.put_cascade / aie.get_cascade inside func.func bodies (no aie.device
// context). In the full pipeline (air-opt with --air-hierarchy-to-aie first),
// these ops appear inside aie.core regions and the verifier is satisfied.
// This unit test exercises Pass B in isolation without hierarchy lowering.
//
// With --verify-each=false, output is in generic (quoted) form.

// CHECK-LABEL: "builtin.module"

// --- Cascade channel: conduit.create with routing_mode = cascade ---
// CHECK: "conduit.create"
// CHECK-SAME: routing_mode = #conduit.routing_mode<cascade>
// CHECK-SAME: cas_chan

// CHECK-NOT: air.channel

// aie.put_cascade is emitted for the put path (load from memref[0]).
// After cascade migration (#27), Pass B emits aie.put_cascade directly.
// CHECK: "aie.put_cascade"
// CHECK-SAME: vector<16xi32>

// aie.get_cascade is emitted for the get path.
// CHECK: "aie.get_cascade"

module {
  // air.channel declaration with channel_type = "cascade".
  "air.channel"() {sym_name = "cas_chan", size = [1, 1],
                   channel_type = "cascade"} : () -> ()

  // Static-offset put: straightforward load + aie.put_cascade.
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

  // Static-offset get: aie.get_cascade + store.
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
