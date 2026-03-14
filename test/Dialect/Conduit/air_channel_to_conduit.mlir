// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s
//
// Pass B (--air-channel-to-conduit) basic test: lowers air.channel.put/get to Conduit Tier 3 ops.
//
// Pass B test: lower air.channel.put / air.channel.get to Conduit Tier 3 ops.
//
// Input: a minimal AIR channel program in generic MLIR notation (because the
// AIR dialect is not registered in aie-opt; --allow-unregistered-dialect is
// used to parse the unregistered ops).
//
// The program contains:
//   - one air.channel declaration @chan [1, 1]
//   - one air.channel.put (async) with 8x8 descriptor: offsets=[0,0], sizes=[8,8], strides=[8,1]
//   - one air.channel.get (async) with matching descriptor
//   - one air.wait_all (async)
//
// This is the SPMD-specialized form that Pass B receives (Task #32):
// channels are [1,1] scalars, no multi-dimensional indices.
//
// Expected output after --air-channel-to-conduit:
//   - conduit.create {name="chan", capacity=1, depth=1}
//     with element_type inferred from the put memref operand type
//   - conduit.put_memref_async with name="chan", num_elems=64, offsets/sizes/strides extracted
//   - conduit.get_memref_async with name="chan", num_elems=64, matching descriptor
//   - conduit.wait_all_async joining the two tokens
//   - NO air.channel / air.wait_all ops remaining

// CHECK-LABEL: module
//
// --- Channel declaration becomes conduit.create ---
// Attributes are printed in alphabetical order:
//   capacity, depth, element_type, name
// CHECK:   conduit.create
// CHECK-SAME: capacity = 1
// CHECK-SAME: depth = 1
// CHECK-SAME: element_type = memref<8x8xi32>
// CHECK-SAME: name = "chan"
//
// --- air.channel.put becomes conduit.put_memref_async ---
// Static descriptor: offsets=[0,0], sizes=[8,8], strides=[8,1], num_elems=8*8=64
// CHECK:   %[[TOK0:.*]] = conduit.put_memref_async
// CHECK-SAME: name = "chan"
// CHECK-SAME: num_elems = 64
// CHECK-SAME: offsets = array<i64: 0, 0>
// CHECK-SAME: sizes = array<i64: 8, 8>
// CHECK-SAME: strides = array<i64: 8, 1>
// CHECK-SAME: : !conduit.dma.token
//
// --- air.channel.get becomes conduit.get_memref_async ---
// CHECK:   %[[TOK1:.*]] = conduit.get_memref_async
// CHECK-SAME: name = "chan"
// CHECK-SAME: num_elems = 64
// CHECK-SAME: offsets = array<i64: 0, 0>
// CHECK-SAME: sizes = array<i64: 8, 8>
// CHECK-SAME: strides = array<i64: 8, 1>
// CHECK-SAME: : !conduit.dma.token
//
// --- air.wait_all async becomes conduit.wait_all_async ---
// CHECK:   conduit.wait_all_async %[[TOK0]], %[[TOK1]]
// CHECK-SAME: (!conduit.dma.token, !conduit.dma.token) -> !conduit.dma.token
//
// --- No residual AIR ops ---
// CHECK-NOT: air.channel
// CHECK-NOT: air.wait_all

module {
  // Air channel declaration: sym_name="chan", size=[1,1].
  // Generic notation so aie-opt (with --allow-unregistered-dialect) accepts it.
  "air.channel"() {sym_name = "chan", size = [1, 1]} : () -> ()

  func.func @test_air_channel_put_get(
      %src : memref<8x8xi32>,
      %dst : memref<8x8xi32>) {

    // Static constants for the memref descriptor.
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index

    // air.channel.put async: push a 8x8 tile from %src into @chan.
    //
    // The operand_segment_sizes attribute encodes the segmentation:
    //   [ndeps=0, nidx=0, nmemref=1, noffsets=2, nsizes=2, nstrides=2]
    //
    // Operands in order:
    //   (src, off0, off1, size0, size1, stride0, stride1)
    //    = (%src, %c0, %c0, %c8, %c8, %c8, %c1)
    //
    // So offsets=[0,0], sizes=[8,8], strides=[8,1], num_elems=8*8=64.
    %tok0 = "air.channel.put"(%src, %c0, %c0, %c8, %c8, %c8, %c1)
        {chan_name = @chan,
         operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>}
        : (memref<8x8xi32>, index, index, index, index, index, index)
        -> !air.async.token

    // air.channel.get async: pull a 8x8 tile from @chan into %dst.
    // Same descriptor shape as the put.
    %tok1 = "air.channel.get"(%dst, %c0, %c0, %c8, %c8, %c8, %c1)
        {chan_name = @chan,
         operand_segment_sizes = array<i32: 0, 0, 1, 2, 2, 2>}
        : (memref<8x8xi32>, index, index, index, index, index, index)
        -> !air.async.token

    // air.wait_all async: fan-in over both tokens.
    %merged = "air.wait_all"(%tok0, %tok1)
        : (!air.async.token, !air.async.token) -> !air.async.token

    return
  }
}
