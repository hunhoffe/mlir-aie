// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit --verify-diagnostics %s
//
// Pass B strides extraction test: static constants + dynamic error.
//
// Verifies three scenarios:
//
//   1. Static 1-D descriptor: offsets/sizes/strides from arith.constant (index type).
//      Pass B must extract correct values.
//
//   2. i32 constant stride: arith.constant 4 : i32 (integer type, not index).
//      The generic ConstantOp fallback using getSExtValue() must extract it
//      correctly without sign-extension artifacts.
//
//   3. Dynamic (non-constant) offset: a function block argument.
//      Pass B must emit a hard error and fail — placeholder substitution
//      produces wrong DMA descriptors (stride=0 reads same address repeatedly).
//
// This test uses --verify-diagnostics to check the expected-error on scenario 3.
// Static extraction correctness (scenarios 1 & 2) is covered by
// air_channel_to_conduit.mlir which tests 2-D static descriptors end-to-end.

module {

  // -----------------------------------------------------------------------
  // Scenario 1: index-typed arith.constant operands (no error expected)
  // -----------------------------------------------------------------------
  "air.channel"() {sym_name = "chan_index", size = [1, 1]} : () -> ()

  func.func @test_index_constants(
      %src : memref<128xi32>,
      %dst : memref<128xi32>) {
    %c16  = arith.constant 16 : index
    %c32  = arith.constant 32 : index
    %c1   = arith.constant 1  : index

    // offsets=[16], sizes=[32], strides=[1], num_elems=32 — all static, no error.
    %tok0 = "air.channel.put"(%src, %c16, %c32, %c1)
        {chan_name = @chan_index,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<128xi32>, index, index, index) -> !air.async.token

    %tok1 = "air.channel.get"(%dst, %c16, %c32, %c1)
        {chan_name = @chan_index,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<128xi32>, index, index, index) -> !air.async.token

    "air.wait_all"(%tok0, %tok1)
        : (!air.async.token, !air.async.token) -> ()
    return
  }

  // -----------------------------------------------------------------------
  // Scenario 2: i32 stride constant (no error expected)
  // -----------------------------------------------------------------------
  "air.channel"() {sym_name = "chan_i32", size = [1, 1]} : () -> ()

  func.func @test_i32_stride(
      %src : memref<128xi32>) {
    %c0  = arith.constant 0  : index
    %c64 = arith.constant 64 : index
    %s4  = arith.constant 4  : i32   // stride as i32, not index

    %tok0 = "air.channel.put"(%src, %c0, %c64, %s4)
        {chan_name = @chan_i32,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<128xi32>, index, index, i32) -> !air.async.token

    "air.wait_all"(%tok0) : (!air.async.token) -> ()
    return
  }

  // -----------------------------------------------------------------------
  // Scenario 3: dynamic offset (function block argument, not a constant).
  // Pass B must emit a hard error — placeholder substitution produces
  // wrong DMA descriptors on hardware.
  // -----------------------------------------------------------------------
  "air.channel"() {sym_name = "chan_dyn", size = [1, 1]} : () -> ()

  func.func @test_dynamic_offset(
      %src    : memref<128xi32>,
      %dynoff : index) {             // block argument — not an arith.constant
    %c64 = arith.constant 64 : index
    %c1  = arith.constant 1  : index

    // expected-error @+1 {{air-channel-to-conduit: channel @chan_dyn has dynamic offset/size/stride operands}}
    %tok0 = "air.channel.put"(%src, %dynoff, %c64, %c1)
        {chan_name = @chan_dyn,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<128xi32>, index, index, index) -> !air.async.token

    "air.wait_all"(%tok0) : (!air.async.token) -> ()
    return
  }
}
