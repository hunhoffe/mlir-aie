// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit %s 2>&1 | FileCheck %s
//
// Pass B static strides extraction test.
//
// Verifies three scenarios:
//
//   1. Static 1-D descriptor: offsets/sizes/strides from arith.constant (index type).
//      Pass B must emit non-empty DenseI64ArrayAttr fields with the correct values.
//
//   2. i32 constant stride: arith.constant 4 : i32 (integer type, not index).
//      ConstantIndexOp::classof() rejects this (result type is i32, not index).
//      The generic ConstantOp fallback using getZExtValue() must extract it
//      correctly without sign-extension artifacts.
//
//   3. Dynamic (non-constant) offset: a function block argument.
//      Pass B must replace it with placeholder 0 and emit a warning that names
//      the channel.

// ======================================================================
// Scenario 3 warning check: must appear at the TOP of FileCheck patterns
// because the diagnostic is printed before the IR in the combined output.
// ======================================================================
// CHECK: warning: AirChannelToConduit: channel @chan_dyn has dynamic

// ======================================================================
// Scenario 1 checks: index-typed arith.constant operands
// ======================================================================
// CHECK-LABEL: func.func @test_index_constants
// CHECK:   %[[TOK0:.*]] = conduit.put_memref_async
// CHECK-SAME:   name = "chan_index"
// CHECK-SAME:   num_elems = 32
// CHECK-SAME:   offsets = array<i64: 16>
// CHECK-SAME:   sizes = array<i64: 32>
// CHECK-SAME:   strides = array<i64: 1>
// CHECK-SAME:   : !conduit.dma.token
// CHECK:   %[[TOK1:.*]] = conduit.get_memref_async
// CHECK-SAME:   name = "chan_index"
// CHECK-SAME:   num_elems = 32
// CHECK-SAME:   offsets = array<i64: 16>
// CHECK-SAME:   sizes = array<i64: 32>
// CHECK-SAME:   strides = array<i64: 1>
// CHECK-SAME:   : !conduit.dma.token

// ======================================================================
// Scenario 2 checks: i32 constant stride (generic ConstantOp fallback)
// ======================================================================
// CHECK-LABEL: func.func @test_i32_stride
// CHECK:   conduit.put_memref_async
// CHECK-SAME:   name = "chan_i32"
// CHECK-SAME:   strides = array<i64: 4>

// ======================================================================
// Scenario 3 IR checks: placeholder 0 + no residual air ops
// ======================================================================
// CHECK-LABEL: func.func @test_dynamic_offset
// CHECK:   conduit.put_memref_async
// CHECK-SAME:   name = "chan_dyn"
// CHECK-SAME:   offsets = array<i64: 0>
// CHECK-SAME:   sizes = array<i64: 64>
// CHECK-SAME:   strides = array<i64: 1>

// ======================================================================
// No residual AIR ops in the output
// ======================================================================
// CHECK-NOT: air.channel
// CHECK-NOT: air.wait_all

module {

  // -----------------------------------------------------------------------
  // Scenario 1: index-typed arith.constant operands
  // -----------------------------------------------------------------------
  "air.channel"() {sym_name = "chan_index", size = [1, 1]} : () -> ()

  func.func @test_index_constants(
      %src : memref<128xi32>,
      %dst : memref<128xi32>) {
    %c16  = arith.constant 16 : index
    %c32  = arith.constant 32 : index
    %c1   = arith.constant 1  : index

    // offsets=[16], sizes=[32], strides=[1], num_elems=32
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
  // Scenario 2: i32 stride constant.
  // arith.constant 4 : i32 does not satisfy ConstantIndexOp::classof()
  // (its result type is i32, not index).  The generic ConstantOp branch with
  // getZExtValue() must extract the value 4 correctly.
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
  // Pass B must: (a) replace with placeholder 0, (b) emit a warning naming
  // the channel (@chan_dyn).
  // -----------------------------------------------------------------------
  "air.channel"() {sym_name = "chan_dyn", size = [1, 1]} : () -> ()

  func.func @test_dynamic_offset(
      %src    : memref<128xi32>,
      %dynoff : index) {             // block argument — not an arith.constant
    %c64 = arith.constant 64 : index
    %c1  = arith.constant 1  : index

    // offset is dynamic: should produce offsets=[0] (placeholder) + warning.
    %tok0 = "air.channel.put"(%src, %dynoff, %c64, %c1)
        {chan_name = @chan_dyn,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<128xi32>, index, index, index) -> !air.async.token

    "air.wait_all"(%tok0) : (!air.async.token) -> ()
    return
  }
}
