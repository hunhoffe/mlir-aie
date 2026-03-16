// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit --verify-diagnostics %s
//
// Pass B cascade offset/size/stride diagnostic test.
//
// Verifies that non-trivial offset/size/stride operands on cascade channels
// produce the correct diagnostics:
//
//   1. Constant non-zero offset → warning mentioning "element[0]"
//   2. Constant non-unit size   → warning mentioning "element[0]"
//   3. Constant non-unit stride → warning mentioning "element[0]"
//   4. Fully dynamic (non-constant) offset → hard error (intent cannot be inferred)
//
// Trivial (zero-offset, unit-size, unit-stride) operands on cascade channels
// produce no diagnostic (tested in air_channel_to_conduit_cascade.mlir).

module {

  // -----------------------------------------------------------------------
  // Scenario 1: constant non-zero offset → warning
  // -----------------------------------------------------------------------
  "air.channel"() {sym_name = "cas_nonzero_off", size = [1, 1],
                   channel_type = "cascade"} : () -> ()

  func.func @test_nonzero_offset(%src : memref<1xvector<16xi32>>) {
    %c4 = arith.constant 4 : index  // non-zero offset
    %c1 = arith.constant 1 : index

    // expected-warning @+1 {{non-zero offset}}
    "air.channel.put"(%src, %c4, %c1, %c1)
        {chan_name = @cas_nonzero_off,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<1xvector<16xi32>>, index, index, index)
        -> ()
    return
  }

  // -----------------------------------------------------------------------
  // Scenario 2: constant non-unit size → warning
  // -----------------------------------------------------------------------
  "air.channel"() {sym_name = "cas_nonunit_sz", size = [1, 1],
                   channel_type = "cascade"} : () -> ()

  func.func @test_nonunit_size(%src : memref<4xvector<16xi32>>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index  // non-unit size
    %c1 = arith.constant 1 : index

    // expected-warning @+1 {{non-unit size}}
    "air.channel.put"(%src, %c0, %c4, %c1)
        {chan_name = @cas_nonunit_sz,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<4xvector<16xi32>>, index, index, index)
        -> ()
    return
  }

  // -----------------------------------------------------------------------
  // Scenario 3: constant non-unit stride → warning
  // -----------------------------------------------------------------------
  "air.channel"() {sym_name = "cas_nonunit_st", size = [1, 1],
                   channel_type = "cascade"} : () -> ()

  func.func @test_nonunit_stride(%src : memref<1xvector<16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index  // non-unit stride

    // expected-warning @+1 {{non-unit stride}}
    "air.channel.put"(%src, %c0, %c1, %c2)
        {chan_name = @cas_nonunit_st,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<1xvector<16xi32>>, index, index, index)
        -> ()
    return
  }

  // -----------------------------------------------------------------------
  // Scenario 4: fully dynamic (non-constant) offset → hard error
  // -----------------------------------------------------------------------
  "air.channel"() {sym_name = "cas_dyn_off", size = [1, 1],
                   channel_type = "cascade"} : () -> ()

  func.func @test_dynamic_offset(%src : memref<1xvector<16xi32>>,
                                  %dyn : index) {
    %c1 = arith.constant 1 : index

    // expected-error @+1 {{fully dynamic offset operand}}
    "air.channel.put"(%src, %dyn, %c1, %c1)
        {chan_name = @cas_dyn_off,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<1xvector<16xi32>>, index, index, index)
        -> ()
    return
  }

}
