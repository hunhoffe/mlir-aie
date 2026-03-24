// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s
//
// Pass B (--air-channel-to-conduit) air.execute wrapper test.
//
// Verifies that programs using air.execute wrappers (which wrap memref.alloc and
// yield SSA values) are correctly lowered. The air.execute ops remain as
// unregistered ops (with --allow-unregistered-dialect), but the memref SSA values
// they yield correctly thread through to air.channel.put/get operands and are
// decoded by Phase 2b for element_type inference.
//
// This pattern is universal in the mlir-air corpus: ~136 of 108 files use
// air.execute to wrap memref allocation before channel put/get.
//
// Expected: conduit.create with correct element_type inferred from the
// air.execute-allocated memref; conduit.put_memref_async / get_memref_async emitted.

// CHECK-LABEL: module

// element_type is inferred from the memref<16x16xf32, 2> despite it being
// defined inside an air.execute region.
// CHECK: conduit.create @chan
// CHECK-SAME: element_type = memref<16x16xf32, 2>

// put and get are lowered correctly.
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: num_elems = 256

// CHECK: conduit.get_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: num_elems = 256

// air.channel decl is gone.
// CHECK-NOT: air.channel{{[^._]}}

module {
  "air.channel"() {sym_name = "chan", size = [1, 1]} : () -> ()

  func.func @test_air_execute_wrapper() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    // air.execute wraps memref.alloc: the result %src_buf is a block argument
    // of type memref<16x16xf32, 2>.  This pattern is the standard way AIR
    // programs allocate local L1 buffers for DMA.
    //
    // Pass B correctly sees %src_buf as the memref operand of channel.put
    // and infers element_type = memref<16x16xf32, 2> for conduit.create.
    %exec_tok, %src_buf = "air.execute"() ({
      %alloc = memref.alloc() : memref<16x16xf32, 2>
      "air.execute_terminator"(%alloc) : (memref<16x16xf32, 2>) -> ()
    }) : () -> (!air.async.token, memref<16x16xf32, 2>)

    %exec_tok2, %dst_buf = "air.execute"(%exec_tok) ({
      %alloc = memref.alloc() : memref<16x16xf32, 2>
      "air.execute_terminator"(%alloc) : (memref<16x16xf32, 2>) -> ()
    }) : (!air.async.token) -> (!air.async.token, memref<16x16xf32, 2>)

    // air.channel.put using the air.execute-allocated buffer.
    %put_tok = "air.channel.put"(%exec_tok2, %src_buf, %c0, %c0, %c16, %c16, %c16, %c1)
        {chan_name = @chan,
         operand_segment_sizes = array<i32: 1, 0, 1, 2, 2, 2>}
        : (!air.async.token, memref<16x16xf32, 2>,
           index, index, index, index, index, index)
        -> !air.async.token

    // air.channel.get using the second air.execute-allocated buffer.
    %get_tok = "air.channel.get"(%exec_tok2, %dst_buf, %c0, %c0, %c16, %c16, %c16, %c1)
        {chan_name = @chan,
         operand_segment_sizes = array<i32: 1, 0, 1, 2, 2, 2>}
        : (!air.async.token, memref<16x16xf32, 2>,
           index, index, index, index, index, index)
        -> !air.async.token

    // Fan-in on both tokens.
    "air.wait_all"(%put_tok, %get_tok)
        : (!air.async.token, !air.async.token) -> ()

    return
  }
}
