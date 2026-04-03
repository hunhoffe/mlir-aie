// RUN: /scratch/ehunhoff/conduit-notes/mlir-air/build/bin/air-opt --air-channel-to-conduit --verify-diagnostics %s
//
// Regression test: B-7 — rank-3 memref operand in air.channel.put is handled
// by collapsing leading dimensions to rank-2 (emitting a warning, not an error).
//
// memref<4x4x4xi32> → memref<16x4xi32> (4*4=16 leading dims folded).
// The air.channel.put offsets/sizes/strides are 1-D descriptors independent of
// memref shape, so the DMA descriptor itself is unaffected; only element_type
// on conduit.create is collapsed to rank-2.
//
// Uses air-opt (which has the AIR dialect registered) so that actual
// air.channel syntax can be used.

module {
  // The "remaining uses" error fires on the channel decl because the bare
  // module context has no aie.device, and conduit.put_memref_async keeps
  // a FlatSymbolRefAttr @chan3d in scope. This is expected for a minimal test.
  // expected-error @below {{channel decl 'chan3d' has remaining uses after rewrite}}
  air.channel @chan3d [1, 1]

  func.func @test_rank3(%src : memref<4x4x4xi32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index

    // air.channel.put with a rank-3 memref (4x4x4xi32) — B-7 now collapses
    // to rank-2 and emits a warning instead of a hard error.
    // expected-warning @below {{air-channel-to-conduit: rank-3 memref operand for @chan3d collapsed to rank-2}}
    air.channel.put @chan3d[] (%src[%c0, %c0, %c0][%c4, %c4, %c4][%c16, %c4, %c1])
        : (memref<4x4x4xi32>)
    return
  }
}
