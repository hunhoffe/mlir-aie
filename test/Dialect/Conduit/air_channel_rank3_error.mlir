// RUN: /scratch/ehunhoff/conduit-notes/mlir-air/build/bin/air-opt --air-channel-to-conduit --verify-diagnostics %s
//
// Regression test: B-7 — rank-3 memref operand in air.channel.put emits a
// hard error instead of silently producing wrong output.
//
// Before the B-7 fix, rank-3 operands were silently truncated to the 2-D
// offset/size/stride descriptor, producing DMA descriptors that would access
// wrong memory regions on hardware.
//
// Uses air-opt (which has the AIR dialect registered) so that actual
// air.channel syntax can be used. The expected-error annotation is consumed
// by --verify-diagnostics and causes the test to PASS when the error fires.

module {
  air.channel @chan3d [1, 1]

  func.func @test_rank3(%src : memref<4x4x4xi32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index

    // air.channel.put with a rank-3 memref (4x4x4xi32) — B-7 must reject this.
    // expected-error @below {{air-channel-to-conduit: rank-3 memref operand is not supported}}
    air.channel.put @chan3d[] (%src[%c0, %c0, %c0][%c4, %c4, %c4][%c16, %c4, %c1])
        : (memref<4x4x4xi32>)
    return
  }
}
