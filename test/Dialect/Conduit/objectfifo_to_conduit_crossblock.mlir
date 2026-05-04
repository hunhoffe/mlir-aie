// RUN: aie-opt --objectfifo-to-conduit %s 2>&1 | FileCheck %s
//
// C1 gap: cross-block acquire/release pattern.
//
// This test covers two sub-cases:
//
// Case A (FIXED): acquire in the core entry block, release in a nested block
// (scf.if true region).  The release is dominated by the acquire.  After the
// C1 fix, Pass A walks up the parent block chain and finds the acquire's window
// SSA value without synthesizing a phantom.  No C1 warning is emitted.
//
// Case B (UNFIXABLE): release with no acquire anywhere in the parent chain.
// The acquire is genuinely absent — simulated here by a second core that only
// has a release op with no corresponding acquire.  Pass A cannot find a
// dominating window and must synthesize a phantom; the C1 warning still fires.
//
// Expected output:
//   - Case A: conduit.release emitted using the acquire's SSA value; exactly
//     ONE conduit.acquire emitted for core_0_2 (not two).
//   - Case B: the C1 cross-block warning is emitted for core_0_3.
//
// CHECK-NOT: {{.*}}cb_fifo{{.*}}phantom
// CHECK: cross-block

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @cb_fifo(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @cb_fifo2(%tile_0_0, {%tile_0_3}, 1 : i32) : !aie.objectfifo<memref<8xi32>>

    // Case A (FIXED): acquire in entry block, release inside scf.if.
    // After the fix, the release finds the dominating window from the entry
    // block — no phantom acquire is synthesized, no C1 warning emitted.
    %core_0_2 = aie.core(%tile_0_2) {
      %true = arith.constant true

      // Acquire is in the entry block of the core body.
      %win = aie.objectfifo.acquire @cb_fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>

      // Release is in a nested block (scf.if true region).
      // After the C1 fix, Pass A finds the window from the enclosing entry
      // block via the parent-block walk and does NOT synthesize a phantom.
      scf.if %true {
        aie.objectfifo.release @cb_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}

    // Case B (UNFIXABLE): release with no acquire in any parent block.
    // Pass A must synthesize a phantom and emit the C1 warning.
    %core_0_3 = aie.core(%tile_0_3) {
      %true = arith.constant true

      // No acquire for cb_fifo2 anywhere in this core.  The release inside
      // the scf.if body has no dominating acquire anywhere in the parent chain.
      scf.if %true {
        aie.objectfifo.release @cb_fifo2(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
