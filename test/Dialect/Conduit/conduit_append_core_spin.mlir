// RUN: aie-opt --conduit-append-core-spin %s | FileCheck %s
//
// Test for --conduit-append-core-spin pass.
//
// Verifies that an empty bounded "infinite" spin loop is appended just before
// every aie.core's terminating aie.end op.  Background: Llama decode-hang
// (Task #72/#83/#89) is localized to op18_GEMV (LM-head, ni=1) at the LAST
// orchestrator-position; cores with finite inner-loop trip counts reach
// aie.end after their work, exposing a firmware-runtime hang at
// orchestrator-position-LAST.  Appending an infinite spin prevents cores
// from ever reaching aie.end.

// CHECK-LABEL: aie.core
// Original body's finite scf.for must be preserved.
// CHECK: scf.for
// CHECK:   arith.addi
// Inserted spin loop bounds: [0, i64-max-1) step 1.  Bound bumped from
// 16777214 (0xFFFFFE, ~16 ms at AIE2 ~1 GHz) to 9223372036854775806 (i64
// max - 1) per Task #105 — the original 16 ms bound was finishing in
// ~10% of a 150 ms decode tick, letting cores hit aie.end early and
// re-trigger the orchestrator-tail hang Task #90 was meant to suppress.
// CHECK: arith.constant 0 : index
// CHECK: arith.constant 9223372036854775806 : index
// CHECK: arith.constant 1 : index
// CHECK: scf.for
// Inserted body: single trivial arith.constant.
// CHECK: arith.constant 0 : index
// CHECK: }
// aie.end is still present (just unreachable now).
// CHECK: aie.end

module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %ub = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      // Original body has a FINITE scf.for (the bug-trigger pattern that
      // motivates the spin-loop append).
      scf.for %i = %c0 to %ub step %c1 {
        %a = arith.constant 1 : i32
        %b = arith.constant 2 : i32
        %s = arith.addi %a, %b : i32
      }
      aie.end
    }
  }
}
