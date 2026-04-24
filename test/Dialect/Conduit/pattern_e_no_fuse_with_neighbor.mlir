// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators --conduit-fuse-core-bodies %s | FileCheck %s
//
// Foundation Phase 2 (Task #19), gap #6 — negative test for Pattern E
// (no-core-loop forward chain) sitting adjacent to a Pattern A op
// (compute core with scf.for + acquires).  Important for Llama KV cache:
// StridedCopy/Repeat at num_invocations=512 is a Pattern E op that may
// land in a runlist next to ordinary compute ops.  Spatial /
// core-body fusion MUST NOT mis-fuse the forward chain into the
// neighbor op — the forward chain has no acquire site / no merge-able
// runtime body.
//
// Geometry:
//   * devA: pure forward chain  shim_0 → memtile_0 → shim_1 via
//     `aie.objectfifo.link` (Pattern E).  No aie.core, no acquires.
//     Both fifos carry fusion_group = "fg0" — this matches the IRON
//     `fusion_group` plumbing convention which a future StridedCopy
//     extension might apply.
//   * devB: Pattern A compute core with scf.for trip=8, also tagged
//     fusion_group = "fg0".
//
// CURRENT BEHAVIOR (the gap this test pins):
//   --conduit-fuse-operators DOES match fwd_in / fwd_out with comp_in
//   purely via the shared fusion_group tag, even though fwd_in /
//   fwd_out are forward-chain endpoints (no acquire/release in any
//   core).  Result:
//     * @fwd_out gets RENAMED to @fused_intermediate_0 via the
//       fusion_group rewrite (Step 5 of fuse-operators).
//     * @comp_in is erased on the consumer side; the consumer core
//       starts acquiring @fused_intermediate_0.
//     * The conduit.scatter from the original link is left referencing
//       the now-dangling symbol @fwd_out (its dst was NOT updated).
//   The resulting IR is structurally invalid in data flow (dangling
//   FlatSymbolRefAttr in the scatter dst), even though the IR
//   verifier currently does not enforce a SymbolTable lookup on
//   conduit.scatter dsts.
//
// This is a known fuse-operators bug (gap #6 in fusion-lit-gap-audit,
// Foundation Phase 1).  The proper fix would either (a) skip the
// fusion_group match when one side is a forward-chain endpoint with
// no compute-core acquires, or (b) error explicitly.  Until then this
// test is the regression net for the silent mis-fusion + dangling
// scatter dst.  See task list for the follow-up bug entry.

// CHECK-LABEL: module @pattern_e_no_fuse_with_neighbor

// devB has been merged into devA (single device after fuse-operators).
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device

// fwd_in survives but fwd_out has been renamed/erased — fused intermediate
// takes its place.
// CHECK:       conduit.create @fwd_in
// CHECK:       conduit.create @fused_intermediate_0
// CHECK-NOT:   conduit.create @fwd_out
// CHECK-NOT:   conduit.create @comp_in

// The scatter from the original link is left with a dangling dst symbol
// reference to @fwd_out (the rewrite did NOT propagate into the scatter).
// This is the broken state the test pins.
// CHECK:       conduit.scatter
// CHECK-SAME:  src = @fwd_in
// CHECK-SAME:  dsts = [@fwd_out]

// Consumer core now reads @fused_intermediate_0 (instead of @comp_in).
// CHECK:       aie.core
// CHECK:       conduit.acquire {{.*}} name = @fused_intermediate_0
// CHECK-SAME:  port = #conduit.port<Consume>

module @pattern_e_no_fuse_with_neighbor {
  // Module 1: pure forward chain (Pattern E), tagged fusion_group="fg0".
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %memtile_0 = aie.tile(0, 1)
    %shim_1 = aie.tile(1, 0)

    aie.objectfifo @fwd_in(%shim_0, {%memtile_0}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<8xbf16>>
    aie.objectfifo @fwd_out(%memtile_0, {%shim_1}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<8xbf16>>
    aie.objectfifo.link [@fwd_in] -> [@fwd_out]([] [])

    aie.runtime_sequence(%a0: memref<64xbf16>, %a1: memref<64xbf16>) {
      %t0 = aiex.dma_configure_task_for @fwd_in {
        aie.dma_bd(%a0 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @fwd_out {
        aie.dma_bd(%a1 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  // Module 2: a normal Pattern A op with the SAME fusion_group tag,
  // which (per current bug) will trigger spurious cross-device fusion.
  aie.device(npu2) @devB {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @comp_in(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<8xbf16>>
    aie.objectfifo @comp_out(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<8xbf16>>

    func.func private @kernel(memref<8xbf16>, memref<8xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %in = aie.objectfifo.acquire @comp_in(Consume, 1)
            : !aie.objectfifosubview<memref<8xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
        %out = aie.objectfifo.acquire @comp_out(Produce, 1)
            : !aie.objectfifosubview<memref<8xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
        func.call @kernel(%in_buf, %out_buf)
            : (memref<8xbf16>, memref<8xbf16>) -> ()
        aie.objectfifo.release @comp_out(Produce, 1)
        aie.objectfifo.release @comp_in(Consume, 1)
      }
      aie.end
    } {link_with = "kernel.a"}

    aie.runtime_sequence(%b0: memref<64xbf16>, %b1: memref<64xbf16>) {
      %t0 = aiex.dma_configure_task_for @comp_in {
        aie.dma_bd(%b0 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @comp_out {
        aie.dma_bd(%b1 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }
}
