// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Regression test for FS3 follow-up (task #63):
// `--conduit-fuse-operators` Step 8c trims block args for fused-internal
// channels from the merged `aie.runtime_sequence`.  After the trim, the
// host-side `aiex.run` callsite arg vector (built by Phase 1 of the
// host-orchestrator rewrite as a naive `runA.getArgs() ++ runB.getArgs()`
// concat) carries TOO MANY args.  The Phase 2 helper
// `reconcileHostRunArgsAfterTrim` must project the same drops Step 8c
// applied to the callee onto the run-op arg vector.
//
// Audit context (lit-gap-audit findings 1.1, 1.3, 3.1):
//   - 1.1: an SSA-identity dedup at fold time would only fire when the host
//     aliases the SAME buffer into both confA and confB.  Matrix Row #1's
//     actual harness wires DISTINCT host buffers into the two endpoints —
//     so the bug must be reproduced with distinct buffers.
//   - 1.3: prior fold-time validation ran BEFORE Step 8c, against the
//     pre-trim concat arity, so it could not catch the post-trim mismatch.
//   - 3.1: existing lit cases for the host-orchestrator rewrite use
//     spectator block-args that short-circuit Step 8c
//     (`groups.size() != numOrigArgs`); Step 8c never fires there, so they
//     do NOT exercise the Phase 2 reconcile.  This test is the first to
//     hit the trim path.
//
// What this test exercises:
//   - Two devices @devA / @devB, each with a TIGHT 2-arg runtime_sequence
//     (every block arg has a put/get_memref consumer — no spectators).
//     This makes Step 8c's `groups.size() == numOrigArgs` precondition hold,
//     so the dead-arg trim fires.
//   - `fusion_group = "fg0"` on @inter_out (devA producer) + @inter_in
//     (devB consumer) — Step 8c trims BOTH endpoints of this pair, leaving
//     a merged callee with 2 args (devA's a0, devB's b1).
//   - Host orchestrator with TWO configures and DISTINCT buffers wired into
//     each (`%h_intA` for confA's intermediate output, `%h_intB` for confB's
//     intermediate input — NOT aliased).  After the Phase 1 fold, the run's
//     concat arg vector is `[%h0, %h_intA, %h_intB, %h2]` — 4 args against
//     the now-trimmed 2-arg callee.  Phase 2 must drop position 1 (deadA)
//     and position 0 of segment B (deadB), leaving `[%h0, %h2]`.
//
// FileCheck assertions:
//   1. Single named device remains (devB merged into devA).
//   2. Host orchestrator has exactly one `aiex.configure @devA` and zero
//      `aiex.configure @devB`.
//   3. The merged `aiex.run @sequence(...)` has exactly 2 args (matching
//      the merged callee's surviving block-arg arity).
//   4. The 2 surviving args are `%h0` and `%h2` (NOT `%h_intA`, NOT
//      `%h_intB`).

// CHECK-LABEL: module @fs3_followup_run_callsite_arg_rewrite

// Surviving named device — only one remains:
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device(npu2) @devB

// Host orchestrator. The host's runtime_sequence has 4 args (H0, HINTA,
// HINTB, H2 — pre-merge concat); after Step 8c trims the merged callee
// to 2 args, reconcileHostRunArgsAfterTrim must drop deadA (segment A
// pos 1) + deadB (segment B pos 0) from the run's arg vector.
//
// Exactly one fold-into-confA, no surviving @devB configure:
// CHECK:       aiex.configure @devA
// CHECK-NOT:   aiex.configure @devB
//
// The reconciled `aiex.run` keeps positions 0 and 3 only (2 args, not 4).
// `%h_intA` (pos 1) and `%h_intB` (pos 2) are dropped — even though they
// are DISTINCT SSA values (reconciliation is index-based, NOT identity-
// based).
// CHECK:       aiex.run @sequence(%{{[^,]+}}, %{{[^)]+}}) : (memref<128xbf16>, memref<128xbf16>)
// CHECK-NOT:   aiex.run @sequence(%{{.+}}, %{{.+}}, %{{.+}})

module @fs3_followup_run_callsite_arg_rewrite {
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @ext_inA(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    // Producer's intermediate output — fusion_group "fg0" matches devB.
    aie.objectfifo @inter_out(%tile_0_2, {%shim_0}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @producer_kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_inA(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_out(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @producer_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @inter_out(Produce, 1)
        aie.objectfifo.release @ext_inA(Consume, 1)
      }
      aie.end
    } {link_with = "producer.a"}

    // Tight 2-arg runtime_sequence: BOTH block args are consumed by
    // put_memref ops after --dma-task-to-conduit.  No spectators →
    // `groups.size() == numOrigArgs` holds → Step 8c trim fires for the
    // fused-internal-channel arg.
    aie.runtime_sequence(%a0: memref<128xbf16>, %a1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_inA {
        aie.dma_bd(%a0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_out {
        aie.dma_bd(%a1 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  aie.device(npu2) @devB {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Consumer's intermediate input — fusion_group "fg0" matches devA.
    aie.objectfifo @inter_in(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

    aie.objectfifo @ext_outB(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @consumer_kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @inter_in(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @ext_outB(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @consumer_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_outB(Produce, 1)
        aie.objectfifo.release @inter_in(Consume, 1)
      }
      aie.end
    } {link_with = "consumer.a"}

    aie.runtime_sequence(%b0: memref<128xbf16>, %b1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @inter_in {
        aie.dma_bd(%b0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_outB {
        aie.dma_bd(%b1 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  // Host orchestrator.  CRITICAL: `%h_intA` and `%h_intB` are DISTINCT
  // SSA values — this reproduces matrix Row #1's actual harness shape and
  // exercises the index-based Phase 2 reconcile (NOT identity dedup).
  aie.device(npu2) {
    aie.runtime_sequence(%h0: memref<128xbf16>,
                         %h_intA: memref<128xbf16>,
                         %h_intB: memref<128xbf16>,
                         %h2: memref<128xbf16>) {
      aiex.configure @devA {
        aiex.run @sequence(%h0, %h_intA)
            : (memref<128xbf16>, memref<128xbf16>)
      }
      aiex.configure @devB {
        aiex.run @sequence(%h_intB, %h2)
            : (memref<128xbf16>, memref<128xbf16>)
      }
    }
  }
}
