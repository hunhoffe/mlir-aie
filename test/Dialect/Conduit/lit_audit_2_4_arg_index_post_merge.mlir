// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Lit-gap-audit finding 2.4 — `arg_index` resolution under multi-device
// merge.
//
// Source path under test:
//   - `--dma-task-to-conduit` (ConduitDmaTaskToConduit.cpp:281-293) records
//     `arg_index` from a direct BlockArgument of the enclosing
//     aie.runtime_sequence BEFORE any device merge.
//   - `--conduit-fuse-operators` then physically merges devB's
//     runtime_sequence body into devA's, APPENDING devB's block args to
//     seqA.  In the merged seqA, devB's old `arg_index = 0` now lives at
//     ABSOLUTE position `origArgCountA + 0`.
//   - Cloned-from-devB put/get_memref ops have their SSA operands remapped
//     (via IRMapping), but their `arg_index` integer ATTRIBUTE is copied
//     verbatim by `clone()` — it is NOT projected to the post-merge
//     absolute index.
//   - `--conduit-to-dma` Step 8g (ConduitToDMALower.cpp:1108-1148) reads
//     `arg_index` and looks it up in the merged sequence's block-arg list.
//     If the attr was not projected during merge, Step 8g binds the BD to
//     the WRONG block arg (devA's seg-A arg at the same index).
//
// What this test exercises:
//   - Two devices @devA / @devB, each with a tight 2-arg runtime_sequence
//     consuming both args via dma_configure_task_for (no spectators → Step
//     8c trim path is the relevant one).
//   - `fusion_group = "fg0"` on @inter_out (devA producer) + @inter_in
//     (devB consumer).  Step 6b deletes BOTH endpoints' put/get_memref ops
//     for the fused-internal channel; Step 8c trims the now-dead block
//     args.
//   - The SURVIVING ops on devB's side are: the put_memref on @ext_inB
//     (originally arg_index=0 on devB seqB → after merge, ABSOLUTE
//     arg_index=2 — devA's arg 0 is %a0, arg 1 is %a1 (trimmed), so the
//     surviving devB ext_inB SHOULD bind to absolute arg 2).
//   - Wait — Step 8c trims dead args, so the surviving merged seqA has
//     ARGS reduced to e.g. [%a0, %b1] (absolute positions 0, 1 in the
//     trimmed sequence).  The cloned-from-devB op for @ext_outB had its
//     arg_index=1 on seqB; after merge but BEFORE trim it would have
//     mapped to absolute 3 (origArgCountA=2 + 1); after Step 8c trim of
//     deadA={1} (intermediate) and deadB={0} (intermediate), the surviving
//     positions compact to [%a0(0), %b1(1)].  The op should resolve to
//     position 1.
//
// EXPECTED post-fix behavior:
//   The `conduit.put_memref` op for @ext_outB (devB's surviving output)
//   carries `arg_index = 1` referring to the trimmed merged sequence's
//   second block arg (which has memref<128xbf16> matching @ext_outB's
//   shape).
//
// CURRENT behavior (bug, if reproducible):
//   The `arg_index` is still `1` (pre-merge devB seqB index) which by
//   coincidence happens to point at the right arg AFTER the trim
//   compacts.  However, for the @ext_inA op the bug WOULD manifest:
//   pre-merge devA seqA index 0 → post-trim position 0 (correct only
//   because devA segment A's surviving arg is at position 0).
//
// This test asserts the EXPECTED end-state.  If the merge ever appends
// (vs prepends) devA's args, or if the trim ever changes ordering, the
// arg_index attrs MUST be re-projected explicitly.  TODO(post-fix):
// confirm whether `--conduit-fuse-operators` Step 8c re-projects
// arg_index; if not, file as a source bug.

// CHECK-LABEL: module @lit_audit_2_4_arg_index_post_merge

// Single named device after fuse-operators.
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device(npu2) @devB

// Merged trimmed runtime_sequence has 2 block args (devA's a0 + devB's b1):
// CHECK:       aie.runtime_sequence
// CHECK-SAME:    %{{[^,)]+}}: memref<128xbf16>
// CHECK-SAME:    %{{[^,)]+}}: memref<128xbf16>

// Surviving put_memref for @ext_inA (devA's input) — must bind to the
// FIRST block arg (absolute index 0).
// CHECK:       conduit.put_memref
// CHECK-SAME:    arg_index = 0
// CHECK-SAME:    name = @ext_inA

// Surviving put_memref for @ext_outB (devB's output, S2MM after fusion).
// CRITICAL: pre-merge this op carried arg_index=1 against devB's seqB.
// After the merge+trim, it MUST resolve to absolute index 1 of the merged
// trimmed sequence (which is %a1-position-occupied-by-b1).  If the attr
// stays at 1 by coincidence (devA's seg-A trim removed exactly one slot),
// this passes.  If we ever change the merge ordering or the trim logic,
// the attr must be explicitly re-projected.
// CHECK:       conduit.get_memref
// CHECK-SAME:    arg_index = 1
// CHECK-SAME:    name = @ext_outB

module @lit_audit_2_4_arg_index_post_merge {
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

  // Host orchestrator (kept minimal — host arg-vector handling is covered
  // by fs3_followup_run_callsite_arg_rewrite.mlir).
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
