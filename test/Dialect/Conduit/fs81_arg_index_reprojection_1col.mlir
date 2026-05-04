// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Regression test for Bug A (Matrix Row #1 1col_small numerical wrong-output,
// task #77 / #81): `--conduit-fuse-operators` Step 8c trims dead block args
// from the merged `aie.runtime_sequence` but did NOT re-project the
// `arg_index` attribute on surviving `conduit.put_memref` /
// `conduit.get_memref` ops belonging to op B.
//
// `--conduit-to-dma` Step 8g consumes `arg_index` directly as
// `blockArgs[argIdx]` (`ConduitToDMALower.cpp:1126-1148`), so without the
// reprojection BDs bind to wrong block args.  In matrix Row #1's 1col_small
// the misbinding was routing-independent (1018/1024 wrong, identical signature
// across all 3 routings) because routing is correct — block-arg binding is
// wrong.
//
// Concrete reproducer here: Add (devA: in0, in1 → addOut) → Mul (devB:
// mulIn0=intermediate, mulIn1, mulOut).  Fusion erases addOut ↔ mulIn0.
//
// Pre-trim merged signature: [a0=in0, a1=in1, a2=addOut, b0=mulIn0,
//                              b1=mulIn1, b2=mulOut]
// Step 8c trims deadA={2} (addOut erased) + deadB={0} (mulIn0 erased).
// Post-trim signature:        [in0, in1, mulIn1, mulOut]   ← 4 args
//
// Surviving `conduit.put_memref` / `conduit.get_memref` ops carry stale
// `arg_index` attributes pointing to their pre-merge positions:
//   * A-side in0_BD:    arg_index = 0   (correct unshifted: 0)
//   * A-side in1_BD:    arg_index = 1   (correct unshifted: 1)
//   * B-side mulIn1_BD: arg_index = 1   ← MUST become 2 after reprojection
//   * B-side mulOut_BD: arg_index = 2   ← MUST become 3 after reprojection
//
// Without the reprojection, mulIn1's BD binds to %arg1 (in1), and mulOut's
// BD binds to %arg2 (mulIn1) — silently routing the host-side mul output
// to the secondary input slot, leaving the real output buffer (%arg3)
// unwritten.

// CHECK-LABEL: module @fs81_arg_index_reprojection_1col

// Single surviving device after fusion:
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device(npu2) @devB

// Merged runtime_sequence has exactly 4 surviving block args (4 × bf16
// memrefs); the two fused-channel endpoints have been trimmed.  Note
// that the merged runtime_sequence emits anonymously — the @sequence
// symbol from the input devA/devB sequences is dropped during fusion.
// CHECK:       aie.runtime_sequence(
// CHECK-SAME:    memref<128xbf16>
// CHECK-SAME:    memref<128xbf16>
// CHECK-SAME:    memref<128xbf16>
// CHECK-SAME:    memref<128xbf16>

// Reprojected arg_index values must reference the post-trim block-arg
// positions: A-side untouched (0, 1), B-side shifted by (origCountA −
// |deadA|) = 2 and re-based across deadB.  All four expected values must
// appear; no surviving op may carry a stale pre-merge index that would
// collide with an A-side arg.  These DAG checks run within the merged
// runtime_sequence body (devA-side, before the host orchestrator).
//
// CHECK-DAG:   arg_index = 0
// CHECK-DAG:   arg_index = 1
// CHECK-DAG:   arg_index = 2
// CHECK-DAG:   arg_index = 3

// The discardable `_origin_device` provenance tag is internal-only and must
// not leak past --conduit-fuse-operators.
// CHECK-NOT:   _origin_device

// Host orchestrator side-check (catches symbol-resolution + arity
// reconciliation).  After fusion, the originally 6-arg concat (3 from
// devA + 3 from devB) is reconciled by the reactive arity reconciliation
// (commit e4e3e45d63) to drop the two fused-channel intermediates
// (`%h_intA` at A-pos 2, `%h_intB` at B-pos 0).  Surviving args from
// the host vector = positions 0, 1, 4, 5.  confB folded into confA.
//
// CHECK:       aiex.configure @devA
// CHECK-NOT:   aiex.configure @devB
// CHECK:       aiex.run @sequence(%{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}}, %{{[^)]+}}) : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>, memref<128xbf16>)
// CHECK-NOT:   aiex.run @sequence(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}})

module @fs81_arg_index_reprojection_1col {
  // DevA: Add — in0 + in1 → addOut(intermediate, fusible).
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @in0(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @in1(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @addOut(%tile_0_2, {%shim_0}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @add_kernel(memref<128xbf16>, memref<128xbf16>,
                                   memref<128xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @in0(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %a_buf = aie.objectfifo.subview.access %a[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %b = aie.objectfifo.acquire @in1(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %b_buf = aie.objectfifo.subview.access %b[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %o = aie.objectfifo.acquire @addOut(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %o_buf = aie.objectfifo.subview.access %o[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @add_kernel(%a_buf, %b_buf, %o_buf)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @addOut(Produce, 1)
        aie.objectfifo.release @in1(Consume, 1)
        aie.objectfifo.release @in0(Consume, 1)
      }
      aie.end
    } {link_with = "add.a"}

    // 3-arg seqA: arg_index 0=in0, 1=in1, 2=addOut.
    aie.runtime_sequence @sequence(%a0: memref<128xbf16>,
                                    %a1: memref<128xbf16>,
                                    %a2: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%a0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%a1 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @addOut {
        aie.dma_bd(%a2 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
    }
  }

  // DevB: Mul — mulIn0(intermediate, fusible) * mulIn1 → mulOut.
  aie.device(npu2) @devB {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @mulIn0(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @mulIn1(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @mulOut(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @mul_kernel(memref<128xbf16>, memref<128xbf16>,
                                   memref<128xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @mulIn0(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %a_buf = aie.objectfifo.subview.access %a[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %b = aie.objectfifo.acquire @mulIn1(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %b_buf = aie.objectfifo.subview.access %b[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %o = aie.objectfifo.acquire @mulOut(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %o_buf = aie.objectfifo.subview.access %o[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @mul_kernel(%a_buf, %b_buf, %o_buf)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @mulOut(Produce, 1)
        aie.objectfifo.release @mulIn1(Consume, 1)
        aie.objectfifo.release @mulIn0(Consume, 1)
      }
      aie.end
    } {link_with = "mul.a"}

    // 3-arg seqB: arg_index 0=mulIn0, 1=mulIn1, 2=mulOut.
    aie.runtime_sequence @sequence(%b0: memref<128xbf16>,
                                    %b1: memref<128xbf16>,
                                    %b2: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @mulIn0 {
        aie.dma_bd(%b0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @mulIn1 {
        aie.dma_bd(%b1 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @mulOut {
        aie.dma_bd(%b2 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
    }
  }

  // Host orchestrator.  4 visible host-buffer args (in0, in1, mulIn1,
  // mulOut); the two fused-internal channels (addOut, mulIn0) ARE
  // staged through host-side intermediate buffers (h_intA, h_intB) at
  // pre-fusion positions A:2 and B:0 — these are exactly the deadA /
  // deadB indices that the reactive arity reconciliation must drop
  // when the merged callee trims them.  Distinct SSA values for the
  // intermediates (NOT aliased) — exercises the index-based reconcile
  // path, not identity-dedup.
  aie.device(npu2) {
    aie.runtime_sequence(%h_in0:    memref<128xbf16>,
                         %h_in1:    memref<128xbf16>,
                         %h_intA:   memref<128xbf16>,
                         %h_intB:   memref<128xbf16>,
                         %h_mulIn1: memref<128xbf16>,
                         %h_mulOut: memref<128xbf16>) {
      aiex.configure @devA {
        aiex.run @sequence(%h_in0, %h_in1, %h_intA)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>)
      }
      aiex.configure @devB {
        aiex.run @sequence(%h_intB, %h_mulIn1, %h_mulOut)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>)
      }
    }
  }
}
