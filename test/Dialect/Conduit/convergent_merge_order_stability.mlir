// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Task #82 lit pin — K=2 convergent merge runtime_sequence arg ORDER
// stability under `--conduit-fuse-operators`.
//
// Bug shape:
//   K=2 convergent merge produces a merged consumer-device runtime_sequence
//   whose block-arg order is determined by the iterative pairwise merge order
//   inside the pass (which device gets absorbed first, etc.), NOT by the
//   source-IR device declaration order.  With symmetric inputs (gate == up)
//   the swap is invisible; with non-symmetric inputs the gate-side and
//   up-side host buffers are bound to each other's arg slots at runtime,
//   corrupting downstream consumers.
//
//   In this fixture devGate is declared FIRST in source IR and uses a
//   memref<64xbf16> external input; devUp is declared SECOND and uses a
//   memref<128xbf16> external input; devMul (the convergent consumer) is
//   declared THIRD and uses a memref<32xbf16> external output.  After
//   `--conduit-fuse-operators` the merged consumer device's runtime_sequence
//   currently emits the up-side put_memref BEFORE the gate-side put_memref,
//   inverting the source-IR producer-declaration order.  Pre-fix this fixture
//   FAILs (CHECK lines below encode source-IR-declaration order = correct);
//   post-fix it flips to PASS.
//
// Pin convention: pin the CORRECT behavior (FAIL today, PASS after fix).
//   Same convention as conduit_fused_runtime_sequence_dead_arg_prune.mlir
//   case (a)/(b) (Task #81 sister) — CHECK lines encode the post-fix shape;
//   pre-fix the test fails on a CHECK miss.
//
// Why this is the right CHECK behavior (first principles):
//   The merged consumer device's `aie.runtime_sequence` is the entry point
//   that XRT (or the host orchestrator) binds buffers into by ordinal arg
//   position.  In `test/npu-xrt/fuse_hybrid_swiglu_npu/test.cpp` and any
//   other no-host-orchestrator NPU harness, the host code hard-codes BO
//   indices to the source-IR device declaration order (e.g., bo[0] = gate,
//   bo[1] = up, bo[2] = out).  Source-IR declaration order is the only
//   stable contract the host can rely on.  If the merge swaps gate and up
//   in the merged arg list, `bo[0]` (gate data) silently DMAs into the
//   `@ext_in_up` channel and `bo[1]` (up data) DMAs into `@ext_in_gate`.
//   The fix MUST emit producer-side put_memref ops in source-IR
//   producer-declaration order so that arg_index N of the merged sequence
//   binds the same shim_alloc the host expected at host arg N.
//
//   Distinct producer-side INPUT shapes (gate = memref<64xbf16>,
//   up = memref<128xbf16>) make the swap directly visible at the IR type
//   level: the merged arg-list types must come out in `(memref<64>,
//   memref<128>, memref<32>)` order to match source-IR declaration order.
//   Pre-fix the order is `(memref<128>, memref<64>, memref<32>)` — visible
//   as a type swap on the first two args, not just a name swap.
//
//   The intermediate channels (`@inter_gate`, `@inter_up`,
//   `@consume_gate`, `@consume_up`) all share the same `memref<32xbf16>`
//   element type per the locked Track 3 design (cross-element-type fan-in
//   scoped out for initial landing, CLAUDE.md USER-LOCKED 2026-04-26) —
//   only the producer-side EXTERNAL inputs need to be distinguishable for
//   the order-stability pin to surface the bug.
//
// Sibling fixtures matched (sibling-pattern reading req):
//   * `conduit_fused_runtime_sequence_dead_arg_prune.mlir` (Task #81 sister):
//     pin convention (CHECK encodes post-fix shape; pre-fix FAIL → post-fix
//     PASS).  Confirms `arg_index` is the authoritative binding contract
//     between conduit ops and runtime_sequence block args.
//   * `fuse_operators_convergent_basic.mlir`: the existing K=2 convergent
//     fixture.  Symmetric inputs (gate and up both `memref<128xbf16>`)
//     hide the order swap from CHECK lines but my fix-prep probe of this
//     same fixture confirms the merged arg order today is up-first
//     (`arg_index = 0 → @ext_in_up`, `= 1 → @ext_in_gate`) regardless of
//     gate-first source-IR declaration.  This pin makes that swap visible
//     by changing input shapes only.
//
// Cross-reference (fixture-author-acknowledged hiding mechanism):
//   `test/npu-xrt/fuse_hybrid_swiglu_npu/aie.mlir` lines 87-91 explicitly
//   note that "Symmetric inputs make verification independent of post-
//   fusion arg-reprojection order on the PRODUCER side (gate vs up arg
//   slots interchangeable)" — i.e., the swiglu HW smoke deliberately uses
//   symmetric data because the order is not stable.  This pin promotes
//   that latent-knowledge into a regression-locked CHECK.

// CHECK-LABEL: module @convergent_merge_order_stability

// Only ONE merged device should remain after K=2 convergent fusion:
// CHECK:       aie.device(npu2)
// CHECK-NOT:   aie.device(npu2)

// Two distinct fused intermediates (one per producer) survive:
// CHECK-DAG:   conduit.create @fused_intermediate_0
// CHECK-DAG:   conduit.create @fused_intermediate_1

// The merged consumer device's runtime_sequence MUST emit block args in
// source-IR producer-declaration order: gate first (memref<64xbf16>), up
// second (memref<128xbf16>), output third (memref<32xbf16>).  Pre-fix the
// observed order is (memref<128>, memref<64>, memref<32>) — gate and up
// swapped.
// CHECK:       aie.runtime_sequence
// CHECK-SAME:    %{{[^,)]+}}: memref<64xbf16>,
// CHECK-SAME:    %{{[^,)]+}}: memref<128xbf16>,
// CHECK-SAME:    %{{[^,)]+}}: memref<32xbf16>

// Gate's external input must bind to arg_index = 0 (the first source-IR-
// declared producer's external input):
// CHECK:       conduit.put_memref{{(_async)?}}
// CHECK-SAME:    arg_index = 0
// CHECK-SAME:    name = @ext_in_gate

// Up's external input must bind to arg_index = 1 (the second source-IR-
// declared producer's external input):
// CHECK:       conduit.put_memref{{(_async)?}}
// CHECK-SAME:    arg_index = 1
// CHECK-SAME:    name = @ext_in_up

// The mul output must bind to arg_index = 2:
// CHECK:       conduit.get_memref{{(_async)?}}
// CHECK-SAME:    arg_index = 2
// CHECK-SAME:    name = @ext_out_mul

module @convergent_merge_order_stability {
  // Producer 1 (declared FIRST in source IR): gate.
  // External input is memref<64xbf16> — distinguishable from up's input.
  aie.device(npu2) @devGate {
    %shim   = aie.tile(0, 0)
    %tile_g = aie.tile(0, 2)

    aie.objectfifo @ext_in_gate(%shim, {%tile_g}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @inter_gate(%tile_g, {%shim}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<32xbf16>>

    func.func private @gate_kernel(memref<64xbf16>, memref<32xbf16>)

    %core_g = aie.core(%tile_g) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_gate(Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        %out = aie.objectfifo.acquire @inter_gate(Produce, 1)
            : !aie.objectfifosubview<memref<32xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
        func.call @gate_kernel(%in_buf, %out_buf)
            : (memref<64xbf16>, memref<32xbf16>) -> ()
        aie.objectfifo.release @inter_gate(Produce, 1)
        aie.objectfifo.release @ext_in_gate(Consume, 1)
      }
      aie.end
    } {link_with = "gate.a"}

    aie.runtime_sequence(%a0: memref<64xbf16>, %a1: memref<32xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_gate {
        aie.dma_bd(%a0 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_gate {
        aie.dma_bd(%a1 : memref<32xbf16>, 0, 32,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 32, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  // Producer 2 (declared SECOND in source IR): up.
  // External input is memref<128xbf16> — distinguishable from gate's input.
  aie.device(npu2) @devUp {
    %shim   = aie.tile(0, 0)
    %tile_u = aie.tile(0, 3)

    aie.objectfifo @ext_in_up(%shim, {%tile_u}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    aie.objectfifo @inter_up(%tile_u, {%shim}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<32xbf16>>

    func.func private @up_kernel(memref<128xbf16>, memref<32xbf16>)

    %core_u = aie.core(%tile_u) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_up(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_up(Produce, 1)
            : !aie.objectfifosubview<memref<32xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
        func.call @up_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<32xbf16>) -> ()
        aie.objectfifo.release @inter_up(Produce, 1)
        aie.objectfifo.release @ext_in_up(Consume, 1)
      }
      aie.end
    } {link_with = "up.a"}

    aie.runtime_sequence(%b0: memref<128xbf16>, %b1: memref<32xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_up {
        aie.dma_bd(%b0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_up {
        aie.dma_bd(%b1 : memref<32xbf16>, 0, 32,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 32, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  // Convergent consumer (declared THIRD in source IR): mul.
  aie.device(npu2) @devMul {
    %shim   = aie.tile(0, 0)
    %tile_c = aie.tile(0, 4)

    aie.objectfifo @consume_gate(%shim, {%tile_c}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<32xbf16>>

    aie.objectfifo @consume_up(%shim, {%tile_c}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<32xbf16>>

    aie.objectfifo @ext_out_mul(%tile_c, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<32xbf16>>

    func.func private @mul_kernel(memref<32xbf16>, memref<32xbf16>, memref<32xbf16>)

    %core_c = aie.core(%tile_c) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in_g = aie.objectfifo.acquire @consume_gate(Consume, 1)
            : !aie.objectfifosubview<memref<32xbf16>>
        %in_g_buf = aie.objectfifo.subview.access %in_g[0]
            : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
        %in_u = aie.objectfifo.acquire @consume_up(Consume, 1)
            : !aie.objectfifosubview<memref<32xbf16>>
        %in_u_buf = aie.objectfifo.subview.access %in_u[0]
            : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
        %out = aie.objectfifo.acquire @ext_out_mul(Produce, 1)
            : !aie.objectfifosubview<memref<32xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
        func.call @mul_kernel(%in_g_buf, %in_u_buf, %out_buf)
            : (memref<32xbf16>, memref<32xbf16>, memref<32xbf16>) -> ()
        aie.objectfifo.release @ext_out_mul(Produce, 1)
        aie.objectfifo.release @consume_up(Consume, 1)
        aie.objectfifo.release @consume_gate(Consume, 1)
      }
      aie.end
    } {link_with = "mul.a"}

    aie.runtime_sequence(%c0: memref<32xbf16>, %c1: memref<32xbf16>, %c2: memref<32xbf16>) {
      %t0 = aiex.dma_configure_task_for @consume_gate {
        aie.dma_bd(%c0 : memref<32xbf16>, 0, 32,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 32, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @consume_up {
        aie.dma_bd(%c1 : memref<32xbf16>, 0, 32,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 32, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @ext_out_mul {
        aie.dma_bd(%c2 : memref<32xbf16>, 0, 32,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 32, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }
}
