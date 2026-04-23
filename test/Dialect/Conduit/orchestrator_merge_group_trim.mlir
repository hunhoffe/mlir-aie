// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators --split-input-file %s | FileCheck %s
//
// Regression test for Task #125: orchestrator dead-arg trim must use
// merge-group provenance, NOT post-hoc body liveness.
//
// Background: the original Task #99 trim used post-hoc liveness analysis on
// the orchestrator body (any block-arg referenced by no op was erased).
// That broke the host ABI for callers whose convention reserves block-arg
// slots even when those slots are unreferenced inside the body — most
// notably IRON's `FusedFullELFCallable` which always passes 3 parent
// buffers (input/output/scratch) via positional `set_arg(i, bo)` calls,
// regardless of whether the kernel body references all 3.  Symptom: the
// runtime continued to call `set_arg` against now-nonexistent kernel slots
// and the output buffer was never written (all-zero outputs, no exception
// — silent corruption).
//
// Fix: trim only args whose orchestrator-body aiex.run callsite references
// went from N>0 to 0 BECAUSE OF this merge invocation's run-callsite
// rewrite (Phase 2).  Args dead-from-start (prevUsage==0) survive.
//
// Two cases below cover the survival paths.

// -----

// Case A — ABI-reserved slot dead from start: an orchestrator block-arg that
// no aiex.run callsite ever references must SURVIVE the trim, even after
// fusion-induced merge-group changes elsewhere.
//
// Setup: tight 2-arg seqA/seqB (Step 8c trims fused-internal-channel arg of
// each), so the post-fusion merged callee is 2 args and the merged aiex.run
// gets reconciled to (h0, h2).  The orchestrator carries a third arg
// `%h_reserved` that NO aiex.run callsite ever references (simulating the
// FusedFullELFCallable scratch parent-buffer slot).  Pre-fix it would have
// been silently trimmed by the body-liveness rule.  Post-fix it survives.

// CHECK-LABEL: module @case_a_abi_reserved_arg_survives

// Single surviving named device:
// CHECK:       aie.device(npu2) @devA

// The HOST orchestrator must keep THREE args: the two intermediate args
// (h_intA, h_intB) ARE referenced by aiex.run callsites pre-fusion (and
// become merge-group-dead post-fusion → trimmed).  Plus h0, h_reserved, h2
// stay.  Net post-fix: 3 surviving args (h0, h_reserved, h2).
//
// The third regex `[^,)]+: memref<128xbf16>` forbids commas so the pattern
// cannot greedily swallow additional args.
//
// CHECK:       aie.device(npu2) {
// CHECK-NEXT:    aie.runtime_sequence(%{{[^,]+}}: memref<128xbf16>, %{{[^,]+}}: memref<128xbf16>, %{{[^,)]+}}: memref<128xbf16>) {

// CHECK:       aiex.run @sequence(%{{[^,]+}}, %{{[^)]+}}) : (memref<128xbf16>, memref<128xbf16>)

module @case_a_abi_reserved_arg_survives {
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @ext_inA(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
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

  // Host orchestrator: 5 host buffer args.  Two intermediates (h_intA,
  // h_intB) are referenced by per-device aiex.run callsites pre-fusion
  // and become merge-group-dead post-fusion → trimmed.  %h_reserved is
  // NEVER referenced by ANY aiex.run callsite (simulates an
  // FusedFullELFCallable-style ABI-reserved slot like the scratch parent
  // buffer) → must SURVIVE the new merge-group-provenance trim.
  aie.device(npu2) {
    aie.runtime_sequence(%h0:        memref<128xbf16>,
                         %h_intA:    memref<128xbf16>,
                         %h_reserved: memref<128xbf16>,
                         %h_intB:    memref<128xbf16>,
                         %h2:        memref<128xbf16>) {
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

// -----

// Case B — survives because aiex.run still references it: the standard live
// case.  No surprise here, just regression coverage that surviving aiex.run
// arg references protect their orchestrator slots.

// CHECK-LABEL: module @case_b_aiex_run_referenced_arg_survives

// CHECK:       aie.device(npu2) @devA

// Same pattern as the merge-introduced-dead test: 4 host args pre-fusion;
// h_intA + h_intB get trimmed (merge-group dead); h0 and h2 survive (still
// referenced by the merged aiex.run).  Net = 2 surviving args.
//
// CHECK:       aie.device(npu2) {
// CHECK-NEXT:    aie.runtime_sequence(%{{[^,]+}}: memref<128xbf16>, %{{[^,)]+}}: memref<128xbf16>) {
// CHECK:         aiex.run @sequence(%{{[^,]+}}, %{{[^)]+}}) : (memref<128xbf16>, memref<128xbf16>)

module @case_b_aiex_run_referenced_arg_survives {
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @ext_inA(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @inter_out(%tile_0_2, {%shim_0}, 2 : i32)
        {fusion_group = "fg1"}
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

    aie.objectfifo @inter_in(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg1"}
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

  aie.device(npu2) {
    aie.runtime_sequence(%h0:     memref<128xbf16>,
                         %h_intA: memref<128xbf16>,
                         %h_intB: memref<128xbf16>,
                         %h2:     memref<128xbf16>) {
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
