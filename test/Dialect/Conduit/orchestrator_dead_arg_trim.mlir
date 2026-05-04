// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Regression test for Task #99: orchestrator dead-arg cleanup post-fusion.
//
// Background: Step 8c in --conduit-fuse-operators trims dead block args from
// the merged DEVICE-side runtime_sequence (the fused-internal-channel
// endpoints).  Step 8d (`reconcileHostRunArgsAfterTrim`) then projects the
// same drops into the host-side `aiex.run` callsite arg vectors.  However,
// prior to this fix nothing trimmed the OUTER host-orchestrator
// `aie.runtime_sequence` itself: its block-arg signature retained the
// pre-fusion args even after the inner aiex.run no longer referenced the
// fused-intermediate slots.
//
// Effect (passing-cells-static-verify Task #95): every matrix Row #1 cell's
// orchestrator runtime_sequence retained a dead memref block arg per fused
// intermediate.  Per invocation this wastes 1024×2 = 2 KB (1col) or
// 8192×2 = 16 KB (4col) of L3 — host code allocates a buffer no kernel ever
// reads or writes.  Not a correctness bug; pure waste.
//
// Fix: after Step 8d's run-callsite trim, walk the orchestrator body and
// erase any block args that no remaining op references.
//
// Reproducer here mirrors the FS3-followup test shape: tight 2-arg sequences
// per device (no spectators → Step 8c trim fires) with `fusion_group = "fg0"`
// on the producer's intermediate output and the consumer's intermediate
// input.  Pre-trim merged callee = 4 args (a0, a1, b0, b1); deadA={1},
// deadB={0}; post-trim callee = 2 args.
//
// Host orchestrator pre-fix:
//   aie.runtime_sequence(%h0, %h_intA, %h_intB, %h2)   ← 4 args
//     aiex.run @sequence(%h0, %h2)                     ← post-Step-8d, 2 args
//
// Host orchestrator post-fix:
//   aie.runtime_sequence(%h0, %h2)                     ← 2 args (intermediates trimmed)
//     aiex.run @sequence(%h0, %h2)                     ← unchanged

// CHECK-LABEL: module @orchestrator_dead_arg_trim

// Single surviving named device (devB merged into devA):
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device(npu2) @devB

// The HOST orchestrator's runtime_sequence (inside the anonymous outer
// `aie.device(npu2)`) must have only the live block args remaining: the two
// fused-internal-channel intermediate host buffer args are trimmed.  After
// the fix, the orchestrator carries exactly 2 args; pre-fix it carried 4.
//
// Pattern explanation: `%{{[^,]+}}: memref<128xbf16>, %{{[^,)]+}}: memref<128xbf16>) {`
// requires exactly two `arg: memref<128xbf16>` entries separated by exactly
// one `, ` and terminated by `) {`.  The second arg-name regex `[^,)]+`
// FORBIDS commas — so it cannot greedily swallow additional args.  A
// regression to >2 args would not match this pattern.
//
// CHECK:       aie.device(npu2) {
// CHECK-NEXT:    aie.runtime_sequence(%{{[^,]+}}: memref<128xbf16>, %{{[^,)]+}}: memref<128xbf16>) {

// Single fold-into-confA inside the orchestrator, no surviving @devB
// configure, and the inner run keeps exactly the two host args (matching the
// trimmed callee).
// CHECK-NEXT:      aiex.configure @devA
// CHECK-NOT:       aiex.configure @devB
// CHECK:           aiex.run @sequence(%{{[^,]+}}, %{{[^)]+}}) : (memref<128xbf16>, memref<128xbf16>)

module @orchestrator_dead_arg_trim {
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

    // Tight 2-arg sequence (no spectators) so Step 8c trim fires for the
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

  // Host orchestrator: 4 host buffer args.  After fusion + Step 8d, the
  // merged aiex.run keeps only %h0 and %h2; the new orchestrator-trim Phase 3
  // drops %h_intA and %h_intB from the runtime_sequence signature.
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
