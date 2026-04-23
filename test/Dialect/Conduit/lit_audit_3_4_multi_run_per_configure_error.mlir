// RUN: not aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s 2>&1 | FileCheck %s
//
// Lit-gap-audit finding 3.4 — multi-RunOp-per-Configure error path at
// `DeviceMergeUtils.cpp:243-249`.
//
// Source path under test:
//   When `rewriteHostConfigureOnDeviceMerge` reaches the FOLD branch
//   (sibling `aiex.configure @<devA>` precedes confB in the same parent
//   block), it checks both confA and confB for >1 `aiex.run` ops.  More
//   than one run in either configure is a violation of the IRON convention
//   (exactly one aiex.run per configure), and the helper emits a hard
//   error rather than risking silent mis-fold:
//
//     "device-merge: cannot fold aiex.configure with multiple aiex.run
//      ops; expected at most one aiex.run per configure (IRON convention)"
//
// What this test exercises:
//   - Two named devices @devA / @devB with a fusion_group connection so
//     `--conduit-fuse-operators` triggers a merge.
//   - Host orchestrator: confA (single run) precedes confB (TWO runs).
//     The fold path is taken; confB has >1 RunOps → error.
//
// Diagnostic substring is matched verbatim per the helper at
// `DeviceMergeUtils.cpp:245-248`.

// CHECK: device-merge: cannot fold aiex.configure with multiple aiex.run ops

module @lit_audit_3_4_multi_run_per_configure_error {
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
        aie.dma_bd(%a0 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_out {
        aie.dma_bd(%a1 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
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
        aie.dma_bd(%b0 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_outB {
        aie.dma_bd(%b1 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  // Host orchestrator: confA (single run) precedes confB (TWO runs).
  // The fold path is taken (confA precedes confB in the same parent block).
  // The helper rejects confB because it carries >1 aiex.run.
  aie.device(npu2) {
    aie.runtime_sequence(%h0: memref<128xbf16>,
                         %h_intA: memref<128xbf16>,
                         %h_intB1: memref<128xbf16>,
                         %h2a: memref<128xbf16>,
                         %h_intB2: memref<128xbf16>,
                         %h2b: memref<128xbf16>) {
      aiex.configure @devA {
        aiex.run @sequence(%h0, %h_intA)
            : (memref<128xbf16>, memref<128xbf16>)
      }
      aiex.configure @devB {
        aiex.run @sequence(%h_intB1, %h2a)
            : (memref<128xbf16>, memref<128xbf16>)
        aiex.run @sequence(%h_intB2, %h2b)
            : (memref<128xbf16>, memref<128xbf16>)
      }
    }
  }
}
