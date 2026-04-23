// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Regression test for FS3: after spatial fusion merges devB into devA, the
// host orchestrator (third aie.device whose runtime_sequence contains
// `aiex.configure @devA { ... }` and `aiex.configure @devB { ... }` blocks)
// must remain valid.  Without the post-merge host-orchestrator rewrite the
// `aiex.configure @devB` block is left dangling and the verifier emits
// "No such device: '@devB'".
//
// After --conduit-fuse-operators we expect:
//   1. devB is gone (single named device remains).
//   2. The orchestrator now contains a SINGLE `aiex.configure @devA` block.
//   3. Its inner `aiex.run @sequence(...)` takes the CONCATENATION of the
//      two original arg lists (3 + 3 = 6 memref args) — collapsing the two
//      LoadPDI cycles into one.

// CHECK-LABEL: module @fuse_operators_host_orchestrator

// Surviving named device — only one remains:
// CHECK:       aie.device(npu2) @devA

// Host orchestrator must reference devA and NOT devB:
// CHECK:       aiex.configure @devA
// CHECK-NOT:   aiex.configure @devB

// The merged run must take the concatenation of A's + B's args (6 total):
// CHECK:       aiex.run @sequence(
// CHECK-SAME:    %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}},
// CHECK-SAME:    %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}})
// CHECK-SAME:    : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>,
// CHECK-SAME:       memref<128xbf16>, memref<128xbf16>, memref<128xbf16>)

module @fuse_operators_host_orchestrator {
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

    aie.runtime_sequence(%a0: memref<128xbf16>, %a1: memref<128xbf16>, %a2: memref<128xbf16>) {
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

    aie.runtime_sequence(%b0: memref<128xbf16>, %b1: memref<128xbf16>, %b2: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @inter_in {
        aie.dma_bd(%b0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_outB {
        aie.dma_bd(%b2 : memref<128xbf16>, 0, 128,
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

  // Host orchestrator: anonymous aie.device whose runtime_sequence contains
  // aiex.configure blocks for each named device. This is the FS3 trigger —
  // when @devB is erased by the merge, this configure becomes dangling
  // unless our rewrite folds it into @devA.
  aie.device(npu2) {
    aie.runtime_sequence(%h0: memref<128xbf16>,
                         %h1: memref<128xbf16>,
                         %h2: memref<128xbf16>,
                         %h3: memref<128xbf16>,
                         %h4: memref<128xbf16>,
                         %h5: memref<128xbf16>) {
      aiex.configure @devA {
        aiex.run @sequence(%h0, %h1, %h2)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>)
      }
      aiex.configure @devB {
        aiex.run @sequence(%h3, %h4, %h5)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>)
      }
    }
  }
}
