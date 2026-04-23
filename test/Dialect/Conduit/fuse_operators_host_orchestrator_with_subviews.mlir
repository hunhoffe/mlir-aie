// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Regression test for FS3 follow-up: the host-orchestrator rewrite (used by
// --conduit-fuse-operators / --conduit-fuse-core-bodies / --aie-combine-device)
// must handle aiex.configure bodies that contain ops other than the single
// aiex.run — e.g. memref.subview / memref.reinterpret_cast that compute the
// arguments handed to aiex.run.  Real IRON-emitted host orchestrators (matrix
// Add->Mul harness, Llama runlist) always have these.
//
// The original FS3 fix had a use-after-detach bug in DeviceMergeUtils when
// moving these non-Run ops from confB into confA: it called op->remove()
// (detaches; block becomes null) before op->moveBefore(insertBefore), which
// asserts that the op is currently in a block.  That assertion fired only
// when toMove was non-empty, so the original lit test (configure bodies hold
// just an aiex.run) didn't catch it.  This test exercises the non-empty path.
//
// After --conduit-fuse-operators we expect:
//   1. Single named device remains (devB merged into devA).
//   2. Host orchestrator has one `aiex.configure @devA` block.
//   3. Both subview ops from the original two configures are present in the
//      merged configure body.
//   4. The merged aiex.run takes the concatenation of A's + B's args (6 total).

// CHECK-LABEL: module @fuse_operators_host_orchestrator_with_subviews

// Surviving named device — only one remains:
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device(npu2) @devB

// Host orchestrator must reference devA and NOT devB:
// CHECK:       aiex.configure @devA
// CHECK-NOT:   aiex.configure @devB

// Both subview chains survive in the merged configure:
// CHECK:       memref.subview
// CHECK:       memref.reinterpret_cast
// CHECK:       memref.subview
// CHECK:       memref.reinterpret_cast

// The merged run must take the concatenation of A's + B's args (6 total):
// CHECK:       aiex.run @sequence(
// CHECK-SAME:    %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}},
// CHECK-SAME:    %{{[^,]+}}, %{{[^,]+}}, %{{[^,]+}})

module @fuse_operators_host_orchestrator_with_subviews {
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

  // Host orchestrator: anonymous aie.device.  Each aiex.configure body
  // contains memref.subview + memref.reinterpret_cast ops that compute the
  // values handed to aiex.run — mirroring real IRON-emitted host code.  This
  // is the FS3-followup trigger: the helper must move these intermediate ops
  // from confB into confA correctly (without the use-after-detach bug).
  aie.device(npu2) {
    aie.runtime_sequence(%h_in: memref<384xbf16>,
                         %h_outA: memref<128xbf16>,
                         %h_outB: memref<128xbf16>) {
      aiex.configure @devA {
        %sv_a0 = memref.subview %h_in[0] [128] [1]
            : memref<384xbf16> to memref<128xbf16>
        %rc_a0 = memref.reinterpret_cast %sv_a0 to offset: [0],
            sizes: [128], strides: [1]
            : memref<128xbf16> to memref<128xbf16>
        %sv_a1 = memref.subview %h_in[128] [128] [1]
            : memref<384xbf16> to memref<128xbf16, strided<[1], offset: 128>>
        %rc_a1 = memref.reinterpret_cast %sv_a1 to offset: [0],
            sizes: [128], strides: [1]
            : memref<128xbf16, strided<[1], offset: 128>> to memref<128xbf16>
        aiex.run @sequence(%rc_a0, %rc_a1, %h_outA)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>)
      }
      aiex.configure @devB {
        %sv_b0 = memref.subview %h_in[256] [128] [1]
            : memref<384xbf16> to memref<128xbf16, strided<[1], offset: 256>>
        %rc_b0 = memref.reinterpret_cast %sv_b0 to offset: [0],
            sizes: [128], strides: [1]
            : memref<128xbf16, strided<[1], offset: 256>> to memref<128xbf16>
        aiex.run @sequence(%h_outA, %rc_b0, %h_outB)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>)
      }
    }
  }
}
