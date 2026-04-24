// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Foundation Phase 2 (Task #19), gap #6 — fixed (Task #44): Pattern E
// (no-core-loop forward chain) sitting adjacent to a Pattern A op
// (compute core with scf.for + acquires) MUST NOT be fused via
// fusion_group or element_type matching.  Important for Llama KV cache:
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
// FIXED BEHAVIOR (Task #44, ConduitFuseOperators.cpp `isForwardChainEndpoint`):
//   --conduit-fuse-operators detects that @fwd_out is referenced by a
//   `conduit.scatter` op (Step 6's rename walk only updates ops with a
//   `name` FlatSymbolRefAttr — scatter src/dsts symbols would be left
//   dangling after a Step 5 erasure).  The match is rejected with a
//   remark; the element_type fallback applies the same guard so the
//   bug cannot resurface via the secondary matching path.  Result:
//     * Both devices remain intact and unfused.
//     * @fwd_in, @fwd_out, and the conduit.scatter survive in devA.
//     * @comp_in, @comp_out, and the consumer core in devB are untouched.
//
// Sibling: `--conduit-fuse-core-bodies` had an analogous Pattern E gap
// (independent code path in ConduitFuseCoreBodyPass.cpp Step 0); FIXED
// in commit c5a3df0435 via the shared `isForwardChainEndpoint` helper
// in DeviceMergeUtils.{h,cpp}. Pinned by
// pattern_e_no_fuse_core_bodies_with_neighbor.mlir.

// CHECK-LABEL: module @pattern_e_no_fuse_with_neighbor

// devA forward chain stays intact: both fifos + the scatter remain.
// CHECK:       aie.device(npu2) @devA
// CHECK-DAG:   conduit.create @fwd_in
// CHECK-DAG:   conduit.create @fwd_out
// CHECK:       conduit.scatter
// CHECK-SAME:  src = @fwd_in
// CHECK-SAME:  dsts = [@fwd_out]
// CHECK-NOT:   conduit.create @fused_intermediate

// devB stays as its own device with the original compute channels.
// CHECK:       aie.device(npu2) @devB
// CHECK-DAG:   conduit.create @comp_in
// CHECK-DAG:   conduit.create @comp_out
// CHECK:       aie.core
// CHECK:       conduit.acquire {{.*}} name = @comp_in
// CHECK-SAME:  port = #conduit.port<Consume>
// CHECK:       conduit.acquire {{.*}} name = @comp_out
// CHECK-SAME:  port = #conduit.port<Produce>

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

  // Module 2: a normal Pattern A op with the SAME fusion_group tag.
  // The Pattern E guard MUST prevent cross-device fusion here.
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
