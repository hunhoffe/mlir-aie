// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-core-bodies %s | FileCheck %s
//
// Sibling of `pattern_e_no_fuse_with_neighbor.mlir` (Task #44, fuse-operators):
// `--conduit-fuse-core-bodies` has the same Pattern E mis-fusion gap in its
// Step 0 cross-device fusion_group matcher (`devicesConnectedByFusionGroup` +
// `mergeAndUnifyDevices` consumer→producer name unification only updates ops
// with a `name` FlatSymbolRefAttr — scatter/gather src/dsts symbols would be
// left dangling after the rename).
//
// Geometry mirrors the fuse-operators test:
//   * devA: pure forward chain shim_0 → memtile_0 → shim_1 via
//     `aie.objectfifo.link` (Pattern E).  Both fifos carry
//     fusion_group = "fg0".
//   * devB: Pattern A compute core with scf.for trip=8, also tagged
//     fusion_group = "fg0".
//
// FIXED BEHAVIOR (Task #55, ConduitFuseCoreBodyPass.cpp shared
// `detail::isForwardChainEndpoint`):
//   --conduit-fuse-core-bodies detects that @fwd_out is referenced by a
//   `conduit.scatter` op and rejects the fusion_group match in Step 0.
//   The defensive guard inside `mergeAndUnifyDevices` is a hard backstop.
//   Result:
//     * Both devices remain intact and unfused.
//     * @fwd_in, @fwd_out, and the conduit.scatter survive in devA.
//     * @comp_in, @comp_out, and the consumer core in devB are untouched.

// CHECK-LABEL: module @pattern_e_no_fuse_core_bodies_with_neighbor

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

module @pattern_e_no_fuse_core_bodies_with_neighbor {
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
