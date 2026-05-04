// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Foundation Phase 2 (Task #19), gap #3 — pin the "softmax fusion
// landmine" current behavior.
//
// Geometry (cross-device fusion via fusion_group="fg0"):
//   * devA producer (@inter_out): nested loop 16 outer × 4 inner =
//     64 acquires — Pattern A.
//   * devB consumer (@inter_in):  single flattened scf.for(0,64) =
//     64 acquires — Pattern C (the IRON Softmax / range_(N*M) shape).
//   * Both endpoints use 1-elem fifo type and a single shim BD on each
//     device, so total acquires per side match (64 == 64).
//
// CURRENT BEHAVIOR (the landmine this test pins):
//   --conduit-fuse-operators DOES NOT error on this geometry — the
//   chunk-count compatibility check (ConduitFuseOperators.cpp:894-907)
//   only compares partitioned runtime-sequence chunks, not the
//   per-side core-loop SHAPE.  After fusion:
//     * devB is merged into devA; a single device remains.
//     * @inter_out and @inter_in are erased; @fused_intermediate_0
//       takes their place.
//     * BOTH cores survive intact with their original loop SHAPES
//       (Pattern A nest on one tile; Pattern C flat loop on the other).
//       fuse-operators auto-spreads them across columns (tile(0,2) and
//       tile(1,2)).
//     * The merged runtime sequence carries one put_memref + one
//       get_memref in a single chunk.
//
//   This is NOT necessarily desirable: the producer and consumer cores
//   now run on the FUSED intermediate channel with mismatched
//   per-iteration step counts (4-deep producer iters per outer vs flat
//   1-deep consumer iters), which can manifest as buffer-protocol
//   asymmetry under depth-2 double-buffering even though the totals
//   agree.  See PLAN.md backlog entry "softmax-fusion landmine"
//   (gap #3) — the current "silently merge with divergent loop
//   shapes" behavior should likely be tightened to either error or
//   reshape the merged loops, but until then this test is the
//   regression net for the silent-merge contract.

// CHECK-LABEL: module @softmax_fusion_landmine

// Single device after the cross-device merge (devB was merged into devA).
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device

// Fused intermediate replaces both inter_out and inter_in.
// CHECK:       conduit.create @fused_intermediate_0
// CHECK-NOT:   conduit.create @inter_out
// CHECK-NOT:   conduit.create @inter_in

// Producer core (Pattern A nested 16 * 4) survives intact.
// CHECK:       aie.core
// CHECK:       scf.for
// CHECK:       scf.for
// CHECK:       conduit.acquire {{.*}} name = @fused_intermediate_0
// CHECK-SAME:  port = #conduit.port<Produce>

// Consumer core (Pattern C flat 64) survives intact — single scf.for
// at this nesting level, on a different tile from the producer.
// CHECK:       aie.core(%tile_1_2)
// CHECK:       scf.for
// CHECK-NOT:   scf.for
// CHECK:       conduit.acquire {{.*}} name = @fused_intermediate_0
// CHECK-SAME:  port = #conduit.port<Consume>

module @softmax_fusion_landmine {
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @ext_in_a(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<1xbf16>>
    aie.objectfifo @inter_out(%tile_0_2, {%shim_0}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<1xbf16>>

    func.func private @producer_kernel(memref<1xbf16>, memref<1xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      // Pattern A: 16 outer × 4 inner = 64 acquires.
      scf.for %i = %c0 to %c16 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %in = aie.objectfifo.acquire @ext_in_a(Consume, 1)
              : !aie.objectfifosubview<memref<1xbf16>>
          %in_buf = aie.objectfifo.subview.access %in[0]
              : !aie.objectfifosubview<memref<1xbf16>> -> memref<1xbf16>
          %out = aie.objectfifo.acquire @inter_out(Produce, 1)
              : !aie.objectfifosubview<memref<1xbf16>>
          %out_buf = aie.objectfifo.subview.access %out[0]
              : !aie.objectfifosubview<memref<1xbf16>> -> memref<1xbf16>
          func.call @producer_kernel(%in_buf, %out_buf)
              : (memref<1xbf16>, memref<1xbf16>) -> ()
          aie.objectfifo.release @inter_out(Produce, 1)
          aie.objectfifo.release @ext_in_a(Consume, 1)
        }
      }
      aie.end
    } {link_with = "producer.a"}

    aie.runtime_sequence(%a0: memref<64xbf16>, %a1: memref<64xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_a {
        aie.dma_bd(%a0 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_out {
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

  aie.device(npu2) @devB {
    // shim at column 1 (distinct from devA's column-0 shim) so devB's
    // tile set is NOT a subset of devA's; preserves the offset path the
    // original CHECK assertions implicitly assume.
    %shim_0 = aie.tile(1, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @inter_in(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<1xbf16>>
    aie.objectfifo @ext_out(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<1xbf16>>

    func.func private @consumer_kernel(memref<1xbf16>, memref<1xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      // Pattern C: single flattened 64-iter loop.
      scf.for %i = %c0 to %c64 step %c1 {
        %in = aie.objectfifo.acquire @inter_in(Consume, 1)
            : !aie.objectfifosubview<memref<1xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<1xbf16>> -> memref<1xbf16>
        %out = aie.objectfifo.acquire @ext_out(Produce, 1)
            : !aie.objectfifosubview<memref<1xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<1xbf16>> -> memref<1xbf16>
        func.call @consumer_kernel(%in_buf, %out_buf)
            : (memref<1xbf16>, memref<1xbf16>) -> ()
        aie.objectfifo.release @ext_out(Produce, 1)
        aie.objectfifo.release @inter_in(Consume, 1)
      }
      aie.end
    } {link_with = "consumer.a"}

    aie.runtime_sequence(%b0: memref<64xbf16>, %b1: memref<64xbf16>) {
      %t0 = aiex.dma_configure_task_for @inter_in {
        aie.dma_bd(%b0 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_out {
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
