// RUN: aie-opt --objectfifo-to-conduit --conduit-fuse-operators --conduit-to-dma %s | FileCheck %s
//
// Pass A → spatial-fusion → Pass C end-to-end test starting from
// `aie.objectfifo {fusion_group = ...}`.
//
// Coverage gap (audit GAP C): existing tests cover Pass A's fusion_group
// propagation in isolation (objectfifo_to_conduit_fusion_group.mlir) and
// fuse_operators_*.mlir tests that start from hand-written conduit IR.
// Nothing exercises the full IRON-style entry point: an aie.objectfifo
// carrying fusion_group, fused by --conduit-fuse-operators, then lowered
// by Pass C to real DMA.  This test pins down that the entire chain is
// wired correctly so that an IRON ObjectFifo(..., fusion_group="g0")
// lowers all the way to a flow + DMA path with the intermediate shim
// round-trip eliminated.
//
// Two minimal devices each with: one external shim channel + one fusible
// intermediate channel marked `fusion_group = "fg0"` + one core.  After
// --conduit-fuse-operators the two devices merge (devB's tiles are
// column-offset to col 1) and the intermediate channels collapse into a
// single fused_intermediate.

// CHECK-LABEL: module @objectfifo_fusion_group_passa_fuse_passc

// Only one device remains after fusion (devA/devB merged).
// CHECK:       aie.device(npu2)

// Both producer and consumer compute tiles survive (devB's col offset to 1).
// CHECK-DAG:   aie.tile(0, 2)
// CHECK-DAG:   aie.tile(1, 2)

// External shim allocations for the surviving I/O channels remain, but
// the per-device intermediate shim allocations must be GONE (fusion
// eliminates the LPDDR round-trip).
// CHECK-DAG:   aie.shim_dma_allocation @ext_in_a
// CHECK-DAG:   aie.shim_dma_allocation @ext_out_b
// CHECK-NOT:   aie.shim_dma_allocation @inter_a
// CHECK-NOT:   aie.shim_dma_allocation @inter_b

// The fused intermediate becomes a direct compute-to-compute flow (no shim).
// CHECK:       aie.flow(%{{.*}}, DMA :{{.*}}, %{{.*}}, DMA :

// All conduit ops fully lowered.
// CHECK-NOT:   conduit.create
// CHECK-NOT:   conduit.acquire
// CHECK-NOT:   conduit.release

module @objectfifo_fusion_group_passa_fuse_passc {
  aie.device(npu2) @devA {
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)

    // External input: shim → compute.
    aie.objectfifo @ext_in_a (%shim_a, {%tile_a}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    // Fusible intermediate: compute → shim (will be fused away).
    aie.objectfifo @inter_a (%tile_a, {%shim_a}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @producer_kernel(memref<128xbf16>, memref<128xbf16>)

    aie.core(%tile_a) {
      %in = aie.objectfifo.acquire @ext_in_a (Consume, 1)
          : !aie.objectfifosubview<memref<128xbf16>>
      %in_buf = aie.objectfifo.subview.access %in[0]
          : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
      %out = aie.objectfifo.acquire @inter_a (Produce, 1)
          : !aie.objectfifosubview<memref<128xbf16>>
      %out_buf = aie.objectfifo.subview.access %out[0]
          : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
      func.call @producer_kernel(%in_buf, %out_buf)
          : (memref<128xbf16>, memref<128xbf16>) -> ()
      aie.objectfifo.release @inter_a (Produce, 1)
      aie.objectfifo.release @ext_in_a (Consume, 1)
      aie.end
    } {link_with = "producer.a"}
  }

  aie.device(npu2) @devB {
    %shim_b = aie.tile(0, 0)
    %tile_b = aie.tile(0, 2)

    // Fusible intermediate: shim → compute (matches devA's fusion_group).
    aie.objectfifo @inter_b (%shim_b, {%tile_b}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

    // External output: compute → shim.
    aie.objectfifo @ext_out_b (%tile_b, {%shim_b}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @consumer_kernel(memref<128xbf16>, memref<128xbf16>)

    aie.core(%tile_b) {
      %in = aie.objectfifo.acquire @inter_b (Consume, 1)
          : !aie.objectfifosubview<memref<128xbf16>>
      %in_buf = aie.objectfifo.subview.access %in[0]
          : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
      %out = aie.objectfifo.acquire @ext_out_b (Produce, 1)
          : !aie.objectfifosubview<memref<128xbf16>>
      %out_buf = aie.objectfifo.subview.access %out[0]
          : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
      func.call @consumer_kernel(%in_buf, %out_buf)
          : (memref<128xbf16>, memref<128xbf16>) -> ()
      aie.objectfifo.release @ext_out_b (Produce, 1)
      aie.objectfifo.release @inter_b (Consume, 1)
      aie.end
    } {link_with = "consumer.a"}
  }
}
