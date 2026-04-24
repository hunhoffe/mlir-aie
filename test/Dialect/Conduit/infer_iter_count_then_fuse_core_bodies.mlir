// RUN: aie-opt --objectfifo-to-conduit --conduit-fuse-core-bodies %s | FileCheck %s
//
// Task #11 — Pass A dma_repeat inference + fuse-core-bodies.
//
// Two cores on the SAME compute tile (tile(0,2)) connected by an
// intermediate aie.objectfifo.  Each core has a finite scf.for of 16
// iterations.  Pass A infers dma_repeat = 16 on every channel.
// fuse-core-bodies then merges the two cores into one and erases the
// intermediate conduit.
//
// HIGH-risk: fuse-core-bodies pre-existed before dma_repeat was a
// first-class input attr — the merged core retains its outer loop and
// the surviving (external) channels keep their inferred dma_repeat.
// If the merged-loop trip count is not preserved as expected, this test
// is the canary.  Pin XFAIL with a TODO if it fails.

// CHECK-LABEL: module @infer_then_fuse_core_bodies

// Intermediate channel erased; inputs and outputs survive with dma_repeat.
// CHECK-NOT:   conduit.create @intermediate
// CHECK:       conduit.create @input
// CHECK-SAME:  dma_repeat = 16
// CHECK:       conduit.create @output
// CHECK-SAME:  dma_repeat = 16

// One surviving aie.core after fusion.
// CHECK:       aie.core
// CHECK-NOT:   aie.core

module @infer_then_fuse_core_bodies {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @input(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @intermediate(%tile_0_2, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @output(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @produce_kernel(memref<128xbf16>, memref<128xbf16>)
    func.func private @consume_kernel(memref<128xbf16>, memref<128xbf16>)

    // Core A: producer.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c16 step %c1 {
        %in = aie.objectfifo.acquire @input(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %ix = aie.objectfifo.acquire @intermediate(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %ix_buf = aie.objectfifo.subview.access %ix[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @produce_kernel(%in_buf, %ix_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @intermediate(Produce, 1)
        aie.objectfifo.release @input(Consume, 1)
      }
      aie.end
    } {link_with = "producer.o"}

    // Core B: consumer (same tile).
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c16 step %c1 {
        %ix = aie.objectfifo.acquire @intermediate(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %ix_buf = aie.objectfifo.subview.access %ix[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @output(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @consume_kernel(%ix_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @output(Produce, 1)
        aie.objectfifo.release @intermediate(Consume, 1)
      }
      aie.end
    } {link_with = "consumer.o"}
  }
}
