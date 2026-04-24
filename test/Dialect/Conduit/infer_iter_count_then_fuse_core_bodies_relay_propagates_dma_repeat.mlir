// RUN: aie-opt --objectfifo-to-conduit --conduit-fuse-core-bodies %s | FileCheck %s
//
// Foundation Phase 2 (Task #19), gap #5 (FIXED) — Pass A inference followed
// by fuse-core-bodies in the MemTile-relay route.  Pins the post-fix behavior
// of `ConduitFuseCoreBodyPass.cpp::emitMemTileRelay`: the relay-side
// `conduit.create` now propagates `bd_repeat` / `dma_repeat` from the
// source intermediate channel rather than emitting `nullptr` (was the
// documented HIGH-risk gap from the fusion-loop-analysis audit).
//
// Geometry:
//   * Producer + consumer cores share tile(0,2); both have outer scf.for trip
//     = 16 → Pass A infers dma_repeat = 16 on every channel.
//   * Intermediate `@intermediate` element type is memref<131072xbf16> =
//     256 KiB, exceeding npu2 L1 (~64 KiB) → decideRoute() selects MemTile
//     relay (256 KiB << MemTile capacity 512 KiB) instead of L1.
//
// Expected behavior under the current implementation:
//   * @input and @output (the L3-facing channels) keep their inferred
//     dma_repeat = 16.
//   * The intermediate is split into @intermediate (producer side) and
//     @intermediate_relay (consumer side) by emitMemTileRelay; a
//     conduit.scatter is inserted between them.
//   * @intermediate_relay's conduit.create now carries the propagated
//     dma_repeat = 16 from the source intermediate (post-fix).

// CHECK-LABEL: module @infer_then_fuse_core_bodies_relay

// External (L3-facing) input endpoint retains Pass A's inferred dma_repeat.
// CHECK:       conduit.create @input
// CHECK-SAME:  dma_repeat = 16

// MemTile-relay endpoint emitted by emitMemTileRelay propagates the
// inferred dma_repeat from the source intermediate channel (post-fix).
// CHECK:       conduit.create @intermediate_relay
// CHECK-SAME:  dma_repeat = 16
// CHECK:       conduit.scatter

// External (L3-facing) output endpoint retains Pass A's inferred dma_repeat.
// CHECK:       conduit.create @output
// CHECK-SAME:  dma_repeat = 16

// One surviving aie.core after fusion.
// CHECK:       aie.core
// CHECK-NOT:   aie.core

module @infer_then_fuse_core_bodies_relay {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @input(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<131072xbf16>>
    // Intermediate is too large for L1 (256 KiB > 64 KiB npu2 L1) →
    // decideRoute() picks MemTile relay.
    aie.objectfifo @intermediate(%tile_0_2, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<131072xbf16>>
    aie.objectfifo @output(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<131072xbf16>>

    func.func private @produce_kernel(memref<131072xbf16>, memref<131072xbf16>)
    func.func private @consume_kernel(memref<131072xbf16>, memref<131072xbf16>)

    // Core A: producer.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c16 step %c1 {
        %in = aie.objectfifo.acquire @input(Consume, 1)
            : !aie.objectfifosubview<memref<131072xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<131072xbf16>> -> memref<131072xbf16>
        %ix = aie.objectfifo.acquire @intermediate(Produce, 1)
            : !aie.objectfifosubview<memref<131072xbf16>>
        %ix_buf = aie.objectfifo.subview.access %ix[0]
            : !aie.objectfifosubview<memref<131072xbf16>> -> memref<131072xbf16>
        func.call @produce_kernel(%in_buf, %ix_buf)
            : (memref<131072xbf16>, memref<131072xbf16>) -> ()
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
            : !aie.objectfifosubview<memref<131072xbf16>>
        %ix_buf = aie.objectfifo.subview.access %ix[0]
            : !aie.objectfifosubview<memref<131072xbf16>> -> memref<131072xbf16>
        %out = aie.objectfifo.acquire @output(Produce, 1)
            : !aie.objectfifosubview<memref<131072xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<131072xbf16>> -> memref<131072xbf16>
        func.call @consume_kernel(%ix_buf, %out_buf)
            : (memref<131072xbf16>, memref<131072xbf16>) -> ()
        aie.objectfifo.release @output(Produce, 1)
        aie.objectfifo.release @intermediate(Consume, 1)
      }
      aie.end
    } {link_with = "consumer.o"}
  }
}
