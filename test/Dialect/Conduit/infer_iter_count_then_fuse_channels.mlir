// RUN: aie-opt --objectfifo-to-conduit --conduit-fuse-channels %s | FileCheck %s
//
// Task #11 — Pass A dma_repeat inference + fuse-channels.
//
// Two compute-to-compute channels with separate producers but a shared
// consumer tile.  Each side has a finite scf.for of 4 iterations, so
// Pass A infers dma_repeat = 4 on both channels.  fuse-channels then
// annotates them with a shared dma_channel_group_s2mm without disturbing
// the inferred dma_repeat.
//
// LOW-risk: fuse-channels only adds annotations; it does not erase or
// rewrite the conduit.create ops, so the inferred dma_repeat survives
// trivially.

// CHECK-LABEL: module @infer_then_fuse_channels

// Both channels keep dma_repeat = 4 AND gain S2MM fusion annotations.
// MLIR sorts attrs alphabetically: dma_channel_group_s2mm < dma_repeat.
// CHECK-SAME order matches that left-to-right ordering.
// CHECK:       conduit.create @chan_a
// CHECK-SAME:  dma_channel_group_s2mm = "group0"
// CHECK-SAME:  dma_repeat = 4

// CHECK:       conduit.create @chan_b
// CHECK-SAME:  dma_channel_group_s2mm = "group0"
// CHECK-SAME:  dma_repeat = 4

module @infer_then_fuse_channels {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    // Two compute-to-compute fifos sharing tile(0,4) as consumer.
    aie.objectfifo @chan_a(%tile_0_2, {%tile_0_4}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @chan_b(%tile_0_3, {%tile_0_4}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>

    func.func private @produce_a(memref<8xi32>)
    func.func private @produce_b(memref<8xi32>)
    func.func private @consume(memref<8xi32>)

    // Producer A.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = aie.objectfifo.acquire @chan_a(Produce, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %buf = aie.objectfifo.subview.access %w[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @produce_a(%buf) : (memref<8xi32>) -> ()
        aie.objectfifo.release @chan_a(Produce, 1)
      }
      aie.end
    }

    // Producer B.
    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = aie.objectfifo.acquire @chan_b(Produce, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %buf = aie.objectfifo.subview.access %w[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @produce_b(%buf) : (memref<8xi32>) -> ()
        aie.objectfifo.release @chan_b(Produce, 1)
      }
      aie.end
    }

    // Shared consumer: strictly sequential acquire of chan_a then chan_b.
    aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %wa = aie.objectfifo.acquire @chan_a(Consume, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %ba = aie.objectfifo.subview.access %wa[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @consume(%ba) : (memref<8xi32>) -> ()
        aie.objectfifo.release @chan_a(Consume, 1)
        %wb = aie.objectfifo.acquire @chan_b(Consume, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %bb = aie.objectfifo.subview.access %wb[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @consume(%bb) : (memref<8xi32>) -> ()
        aie.objectfifo.release @chan_b(Consume, 1)
      }
      aie.end
    }
  }
}
