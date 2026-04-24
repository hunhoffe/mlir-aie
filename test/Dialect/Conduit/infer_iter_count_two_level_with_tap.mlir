// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #11 — Pass A dma_repeat inference, two-level loop with TAP that
// covers the inner walk.
//
// Geometry:
//   outer trip = 16, inner trip = 4 → total acquires = 64
//   shim BD len = 512, fifo elem count = 128 → acquires_per_BD = 4
//   → dma_repeat = 64 / 4 = 16

// CHECK-LABEL: module @infer_two_level_with_tap
// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 16

module @infer_two_level_with_tap {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c16 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %sub = aie.objectfifo.acquire @chan (Consume, 1)
              : !aie.objectfifosubview<memref<128xbf16>>
          %elem = aie.objectfifo.subview.access %sub[0]
              : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
          aie.objectfifo.release @chan (Consume, 1)
        }
      }
      aie.end
    }

    aie.runtime_sequence(%a0: memref<512xbf16>) {
      %t = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%a0 : memref<512xbf16>, 0, 512,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 512, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}
