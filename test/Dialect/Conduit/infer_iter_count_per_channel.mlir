// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #11 — Pass A dma_repeat inference is per-channel.
//
// Two channels:
//   @chanA — outer 16, inner 4, TAP covers inner walk (BD len 512, fifo
//            elem 128 → acquires_per_BD = 4) → dma_repeat = 64/4 = 16.
//   @chanB — same outer 16, inner 4 acquire site, but TAP covers ONE elem
//            per BD fire (BD len 64, fifo elem 64 → acquires_per_BD = 1)
//            → dma_repeat = 64/1 = 64.
//
// Verifies independence of the per-channel inference and the BD-length
// lookup.

// CHECK-LABEL: module @infer_per_channel
// CHECK: conduit.create @chanA
// CHECK-SAME: dma_repeat = 16
// CHECK: conduit.create @chanB
// CHECK-SAME: dma_repeat = 64

module @infer_per_channel {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chanA(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @chanB(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c16 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %subA = aie.objectfifo.acquire @chanA (Consume, 1)
              : !aie.objectfifosubview<memref<128xbf16>>
          %elemA = aie.objectfifo.subview.access %subA[0]
              : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
          aie.objectfifo.release @chanA (Consume, 1)
          %subB = aie.objectfifo.acquire @chanB (Consume, 1)
              : !aie.objectfifosubview<memref<64xbf16>>
          %elemB = aie.objectfifo.subview.access %subB[0]
              : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
          aie.objectfifo.release @chanB (Consume, 1)
        }
      }
      aie.end
    }

    aie.runtime_sequence(%a0: memref<512xbf16>, %b0: memref<64xbf16>) {
      // chanA: BD covers 4 elems per fire → acquires_per_BD = 4.
      %tA = aiex.dma_configure_task_for @chanA {
        aie.dma_bd(%a0 : memref<512xbf16>, 0, 512,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 512, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tA)
      aiex.dma_await_task(%tA)
      aiex.dma_free_task(%tA)
      // chanB: BD covers 1 elem per fire → acquires_per_BD = 1.
      %tB = aiex.dma_configure_task_for @chanB {
        aie.dma_bd(%b0 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tB)
      aiex.dma_await_task(%tB)
      aiex.dma_free_task(%tB)
    }
  }
}
