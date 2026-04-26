// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit %s | FileCheck %s
//
// Regression test for FS5: --dma-task-to-conduit must handle BDs with a
// non-trivial broadcast/repeat dim (`size = N, stride = 0` with N > 1).
//
// IRON's Repeat operator emits a BD with an outer broadcast dim:
//
//   aie.dma_bd(%buf : memref<...>, 0, BUFFER_LEN,
//     [<size = R, stride = 0>,         // <-- repeat / broadcast
//      <size = ..., stride = ...>,
//      ...])
//
// Here BUFFER_LEN = 128 elements per pass; the outer <size=4, stride=0>
// dim broadcasts the buffer 4 times into the channel (channel sees
// 4 * 128 = 512 elements, but `len` records 128).
//
// Before fix:
//   `dimsToOffsetsStrides` only stripped LEADING <size=1, stride=0> dims.
//   The <size=4, stride=0> repeat dim was emitted into `sizes`, so
//   `product(sizes) = 4 * 128 = 512`, but `num_elems = len = 128`.
//   The PutMemref verifier rejected this with:
//     "num_elems (128) does not match product of sizes (512)"
//
// After fix:
//   All `stride == 0` dims (broadcast/repeat or trivial filler) are
//   stripped from `sizes/strides`. `producer_dimensions` retains the full
//   BDDimLayout (incl. the repeat dim) so downstream MM2S DMA programming
//   sees the broadcast.

// CHECK-LABEL: module @dma_task_to_conduit_repeat_bd
module @dma_task_to_conduit_repeat_bd {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // External input from LPDDR5; will be programmed with a repeat-style BD.
    aie.objectfifo @ext_in(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @repeat_kernel(memref<128xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @repeat_kernel(%in_buf) : (memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_in(Consume, 1)
      }
      aie.end
    } {link_with = "repeat.a"}

    // Runtime sequence: a single MM2S BD with an outer <size=4, stride=0>
    // broadcast dim (the FS5 reproducer pattern).
    aie.runtime_sequence(%arg0: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128,
          [<size = 4, stride = 0>,
           <size = 1, stride = 0>,
           <size = 1, stride = 0>,
           <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_free_task(%t0)
    }
  }
}

// The MM2S BD lowers to a conduit.put_memref_async (async because the
// input has IRON dma_free_task — see conditional emission in
// --dma-task-to-conduit / ConduitDmaTaskToConduit.cpp file header).
//
// num_elems must equal `len` (128, the per-pass buffer count) and must equal
// the product of `sizes`. After stripping stride=0 dims (repeat + filler
// dims), only the inner <size=128, stride=1> dim survives, so:
//   sizes = [128], strides = [1], product = 128 = num_elems. Verifier passes.
//
// CHECK:       conduit.put_memref_async
// CHECK-SAME:  name = @ext_in
// CHECK-SAME:  num_elems = 128
// CHECK-SAME:  offsets = array<i64: 0>
//
// The full BDDimLayout (including the repeat dim) is preserved as
// producer_dimensions so MM2S DMA programming still sees the broadcast.
//
// CHECK-SAME:  producer_dimensions =
// CHECK-SAME:  <size = 4, stride = 0>
// CHECK-SAME:  <size = 128, stride = 1>
// CHECK-SAME:  sizes = array<i64: 128>
// CHECK-SAME:  strides = array<i64: 1>
