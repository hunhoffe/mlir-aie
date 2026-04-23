// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit %s | FileCheck %s
//
// Regression test for the FS5 EXTENSION: --dma-task-to-conduit must also
// handle BDs whose outermost dim is a REPEAT factor with a NON-ZERO stride
// (i.e. the iteration wraps over already-addressed memory rather than
// indexing disjoint elements).
//
// Surfaced by the first real conduit Llama compile (mode B, 2026-04-22):
// IRON's runlist emits BDs whose `len` reflects one buffer pass, while the
// outermost dim multiplies the iteration count. The repeat is *also*
// reflected on the source `aiex.dma_configure_task_for` via `repeat_count`.
//
// Reproducer pattern (one shape from the failing Llama compile, scaled
// down for testability):
//
//   aie.dma_bd(%arg : memref<8x1024xbf16>, 0, 4096,
//     [<size = 2, stride = 1024>,    // <-- non-zero-stride OUTER repeat
//      <size = 4, stride = 1024>,
//      <size = 32, stride = 32>,
//      <size = 32, stride = 1>])
//        {repeat_count = 1 : i32}
//
// product(sizes) = 2*4*32*32 = 8192 = 2 * len(4096) → 2× over-count.
//
// Before this fix:
//   `dimsToOffsetsStrides` only stripped `stride==0` dims. The non-zero
//   repeat dim was emitted into `sizes`, so the verifier rejected the
//   resulting put_memref:
//     "num_elems (4096) does not match product of sizes (8192)"
//
// After this fix:
//   Outer dims are peeled iteratively while product(sizes) > len, leaving
//   only addressable iteration dims. The full BDDimLayout (including the
//   peeled repeat) is preserved on `producer_dimensions` for downstream
//   MM2S DMA programming.

// CHECK-LABEL: module @dma_task_to_conduit_repeat_bd_nonzero_stride
module @dma_task_to_conduit_repeat_bd_nonzero_stride {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Channel sized to one buffer pass (4096 elements).
    aie.objectfifo @ext_in(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<4096xbf16>>

    func.func private @kernel(memref<4096xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in(Consume, 1)
            : !aie.objectfifosubview<memref<4096xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
        func.call @kernel(%in_buf) : (memref<4096xbf16>) -> ()
        aie.objectfifo.release @ext_in(Consume, 1)
      }
      aie.end
    } {link_with = "kernel.a"}

    // Runtime sequence: one MM2S BD with an outer non-zero-stride repeat.
    aie.runtime_sequence(%arg0: memref<8x1024xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<8x1024xbf16>, 0, 4096,
          [<size = 2, stride = 1024>,
           <size = 4, stride = 1024>,
           <size = 32, stride = 32>,
           <size = 32, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 1 : i32}
      aiex.dma_start_task(%t0)
      aiex.dma_free_task(%t0)
    }
  }
}

// num_elems must equal `len` (4096). The outer <size=2, stride=1024> repeat
// dim is peeled from sizes/strides; the inner addressable dims survive:
//   sizes = [4, 32, 32], strides = [1024, 32, 1], product = 4096 = num_elems.
//
// CHECK:       conduit.put_memref
// CHECK-SAME:  name = @ext_in
// CHECK-SAME:  num_elems = 4096
// CHECK-SAME:  offsets = array<i64: 0, 0, 0>
//
// Full BDDimLayout (including the peeled repeat dim) is preserved as
// producer_dimensions for downstream MM2S DMA programming.
//
// CHECK-SAME:  producer_dimensions =
// CHECK-SAME:  <size = 2, stride = 1024>
// CHECK-SAME:  <size = 4, stride = 1024>
// CHECK-SAME:  <size = 32, stride = 32>
// CHECK-SAME:  <size = 32, stride = 1>
// CHECK-SAME:  sizes = array<i64: 4, 32, 32>
// CHECK-SAME:  strides = array<i64: 1024, 32, 1>
