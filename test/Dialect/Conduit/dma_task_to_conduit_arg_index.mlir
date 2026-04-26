// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit %s | FileCheck %s
//
// Regression test for FS7: --dma-task-to-conduit must capture the
// BlockArgument that each `aie.dma_bd` references via an explicit
// `arg_index` attribute on the rebuilt `conduit.put_memref` /
// `conduit.get_memref` op.  This binding is required by --conduit-to-dma
// Step 8g to round-trip back to `aiex.dma_configure_task_for`.
//
// Background:
//   The Llama runlist emits S2MM output BDs whose `aie.dma_bd` offset is
//   a runtime patch marker (e.g. `0xDEADBEE0`).  --conduit-to-dma's old
//   Step 8g rebuild used an offset==0 heuristic to group ops by block
//   arg ("each 0-offset op starts a new arg group").  A non-zero patch
//   marker on the OUTPUT BD silently mis-grouped it with the INPUT arg,
//   producing IR where the output BD wrote to the input host buffer.
//
//   The fix: --dma-task-to-conduit captures `dmaBd.getBuffer()`'s
//   `BlockArgument::getArgNumber()` directly into the `arg_index` attr,
//   and Step 8g uses it as the authoritative binding (heuristic deleted).
//
// Reproducer pattern (two args, the OUTPUT BD has a non-zero offset
// that would have triggered the FS7 mis-grouping):
//
//   aie.runtime_sequence(%arg0: memref<...>, %arg1: memref<...>) {
//     %t0 = aiex.dma_configure_task_for @ext_in {
//       aie.dma_bd(%arg0 : ..., 0, ...)   // INPUT, offset=0
//       aie.end
//     }
//     %t1 = aiex.dma_configure_task_for @ext_out {
//       aie.dma_bd(%arg1 : ..., 3735929056, ...)  // OUTPUT, offset = 0xDEADBEE0
//       aie.end
//     }
//     ...
//   }
//
// Before fix:
//   put_memref (input, arg0) had offsets=[0]  → heuristic: arg0 group
//   get_memref (output, arg1) had offsets=[0xDEADBEE0] (non-zero)
//                              → heuristic: SAME group (arg0). WRONG.
//
// After fix:
//   put_memref carries arg_index=0; get_memref carries arg_index=1.
//   Step 8g rebuilds aie.dma_bd(%argN, ...) from arg_index directly.

// CHECK-LABEL: module @dma_task_to_conduit_arg_index
module @dma_task_to_conduit_arg_index {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // External input from LPDDR5 (MM2S; shim → tile).
    aie.objectfifo @ext_in(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    // External output to LPDDR5 (S2MM; tile → shim).
    aie.objectfifo @ext_out(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @ext_out(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_in(Consume, 1)
        aie.objectfifo.release @ext_out(Produce, 1)
      }
      aie.end
    } {link_with = "kernel.a"}

    // Runtime sequence: two args.  Input BD on %arg0 has offset=0.
    // Output BD on %arg1 has offset=0xDEADBEE0 (a non-zero IRON patch
    // marker).  The OLD offsets[0]==0 heuristic in --conduit-to-dma
    // Step 8g would have grouped the output op into the input arg group;
    // explicit arg_index makes this impossible.
    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      %t1 = aiex.dma_configure_task_for @ext_out {
        aie.dma_bd(%arg1 : memref<128xbf16>, 3735929056, 128)
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
    }
  }
}

// The MM2S BD on %arg0 lowers to a put_memref_async carrying arg_index = 0
// (block-arg 0 of the runtime_sequence).  Async because the input has
// IRON dma_await_task / dma_free_task consumers — see conditional emission
// in --dma-task-to-conduit (ConduitDmaTaskToConduit.cpp file header).
//
// CHECK:       conduit.put_memref_async
// CHECK-SAME:  arg_index = 0
// CHECK-SAME:  name = @ext_in
// CHECK-SAME:  num_elems = 128
// CHECK-SAME:  offsets = array<i64: 0>
//
// The S2MM BD on %arg1 with the non-zero patch-marker offset lowers to a
// get_memref_async carrying arg_index = 1 (block-arg 1 of the
// runtime_sequence).  Crucially the offsets[0] value (0xDEADBEE0 =
// 3735929056) is preserved on the conduit op without affecting the arg
// binding.  Under the old heuristic this op would have been mis-bound to
// %arg0.
//
// CHECK:       conduit.get_memref_async
// CHECK-SAME:  arg_index = 1
// CHECK-SAME:  name = @ext_out
// CHECK-SAME:  num_elems = 128
// CHECK-SAME:  offsets = array<i64: -559038240>
// (= 0xDEADBEE0 sign-interpreted as i64 — MLIR prints DenseArrayAttr<i64> as signed)
//
// IRON's dma_await_task / dma_free_task consumers survive as
// conduit.wait_all ops with the appropriate token attribute (token=true
// for await, token=false for free).
//
// CHECK:       conduit.wait_all
// CHECK:       conduit.wait_all
// CHECK:       conduit.wait_all
