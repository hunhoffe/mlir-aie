// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Regression test for Step 8c bug in --conduit-fuse-operators:
//
// After fusion, Step 8c reconstructs runtime_sequence block args by creating
// one arg per surviving conduit.put/get_memref op, with the type derived from
// num_elems.  This is wrong for multi-column operators where multiple per-tile
// ops (each with num_elems = per-tile-count) share the same full-buffer host
// arg.
//
// This test models a SiLU+EltMul fusion pattern with 2 columns:
//   DevA (SiLU-like):  ext_in (gate) → inter_out (fusible)
//   DevB (EltMul-like): inter_in (fusible) + ext_up → ext_out (hidden)
//
// After fusion:
//   Phase 2 merges args: devA(2) + devB(3) = 5 block args
//   Step 6b erases 4 intermediate ops (2 get_memref + 2 put_memref)
//   Surviving: 2 ext_in + 2 ext_up + 2 ext_out = 6 ops
//   Step 8c fires (6 ≠ 5) and emits 6 × memref<128xbf16> args
//   Correct: 3 × memref<256xbf16> (full-buffer, matching host buffers)

// CHECK-LABEL: module @fuse_operators_runtime_seq_args

// Only one device after fusion:
// CHECK:       aie.device(npu2)

// The fused runtime_sequence must have full-buffer args (memref<256xbf16>),
// NOT per-tile args (memref<128xbf16>):
// CHECK:       aie.runtime_sequence
// CHECK-SAME:  memref<256xbf16>
// CHECK-SAME:  memref<256xbf16>
// CHECK-SAME:  memref<256xbf16>

// No second device:
// CHECK-NOT:   aie.device(npu2)

module @fuse_operators_runtime_seq_args {
  // DevA: SiLU-like operator (2 columns).
  // Reads gate input from LPDDR5, writes intermediate to LPDDR5 (fusible).
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %shim_1 = aie.tile(1, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)

    // External input (gate): full buffer distributed across 2 columns.
    aie.objectfifo @ext_in_0(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @ext_in_1(%shim_1, {%tile_1_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    // Intermediate output (fusible): distributed across 2 columns.
    aie.objectfifo @inter_out_0(%tile_0_2, {%shim_0}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @inter_out_1(%tile_1_2, {%shim_1}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @silu_kernel(memref<128xbf16>, memref<128xbf16>)

    %core_0 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_0(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_out_0(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @silu_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @inter_out_0(Produce, 1)
        aie.objectfifo.release @ext_in_0(Consume, 1)
      }
      aie.end
    } {link_with = "silu.a"}

    %core_1 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_1(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_out_1(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @silu_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @inter_out_1(Produce, 1)
        aie.objectfifo.release @ext_in_1(Consume, 1)
      }
      aie.end
    } {link_with = "silu.a"}

    // Full-buffer args: memref<256xbf16> = 2 columns × 128 elements.
    aie.runtime_sequence(%full_gate: memref<256xbf16>, %full_inter: memref<256xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_0 {
        aie.dma_bd(%full_gate : memref<256xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_in_1 {
        aie.dma_bd(%full_gate : memref<256xbf16>, 128, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @inter_out_0 {
        aie.dma_bd(%full_inter : memref<256xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @inter_out_1 {
        aie.dma_bd(%full_inter : memref<256xbf16>, 128, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t3)
      aiex.dma_await_task(%t3)
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
      aiex.dma_free_task(%t2)
    }
  }

  // DevB: EltMul-like operator (2 columns).
  // Reads intermediate (fusible) + up from LPDDR5, writes hidden to LPDDR5.
  aie.device(npu2) @devB {
    %shim_0 = aie.tile(0, 0)
    %shim_1 = aie.tile(1, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)

    // Intermediate input (fusible): distributed across 2 columns.
    aie.objectfifo @inter_in_0(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @inter_in_1(%shim_1, {%tile_1_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

    // External input (up): full buffer distributed across 2 columns.
    aie.objectfifo @ext_up_0(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @ext_up_1(%shim_1, {%tile_1_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    // External output (hidden): full buffer distributed across 2 columns.
    aie.objectfifo @ext_out_0(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @ext_out_1(%tile_1_2, {%shim_1}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @eltmul_kernel(memref<128xbf16>, memref<128xbf16>,
                                      memref<128xbf16>)

    %core_0 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %inter = aie.objectfifo.acquire @inter_in_0(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %inter_buf = aie.objectfifo.subview.access %inter[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %up = aie.objectfifo.acquire @ext_up_0(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %up_buf = aie.objectfifo.subview.access %up[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @ext_out_0(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @eltmul_kernel(%inter_buf, %up_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_out_0(Produce, 1)
        aie.objectfifo.release @ext_up_0(Consume, 1)
        aie.objectfifo.release @inter_in_0(Consume, 1)
      }
      aie.end
    } {link_with = "eltmul.a"}

    %core_1 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %inter = aie.objectfifo.acquire @inter_in_1(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %inter_buf = aie.objectfifo.subview.access %inter[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %up = aie.objectfifo.acquire @ext_up_1(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %up_buf = aie.objectfifo.subview.access %up[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @ext_out_1(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @eltmul_kernel(%inter_buf, %up_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_out_1(Produce, 1)
        aie.objectfifo.release @ext_up_1(Consume, 1)
        aie.objectfifo.release @inter_in_1(Consume, 1)
      }
      aie.end
    } {link_with = "eltmul.a"}

    // Full-buffer args: 3 host buffers each memref<256xbf16>.
    aie.runtime_sequence(%full_inter: memref<256xbf16>, %full_up: memref<256xbf16>,
                          %full_hidden: memref<256xbf16>) {
      %t0 = aiex.dma_configure_task_for @inter_in_0 {
        aie.dma_bd(%full_inter : memref<256xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_in_1 {
        aie.dma_bd(%full_inter : memref<256xbf16>, 128, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @ext_up_0 {
        aie.dma_bd(%full_up : memref<256xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @ext_up_1 {
        aie.dma_bd(%full_up : memref<256xbf16>, 128, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3)
      %t4 = aiex.dma_configure_task_for @ext_out_0 {
        aie.dma_bd(%full_hidden : memref<256xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t4)
      %t5 = aiex.dma_configure_task_for @ext_out_1 {
        aie.dma_bd(%full_hidden : memref<256xbf16>, 128, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t5)
      aiex.dma_await_task(%t5)
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
      aiex.dma_free_task(%t2)
      aiex.dma_free_task(%t3)
      aiex.dma_free_task(%t4)
    }
  }
}
