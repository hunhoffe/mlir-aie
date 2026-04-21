// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Regression test for Bug 1: Step 8c in --conduit-fuse-operators emits per-tile
// arg types instead of full-buffer types.
//
// This test models the same SiLU+EltMul fusion pattern as
// fuse_operators_runtime_seq_args.mlir, but the runtime_sequence block args
// have per-tile types (memref<128xbf16>) while the dma_bds still reference
// them with full-buffer offsets (0 and 128).  This is the pattern IRON
// generates for multi-column operators: block arg type is per-tile, but the
// DMA descriptor spans the full host buffer.
//
// Before fix: Step 8c used origTypes → 3 × memref<128xbf16> (per-tile, wrong)
// After fix: Step 8c uses maxExtent → 3 × memref<256xbf16> (full-buffer, correct)

// CHECK-LABEL: module @fuse_operators_per_tile_args

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

module @fuse_operators_per_tile_args {
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

    // Per-tile block arg types (memref<128xbf16>) — the block arg type
    // matches one tile's share, but the DMA offsets span the full buffer.
    // This is the pattern that triggers Bug 1: Step 8c used origTypes
    // (memref<128xbf16>) instead of computing full-buffer extent (256).
    aie.runtime_sequence(%gate: memref<128xbf16>, %inter_out: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_0 {
        aie.dma_bd(%gate : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_in_1 {
        aie.dma_bd(%gate : memref<128xbf16>, 128, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @inter_out_0 {
        aie.dma_bd(%inter_out : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @inter_out_1 {
        aie.dma_bd(%inter_out : memref<128xbf16>, 128, 128,
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

    // Per-tile block arg types: 3 × memref<128xbf16>.
    // Each block arg is shared by 2 columns via DMA offsets.
    aie.runtime_sequence(%inter_in: memref<128xbf16>, %up: memref<128xbf16>,
                          %hidden: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @inter_in_0 {
        aie.dma_bd(%inter_in : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_in_1 {
        aie.dma_bd(%inter_in : memref<128xbf16>, 128, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @ext_up_0 {
        aie.dma_bd(%up : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @ext_up_1 {
        aie.dma_bd(%up : memref<128xbf16>, 128, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3)
      %t4 = aiex.dma_configure_task_for @ext_out_0 {
        aie.dma_bd(%hidden : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t4)
      %t5 = aiex.dma_configure_task_for @ext_out_1 {
        aie.dma_bd(%hidden : memref<128xbf16>, 128, 128,
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
