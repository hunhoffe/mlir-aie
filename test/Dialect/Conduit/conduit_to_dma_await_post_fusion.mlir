// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators --conduit-depth-promote --conduit-to-dma --aie-assign-buffer-addresses %s | FileCheck %s
//
// Regression test for Bug 2: --conduit-to-dma Step 8g used positional
// blockArgs[i] mapping for put/get_memref ops.  After fusion, the merged
// runtime_sequence has more ops than block args (multiple per-column ops
// share one full-buffer block arg).  Ops at index >= blockArgs.size() were
// skipped — no DMA task emitted, no await.  If skipped ops were outputs
// (get_memref), the host reads before DMA completes → race condition.
//
// This test models a SiLU+EltMul fusion with 2 columns where block args
// use per-tile types (memref<128xbf16>).  After fusion:
//   - 6 surviving put/get_memref ops (2 ext_in + 2 ext_up + 2 ext_out)
//   - 3 block args (ext_in, ext_up, ext_out)
//   - Step 8g must emit 6 dma_configure_task_for ops (one per put/get op)
//   - Must emit dma_await_task for each S2MM (output) task
//
// Before fix: only 3 dma_configure_task_for emitted (ops 4-6 skipped),
//   output awaits missing → race condition.
// After fix: all 6 ops lowered, correct offsets, proper await/free.

// CHECK-LABEL: module @conduit_to_dma_await_post_fusion

// Only one device after fusion:
// CHECK:       aie.device(npu2)

// The fused runtime_sequence must have 3 full-buffer args:
// CHECK:       aie.runtime_sequence
// CHECK-SAME:  memref<256xbf16>
// CHECK-SAME:  memref<256xbf16>
// CHECK-SAME:  memref<256xbf16>

// Each per-column put/get pair emits a dma_configure_task_for — total 6
// (2 ext_in MM2S + 2 ext_up MM2S + 2 ext_out S2MM).  None silently skipped
// (the original Bug 2 — block-arg index mismatch dropping ops at index >=
// blockArgs.size()).
// CHECK-COUNT-6: aiex.dma_configure_task_for
// CHECK-NOT:     aiex.dma_configure_task_for

// Every configured task must be released — exactly one await + five frees,
// total 6, matching the configure count.  Under conditional async emission,
// only the configure consumed by an IRON dma_await_task in source takes the
// async/await path; the rest emit dma_free_task.  await + frees are
// interleaved in source-relative position, so use CHECK-DAG to count
// occurrences without pinning order.
// CHECK-DAG:     aiex.dma_await_task
// CHECK-DAG:     aiex.dma_free_task
// CHECK-DAG:     aiex.dma_free_task
// CHECK-DAG:     aiex.dma_free_task
// CHECK-DAG:     aiex.dma_free_task
// CHECK-DAG:     aiex.dma_free_task
// CHECK-NOT:     aiex.dma_await_task
// CHECK-NOT:     aiex.dma_free_task

// No second device:
// CHECK-NOT:   aie.device(npu2)

module @conduit_to_dma_await_post_fusion {
  // DevA: SiLU-like operator (2 columns).
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %shim_1 = aie.tile(1, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)

    aie.objectfifo @ext_in_0(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @ext_in_1(%shim_1, {%tile_1_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

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

    // Per-tile block arg types — DMA offsets span full buffer.
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
  // Shims at columns 2/3 (distinct from devA's cols 0/1) so devB's tile set
  // is NOT a subset of devA's; this preserves the offset path the original
  // CHECK assertions assume (post-fix: devB → +colMaxA+1 = +2).
  aie.device(npu2) @devB {
    %shim_0 = aie.tile(2, 0)
    %shim_1 = aie.tile(3, 0)
    %tile_0_2 = aie.tile(2, 2)
    %tile_1_2 = aie.tile(3, 2)

    aie.objectfifo @inter_in_0(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @inter_in_1(%shim_1, {%tile_1_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

    aie.objectfifo @ext_up_0(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @ext_up_1(%shim_1, {%tile_1_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

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

    // Per-tile block arg types — DMA offsets span full buffer.
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
