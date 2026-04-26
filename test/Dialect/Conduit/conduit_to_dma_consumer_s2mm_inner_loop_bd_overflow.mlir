// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// Pass C consumer-S2MM BD-length overflow regression pin.
//
// Bug location: ConduitToDMALink.cpp lines 2235-2241 (Phase 5.5 Case A
// consumer S2MM). The site computes per-buffer BD length from
// `info.numElems` FIRST (the SHIM aggregated per-dispatch transfer window
// from --dma-task-to-conduit) and only falls back to the per-tile buffer
// element count from `info.elemType` when numElems is zero. When an inner
// core loop on the consumer reuses the same FIFO slot multiple times per
// dispatch, numElems = (per-buffer × inner-loop trip count), causing the
// emitted dma_bd length to overflow the actual S2MM buffer.
//
// Bug class: same as the JOIN-last-slice overflow fixed by 7569b3e8e6 and
// pinned by conduit_to_dma_join_last_slice_bd_overflow.mlir. The cure is
// the single-source-of-truth helper `deriveBdLength` in ConduitToDMALink:
// always read getNumElements() off the actual buffer's MemRefType when
// available, and fall back to numElems only when no MemRefType is present.
//
// Topology (minimal reproducer):
//   shim(0,0) ─ aie.objectfifo @in (depth=2, memref<8xi32>) ─> compute(0,2)
//   shim runtime_sequence dispatches a memref<32xi32> in one go
//     → --dma-task-to-conduit stamps get_memref_async num_elems = 32
//     → Pass C info.numElems = 32 (the per-DISPATCH window, NOT per-slot)
//   core(0,2) inner scf.for trip=4 acquires/releases the depth-2 FIFO 4×
//     → S2MM BD length MUST be 8 (per-buffer), not 32 (per-dispatch)
//
// On the buggy emit, the S2MM BDs at compute(0,2) come out with length 32
// against memref<8xi32> buffers — a 4× overflow that hangs firmware.

// CHECK-LABEL: module @consumer_s2mm_inner_loop_bd_overflow
// CHECK:   aie.device(npu2)
// CHECK:     aie.buffer(%{{.*}}) {{.*}}sym_name = "in_cons_buff_0"{{.*}} memref<8xi32>
// CHECK:     aie.buffer(%{{.*}}) {{.*}}sym_name = "in_cons_buff_1"{{.*}} memref<8xi32>
// CHECK:     aie.mem(%{{.*}}tile_0_2)
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(%{{.*}}in_cons_buff_0 : memref<8xi32>, 0, 8)
// CHECK:       aie.dma_bd(%{{.*}}in_cons_buff_1 : memref<8xi32>, 0, 8)
// CHECK-NOT: aie.dma_bd(%{{.*}}memref<8xi32>{{.*}}, 32)

module @consumer_s2mm_inner_loop_bd_overflow {
  aie.device(npu2) {
    %shim_0  = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @in (%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @in(Consume, 1)
                : !aie.objectfifosubview<memref<8xi32>>
        aie.objectfifo.release @in(Consume, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%arg0: memref<32xi32>) {
      %t = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<32xi32>, 0, 32) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
