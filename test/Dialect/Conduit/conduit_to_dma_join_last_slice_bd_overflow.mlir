// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// Pass C JOIN-last-slice BD-length overflow regression pin.
//
// Bug location: ConduitToDMALink.cpp lines 1086-1095 (S2MM ingest path) and
// 1162-1175 (MM2S send path). Both compute the LAST slice's BD length using
// `joinDstPerBufForLen`, which is sourced from the SHIM consumer's
// `get_memref_async num_elems` (the per-DISPATCH transfer window) instead of
// the actual memtile JOIN buffer element count.
//
// Topology (minimum reproducer of the GEMM @ attn_query LM-head pattern):
//   compute(0,2) ─┐
//                 ├──> memtile(0,1) ──> shim(0,0)
//   compute(0,3) ─┘
//   join_src_a:  compute(0,2) -> memtile, 64 bf16
//   join_src_b:  compute(0,3) -> memtile, 64 bf16
//   join_dst:    memtile -> shim, 128 bf16 (= 64 + 64, slice offsets [0, 64])
//   shim consumer: aie.dma_bd(%arg0 : memref<512xbf16>, 0, 512)
//                  -> get_memref_async num_elems = 512 (4× the per-slice buffer)
//
// Pass C's collect phase records info.numElems = 512 (from the get_memref_async
// num_elems attribute, which dominates the conduit.create elemType size = 128).
// ConduitToDMALink then uses joinDstPerBufForLen = 512 for the LAST slice
// length calc, producing dma_bd len = 512 - 64 = 448 against memref<128xbf16>
// JOIN buffers — a 4× overflow that hangs firmware on dispatch.
//
// Slice 0 BDs are unaffected because their length comes from
//   offsets[1] - offsets[0] = 64 - 0 = 64
// (NOT joinDstPerBufForLen). The bug is specific to the LAST slice's
// fall-through arm.
//
// This fixture pins the buggy emit. The CHECK lines below are written for the
// CORRECT post-fix BD length (64 on every memtile JOIN BD, matching the
// 64-elem per-source slice). They FAIL on the current buggy output, hence
// the XFAIL above. The conduit-dev that applies the fix removes the XFAIL.

// CHECK-LABEL: module @join_last_slice_bd_overflow
// CHECK:   aie.device(npu2)

// --- MemTile JOIN DMA section ---
// Two depth=2 JOIN intermediate buffers, each memref<128xbf16>.
// CHECK:     aie.buffer(%{{.*}}mem_tile_0_1) {{.*}}sym_name = "join_dst_buff_0"{{.*}} memref<128xbf16>
// CHECK:     aie.buffer(%{{.*}}mem_tile_0_1) {{.*}}sym_name = "join_dst_buff_1"{{.*}} memref<128xbf16>

// CHECK:     aie.memtile_dma(%{{.*}}mem_tile_0_1)

// --- S2MM JOIN ingest channel 0 (slice 0 from compute(0,2), offsets [0..64), len 64) ---
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.dma_bd(%{{.*}}join_dst_buff_0 : memref<128xbf16>, 0, 64)
// CHECK:       aie.dma_bd(%{{.*}}join_dst_buff_1 : memref<128xbf16>, 0, 64)

// --- S2MM JOIN ingest channel 1 (slice 1 = LAST slice from compute(0,3),
//     offsets [64..128), len MUST be 64 — buggy emit is 448) ---
// CHECK:       aie.dma_start(S2MM, 1
// CHECK:       aie.dma_bd(%{{.*}}join_dst_buff_0 : memref<128xbf16>, 64, 64)
// CHECK:       aie.dma_bd(%{{.*}}join_dst_buff_1 : memref<128xbf16>, 64, 64)

// --- MM2S JOIN send channel 0 (interleaves slice 0 + slice 1 across both
//     depth-2 buffers; the slice-1 BDs are the LAST-slice case and inherit
//     the same overflow on the buggy emit) ---
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.dma_bd(%{{.*}}join_dst_buff_0 : memref<128xbf16>, 0, 64)
// CHECK:       aie.dma_bd(%{{.*}}join_dst_buff_0 : memref<128xbf16>, 64, 64)
// CHECK:       aie.dma_bd(%{{.*}}join_dst_buff_1 : memref<128xbf16>, 0, 64)
// CHECK:       aie.dma_bd(%{{.*}}join_dst_buff_1 : memref<128xbf16>, 64, 64)

// --- Defense-in-depth: no dma_bd anywhere may write more than the 128-elem
//     memtile JOIN buffer. The exact buggy pattern is the 448-element write
//     starting at offset 64 (= 512 num_elems - 64 last-slice offset). ---
// CHECK-NOT: aie.dma_bd(%{{.*}}memref<128xbf16>{{.*}}, 448)

module @join_last_slice_bd_overflow {
  aie.device(npu2) {
    %shim_0  = aie.tile(0, 0)
    %mem_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @join_src_a (%tile_0_2, {%mem_0_1}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @join_src_b (%tile_0_3, {%mem_0_1}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @join_dst (%mem_0_1, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    // JOIN: 2 sources -> 1 destination through MemTile(0,1), per-source byte
    // offsets [0, 64] (each source contributes 64 bf16 to the 128-bf16 dst).
    aie.objectfifo.link [@join_src_a, @join_src_b] -> [@join_dst] ([0, 64][])

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @join_src_a(Produce, 1)
              : !aie.objectfifosubview<memref<64xbf16>>
      aie.objectfifo.release @join_src_a(Produce, 1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %0 = aie.objectfifo.acquire @join_src_b(Produce, 1)
              : !aie.objectfifosubview<memref<64xbf16>>
      aie.objectfifo.release @join_src_b(Produce, 1)
      aie.end
    }

    // Shim consumer: per-dispatch transfer of 512 bf16 (4× the per-slice
    // memtile buffer). --dma-task-to-conduit lowers the dma_bd transfer
    // length into conduit.get_memref_async num_elems = 512, which Pass C's
    // collect phase records as info.numElems for @join_dst. That value then
    // (incorrectly) becomes joinDstPerBufForLen for the JOIN last-slice BD
    // length calc.
    aie.runtime_sequence(%arg0: memref<512xbf16>) {
      %t = aiex.dma_configure_task_for @join_dst {
        aie.dma_bd(%arg0 : memref<512xbf16>, 0, 512) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
