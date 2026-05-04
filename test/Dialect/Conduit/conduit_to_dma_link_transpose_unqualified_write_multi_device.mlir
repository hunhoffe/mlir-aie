//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// Pass C linkPhase device-qualified-key WRITE-side regression pin —
// transpose-shaped 1->1 link variant.
//
// Companion to:
//   - conduit_to_dma_link_lookup_unqualified_key_multi_device.mlir
//     (1->1 READ-side pin, baseline shape — bare dims)
//   - conduit_to_dma_link_join_unqualified_write_multi_device.mlir
//     (JOIN write-side pin — exercises L967 / L743)
//
// This fixture mirrors the Llama decode op10_Transpose runlist op:
// shim -> memtile -> shim, multi-device, with a `dimensionsToStream`
// stride-pattern on the memtile->shim leg expressing the transpose
// readout. linkPhase only walks Scatter/Gather (NOT TransposeOp), so
// the dims attribute lands on the BD layout but does not change which
// linkPhase write sites fire vs. the Round 1 baseline shape.
//
// Therefore this fixture is expected to COLLAPSE structurally with
// the Round 1 read-side pin: pre-fix it fails on the same memtile
// `dma_start(S2MM, 1)` / `dma_start(MM2S, 1)` channel-mismatch
// signature; post-fix both pass. It is retained as
// defense-in-depth for the dims-bearing 1->1 link path because the
// dims attr is one of the most common variations in real Llama IR
// (op10_Transpose), and any future regression that breaks the dims
// path independently of the bare path would be caught here.

// CHECK-LABEL: module @link_transpose_unqualified_write_multi_device

// FIRST device. Memtile flows declare ch 0/0 (matches Phase 4a/4b
// allocation). linkPhase memtile distribute path must emit BD chains
// on ch 0 (post-fix). Pre-fix: emits ch 1 (write-then-failed-lookup
// fallback bumps to next-free).
// CHECK: aie.device(npu2) @first
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
// CHECK: aie.memtile_dma(%mem_tile_0_1)
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_start(MM2S, 0

// SECOND device. Identical pin under `__d1` qualification.
// CHECK: aie.device(npu2) @second
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
// CHECK: aie.memtile_dma(%mem_tile_0_1)
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_start(MM2S, 0

// Defense-in-depth: no memtile DMA chain may be on ch 1 (the buggy
// emit signature).
// CHECK-NOT: aie.dma_start(S2MM, 1
// CHECK-NOT: aie.dma_start(MM2S, 1

module @link_transpose_unqualified_write_multi_device {
  aie.device(npu2) @first {
    %shim = aie.tile(0, 0)
    %mem  = aie.tile(0, 1)

    // Input: bare 32xbf16 from shim into memtile.
    aie.objectfifo @t_in  (%shim, {%mem},  2 : i32)
        : !aie.objectfifo<memref<32xbf16>>

    // Output: memtile -> shim with a 2D non-contiguous read pattern
    // (4 groups of 4 bf16 contiguous, stride 8 between groups; mimics
    // op10_Transpose's strided readout). Inner-most stride = 1
    // satisfies the 32b-datatype BD verifier; outer stride = 8 makes
    // the DMA non-trivial. dimensionsToStream lands on the memtile
    // MM2S BD layout post-Pass-C; linkPhase write sites for this 1->1
    // link still fire identically to the bare case.
    aie.objectfifo @t_out (%mem  dimensionsToStream [<size = 4, stride = 8>, <size = 4, stride = 1>],
                          {%shim}, 2 : i32)
        : !aie.objectfifo<memref<16xbf16>>

    aie.objectfifo.link [@t_in] -> [@t_out] ([] [])

    aie.runtime_sequence(%arg0: memref<32xbf16>, %arg1: memref<16xbf16>) {
      %t_a = aiex.dma_configure_task_for @t_in {
        aie.dma_bd(%arg0 : memref<32xbf16>, 0, 32) {burst_length = 0 : i32}
        aie.end
      }
      %t_b = aiex.dma_configure_task_for @t_out {
        aie.dma_bd(%arg1 : memref<16xbf16>, 0, 16) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_a)
      aiex.dma_start_task(%t_b)
      aiex.dma_await_task(%t_a)
      aiex.dma_await_task(%t_b)
    }
  }

  aie.device(npu2) @second {
    %shim = aie.tile(0, 0)
    %mem  = aie.tile(0, 1)

    aie.objectfifo @t_in  (%shim, {%mem},  2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @t_out (%mem  dimensionsToStream [<size = 4, stride = 8>, <size = 4, stride = 1>],
                          {%shim}, 2 : i32)
        : !aie.objectfifo<memref<16xbf16>>

    aie.objectfifo.link [@t_in] -> [@t_out] ([] [])

    aie.runtime_sequence(%arg0: memref<32xbf16>, %arg1: memref<16xbf16>) {
      %t_a = aiex.dma_configure_task_for @t_in {
        aie.dma_bd(%arg0 : memref<32xbf16>, 0, 32) {burst_length = 0 : i32}
        aie.end
      }
      %t_b = aiex.dma_configure_task_for @t_out {
        aie.dma_bd(%arg1 : memref<16xbf16>, 0, 16) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_a)
      aiex.dma_start_task(%t_b)
      aiex.dma_await_task(%t_a)
      aiex.dma_await_task(%t_b)
    }
  }
}
