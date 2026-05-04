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
// repeat_count-bearing 1->1 link variant (op6_Repeat-shaped).
//
// Companion to:
//   - conduit_to_dma_link_lookup_unqualified_key_multi_device.mlir
//     (1->1 READ-side pin, baseline shape)
//   - conduit_to_dma_link_join_unqualified_write_multi_device.mlir
//     (JOIN write-side pin)
//   - conduit_to_dma_link_transpose_unqualified_write_multi_device.mlir
//     (1->1 transpose-shaped variant, op10_Transpose-like)
//
// This fixture mirrors the Llama decode op5_StridedCopy / op6_Repeat
// runlist ops: shim -> memtile -> shim, multi-device, with a
// `repeat_count = N` attribute on the producer ObjectFifo expressing
// the multi-fire BD pattern. linkPhase walks Scatter (single-dst);
// repeat_count lands on the post-Pass-C BD chain (multiple `aie.dma_bd`
// blocks unrolled per fire). The linkPhase write sites for this 1->1
// link still fire identically to the bare case.
//
// Therefore this fixture is expected to COLLAPSE structurally with the
// Round 1 read-side pin: pre-fix it fails on the same memtile
// `dma_start(S2MM, 1)` / `dma_start(MM2S, 1)` channel-mismatch
// signature; post-fix both pass. Retained as defense-in-depth for the
// repeat-bearing 1->1 link path.

// CHECK-LABEL: module @link_repeat_unqualified_write_multi_device

// FIRST device.
// CHECK: aie.device(npu2) @first
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
// CHECK: aie.memtile_dma(%mem_tile_0_1)
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_start(MM2S, 0

// SECOND device.
// CHECK: aie.device(npu2) @second
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
// CHECK: aie.memtile_dma(%mem_tile_0_1)
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_start(MM2S, 0

// Defense-in-depth: no memtile DMA chain may be on ch 1.
// CHECK-NOT: aie.dma_start(S2MM, 1
// CHECK-NOT: aie.dma_start(MM2S, 1

module @link_repeat_unqualified_write_multi_device {
  aie.device(npu2) @first {
    %shim = aie.tile(0, 0)
    %mem  = aie.tile(0, 1)

    aie.objectfifo @r_in  (%shim, {%mem},  2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    // repeat_count=2 unrolls 2 BD blocks per chain on the memtile->shim
    // leg; mimics op6_Repeat's multi-fire BD pattern.
    aie.objectfifo @r_out (%mem,  {%shim}, 2 : i32) {repeat_count = 2 : i32}
        : !aie.objectfifo<memref<32xbf16>>

    aie.objectfifo.link [@r_in] -> [@r_out] ([] [])

    aie.runtime_sequence(%arg0: memref<32xbf16>, %arg1: memref<32xbf16>) {
      %t_a = aiex.dma_configure_task_for @r_in {
        aie.dma_bd(%arg0 : memref<32xbf16>, 0, 32) {burst_length = 0 : i32}
        aie.end
      }
      %t_b = aiex.dma_configure_task_for @r_out {
        aie.dma_bd(%arg1 : memref<32xbf16>, 0, 32) {burst_length = 0 : i32}
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

    aie.objectfifo @r_in  (%shim, {%mem},  2 : i32)
        : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @r_out (%mem,  {%shim}, 2 : i32) {repeat_count = 2 : i32}
        : !aie.objectfifo<memref<32xbf16>>

    aie.objectfifo.link [@r_in] -> [@r_out] ([] [])

    aie.runtime_sequence(%arg0: memref<32xbf16>, %arg1: memref<32xbf16>) {
      %t_a = aiex.dma_configure_task_for @r_in {
        aie.dma_bd(%arg0 : memref<32xbf16>, 0, 32) {burst_length = 0 : i32}
        aie.end
      }
      %t_b = aiex.dma_configure_task_for @r_out {
        aie.dma_bd(%arg1 : memref<32xbf16>, 0, 32) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_a)
      aiex.dma_start_task(%t_b)
      aiex.dma_await_task(%t_a)
      aiex.dma_await_task(%t_b)
    }
  }
}
