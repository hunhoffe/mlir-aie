//===- passC_shim_bd_consumes_dma_repeat.mlir ----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Bug C / iter_count fix: Pass C must consume `dma_repeat = N` from the
// source conduit.create on the shim BD it emits in `aiex.dma_configure_task_for`.
// Pass A's iter_count inference (Track 1) stamps `dma_repeat = N` on
// shim-facing channels when the per-core outer loop fires the BD chain N
// times per host dispatch (e.g. num_invocations=4 with cores running
// while_true=False produces dma_repeat=4).  Without consuming it here, each
// shim dispatch only fires the BD once and the cores stall after exhausting
// 1/N-th of the work — Bug C root cause for the .bin runtime path.
//
// Option A (TAP placeholder idiom) — implemented in ConduitToDMALower.cpp
// Step 8g: prepend an outermost <size=N, stride=0> dim and scale BD len by
// N.  The shim DMA then walks the same buffer N times per task fire,
// matching the per-core acquire count.  Mirrors how IRON itself emits
// repeat-style BDs (FS5 fix history) and the memtile DMAStartOp.repeat_count
// = N-1 path in ConduitToDMALink.cpp around line 1804.
//
// Geometry:
//   * Single core, single scf.for(0, 4) → 4 acquires per channel.
//   * Host runtime_sequence emits ONE BD per channel; BD len = 64 elements
//     (one fifo tile).  Fifo elem-type memref<64xbf16> → acquires_per_BD = 1.
//   * Three-factor formula: dma_repeat = (4 / 1) / 1 = 4.
//   * Source BD is 1D (no dims) so Pass C synthesizes a unit-stride inner
//     dim of size=64 and prepends the outer <size=4, stride=0> placeholder.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: aie.device(npu1)

// The shim BD for @chan must carry an outer <size = 4, stride = 0> dim
// (the dma_repeat=4 placeholder) followed by the inner <size = 64, stride = 1>
// walk; total len = 4 * 64 = 256 elements per task fire.
// CHECK:       aiex.dma_configure_task_for @chan_shim_alloc
// CHECK:         aie.dma_bd
// CHECK-SAME:    , 256
// CHECK-SAME:    <size = 4, stride = 0>
// CHECK-SAME:    <size = 64, stride = 1>
// CHECK:       aiex.dma_start_task

module @passC_shim_bd_consumes_dma_repeat {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      // num_invocations = 4 → 4 acquires total.
      scf.for %i = %c0 to %c4 step %c1 {
        %sub = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        aie.objectfifo.release @chan (Consume, 1)
      }
      aie.end
    }

    // Single dim-less BD: len = 64 elements per fire (one fifo tile).
    // Pass A infers dma_repeat = 4; Pass C must apply it as outer
    // <size=4, stride=0> on the shim BD with len scaled to 256.
    aie.runtime_sequence(%a0: memref<64xbf16>) {
      %t = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%a0 : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}
