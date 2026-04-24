//===- passC_shim_bd_dma_repeat_no_placeholder.mlir ----------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Bug C / iter_count fix — refinement: when the source shim BD has fewer
// than 4 dims AND no leading <size=1, stride=0> placeholder available to
// fold into, Pass C falls back to PREPENDING <size=N, stride=0> as a new
// outermost dim (the original behavior from commit f651b752ec).  This
// keeps the dim count within the AIE2p 4-dim cap.
//
// Companion to passC_shim_bd_dma_repeat_folds_into_placeholder.mlir
// (which exercises the fold path) and passC_shim_bd_consumes_dma_repeat.mlir
// (which exercises the dim-less case).
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: aie.device(npu1)

// 3 source dims, no leading <1, 0> placeholder → prepend <size = 4,
// stride = 0> as a new outermost dim.  Total dim count = 4 (within cap).
// BD len scales by N: 48 * 4 = 192.
// CHECK:       aiex.dma_configure_task_for @chan_shim_alloc
// CHECK:         aie.dma_bd
// CHECK-SAME:    , 192
// CHECK-SAME:    <size = 4, stride = 0>
// CHECK-SAME:    <size = 3, stride = 32>
// CHECK-SAME:    <size = 8, stride = 4>
// CHECK-SAME:    <size = 2, stride = 1>
// CHECK:       aiex.dma_start_task

module @passC_shim_bd_dma_repeat_no_placeholder {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<48xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      // num_invocations = 4 → 4 acquires total → dma_repeat = 4.
      scf.for %i = %c0 to %c4 step %c1 {
        %sub = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<48xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<48xbf16>> -> memref<48xbf16>
        aie.objectfifo.release @chan (Consume, 1)
      }
      aie.end
    }

    // 3-dim BD with no <1, 0> placeholder anywhere.  Pass C must
    // PREPEND (not fold) because there's no slot to reuse, but room
    // remains under the 4-dim cap.  Note: 3 * 8 * 2 = 48 = numElems.
    aie.runtime_sequence(%a0: memref<48xbf16>) {
      %t = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%a0 : memref<48xbf16>, 0, 48,
            [<size = 3, stride = 32>,
             <size = 8, stride = 4>,
             <size = 2, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}
