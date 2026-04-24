//===- passC_shim_bd_dma_repeat_folds_into_placeholder.mlir --*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Bug C / iter_count fix — refinement: Pass C must fold `dma_repeat = N`
// into a leading <size=1, stride=0> placeholder dim instead of prepending a
// new outer dim, when the source shim BD already carries IRON's idiomatic
// 4-dim layout
//   [<1,0>, <1,0>, <1,0>, <innerSize, innerStride>].
//
// The earlier prepend-only implementation (commit f651b752ec) violated the
// AIE2/AIE2p 4-dim cap on data-layout transformations and was hard-rejected
// by the verifier with:
//   'aie.dma_bd' op At most four data layout transformation dimensions may
//   be provided.
//
// The fold path preserves the dim count at 4 while still scaling BD len by
// N (the placeholder previously contributed factor 1; now contributes
// factor N).  This is the path that fires for IRON-emitted Llama-scale BDs.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: aie.device(npu1)

// The shim BD for @chan must have its OUTERMOST <size = 1, stride = 0>
// placeholder rewritten to <size = 4, stride = 0> (the dma_repeat factor).
// The remaining two <1, 0> placeholder slots are preserved, and the inner
// <256, 1> walk is unchanged.  Total dim count still 4 (within the AIE2p
// hardware cap).  BD len scales by N: 256 * 4 = 1024.
// CHECK:       aiex.dma_configure_task_for @chan_shim_alloc
// CHECK:         aie.dma_bd
// CHECK-SAME:    , 1024
// CHECK-SAME:    <size = 4, stride = 0>
// CHECK-SAME:    <size = 1, stride = 0>
// CHECK-SAME:    <size = 1, stride = 0>
// CHECK-SAME:    <size = 256, stride = 1>
// CHECK:       aiex.dma_start_task

module @passC_shim_bd_dma_repeat_folds_into_placeholder {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<256xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      // num_invocations = 4 → 4 acquires total → dma_repeat = 4.
      scf.for %i = %c0 to %c4 step %c1 {
        %sub = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<256xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
        aie.objectfifo.release @chan (Consume, 1)
      }
      aie.end
    }

    // IRON-idiomatic 4-dim shim BD: 3 leading <1,0> placeholder slots
    // plus one inner <256, 1> walk.  Pass C must fold dma_repeat=4 into
    // the outermost placeholder rather than try to prepend a 5th dim.
    aie.runtime_sequence(%a0: memref<256xbf16>) {
      %t = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%a0 : memref<256xbf16>, 0, 256,
            [<size = 1, stride = 0>,
             <size = 1, stride = 0>,
             <size = 1, stride = 0>,
             <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}
