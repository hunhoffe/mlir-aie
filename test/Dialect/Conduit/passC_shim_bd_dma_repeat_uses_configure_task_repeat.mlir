//===- passC_shim_bd_dma_repeat_uses_configure_task_repeat.mlir *- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Bug C / iter_count fix — robustness pin: when the source shim BD already
// carries a 4-dim data layout (the IRON-idiomatic shape
//   [<1,0>, <1,0>, <1,0>, <innerSize, innerStride>]
// at the AIE2p 4-dim cap), Pass C must NOT mutate the BD's TAP to encode
// dma_repeat — it must surface dma_repeat exclusively via the
// `aiex.dma_configure_task_for.repeat_count` attribute.
//
// The earlier "prepend an outer <size=N, stride=0>" approach (commit
// f651b752ec) violated the 4-dim cap and the AIE verifier rejected the IR
// with "At most four data layout transformation dimensions may be provided."
// The configure_task.repeat_count path is hardware-native (lowers via
// NpuPushQueueOp.repeat_count → NPU command word) and is unaffected by the
// BD's dim count.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: aie.device(npu1)

// configure_task carries repeat_count = 4.  The BD's TAP is preserved
// verbatim (still 4 dims, still <1,0>x3 + <256,1> walk); BD len is the
// original 256, NOT scaled by 4.
// CHECK:       aiex.dma_configure_task_for @chan_shim_alloc
// CHECK:         aie.dma_bd
// CHECK-SAME:    , 256
// CHECK-SAME:    <size = 1, stride = 0>
// CHECK-SAME:    <size = 1, stride = 0>
// CHECK-SAME:    <size = 1, stride = 0>
// CHECK-SAME:    <size = 256, stride = 1>
// CHECK-NOT:     <size = 4, stride = 0>
// CHECK:       {{.*}}repeat_count = 4
// CHECK:       aiex.dma_start_task

module @passC_shim_bd_dma_repeat_uses_configure_task_repeat {
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
    // plus one inner <256, 1> walk.  Pass C must NOT touch this TAP —
    // it must apply dma_repeat=4 via configure_task.repeat_count.
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
