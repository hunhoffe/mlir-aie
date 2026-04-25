//===- passC_drops_dma_repeat_1_on_output_configure_task_BUG.mlir *-MLIR-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// BUG PIN: --conduit-to-dma drops `repeat_count = 1` on the OUTPUT (S2MM)
// shim `aiex.dma_configure_task_for` when the source `conduit.create`
// carries `dma_repeat = 1`.
//
// This is the real root-cause of the GEMM @ attn_query 1.95M-wrong-rows
// (50%-zero) numerical bug observed on NPU after the shim-locks fix
// (`0ab9554ce5`).  IRON's GEMM output channel emits
// `aiex.dma_configure_task_for {issue_token = true, repeat_count = 1 : i32}`
// (= 2 total fires per call, IRON convention).  The pipeline:
//
//   1. Pass A (--objectfifo-to-conduit): SHIM channel sees emit.count = 1,
//      so per dc792ebbd5 it correctly SKIPS stamping `dma_repeat`.  ✓
//   2. --dma-task-to-conduit: surfaces IRON's `repeat_count = 1` onto the
//      conduit.create as `dma_repeat = 1` (existing test
//      `dma_task_to_conduit_propagates_repeat_count_s2mm.mlir` covers
//      this stage).  ✓
//   3. Pass C (--conduit-to-dma): ConduitToDMALower.cpp:1245 guard
//      `if (channelDmaRepeat > 1)` SKIPs emitting `repeat_count = 1` on
//      the output `aiex.dma_configure_task_for`.  ✗ — THE BUG.
//
// Convention: IRON / firmware read `repeat_count = N` as N+1 fires (see
// CLAUDE.md "Convention-divergence" entry; AIEDmaToNpu.cpp:180-183 packs
// the value verbatim into the NPU command word's repeat field).  So:
//   * stateful and IRON: repeat_count = 1 → 2 fires per call.  ✓
//   * conduit (current): no repeat_count attr → default 0 → 1 fire per
//     call.  Half the writes happen → 50% of output rows are zero.
//
// Diff captured at GEMM M=1024 K=2048 N=2048 8 cols num_invocations=16:
//   conduit:  `aiex.dma_configure_task_for @C_L2L3_0_shim_alloc ... ` +
//             `} {issue_token = true}`
//   stateful: `aiex.dma_configure_task_for @C_L2L3_0_shim_alloc ... ` +
//             `} {issue_token = true, repeat_count = 1 : i32}`
//
// Topology (single S2MM channel, compute-tile producer, shim consumer)
// is the minimal shape that exercises the dma_repeat=1 path without
// requiring a memtile or the full GEMM stack.  The same drop is observed
// on every C_L2L3_i shim alloc in the captured GEMM IR
// (/tmp/gemm_pattern_d_postfix_1777124585/build/.../input_with_addresses.mlir).
//
// FORWARD-FLIP NOTE (post-fix): the trailing CHECK-NOT becomes a
// CHECK-SAME that requires the literal `repeat_count = 1` substring
// once ConduitToDMALower.cpp:1245 is changed from `> 1` to `> 0`
// (or equivalently moves to the dma-task-to-conduit-style verbatim
// surfacing).  Test name and header should also drop the `_BUG`
// suffix at that point.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: aie.device(npu2)

// The output configure_task should carry `repeat_count = 1` to match
// stateful and IRON's intent (2 fires per call).  Today it does NOT.
// CHECK:       aiex.dma_configure_task_for @C_shim_alloc
// CHECK:         aie.dma_bd
// CHECK:       } {issue_token = true}
// CHECK-NOT:     repeat_count

module @passC_drops_dma_repeat_1_on_output_configure_task_BUG {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Compute → shim output channel: tile produces, shim consumes (S2MM).
    aie.objectfifo @C(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<4096xbf16>>

    func.func private @gemm_kernel(memref<4096xbf16>)

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %sub = aie.objectfifo.acquire @C(Produce, 1)
            : !aie.objectfifosubview<memref<4096xbf16>>
        %buf = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
        func.call @gemm_kernel(%buf) : (memref<4096xbf16>) -> ()
        aie.objectfifo.release @C(Produce, 1)
      }
      aie.end
    } {link_with = "gemm.c"}

    // IRON emits an OUTPUT-side configure_task with
    //   {issue_token = true, repeat_count = 1 : i32}
    // = 2 fires per call.  --dma-task-to-conduit surfaces the
    // `repeat_count = 1` onto the conduit.create as `dma_repeat = 1`
    // (verified by the existing s2mm propagation test).  Pass C should
    // re-emit `repeat_count = 1` on the output configure_task to match
    // stateful — but currently SKIPs it (the BUG this lit pins).
    aie.runtime_sequence(%arg0: memref<8192xbf16>) {
      %t = aiex.dma_configure_task_for @C {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 0, 4096) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}
