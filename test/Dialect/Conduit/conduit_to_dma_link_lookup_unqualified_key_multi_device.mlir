//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// Pass C linkPhase device-qualified-key lookup regression pin.
//
// Bug location: ConduitToDMALink.cpp memtile distribute/join path.
//   - line 280:  CoreTile relay relaySrcS2MMCh (same shape, non-memtile path)
//   - line 682:  ingestS2MMCh lookup  state.conduitConsS2MMChannel.find({srcName, 0u})
//   - line 700:  distMM2SChannels[i]  state.conduitMM2SChannel.find(dstName)
//   - line 723:  joinS2MMChannels[i]  state.conduitConsS2MMChannel.find({srcName, ...})
//   - line 738:  joinS2MMChannels write under unqualified srcName key
//   - line 750:  joinMM2SCh           state.conduitMM2SChannel.find(dstName)
//
// Root cause: in multi-device pipelines, Pass C qualifies the conduitMap
// key with `__d<deviceIndex>` via state.makeConduitKey() (see
// ConduitToDMACommon.cpp:426-435). Phase 4a (route shim producer) and
// Phase 4b (route shim consumer) iterate `conduitMap` and write the
// per-conduit channel maps `conduitConsS2MMChannel` / `conduitMM2SChannel`
// under the QUALIFIED iteration key. They also bump the per-tile DMA
// channel counters `tileNextS2MMChannel[memtile]` /
// `tileNextMM2SChannel[memtile]`.
//
// Then linkPhase processes the conduit.scatter / conduit.gather, pulls
// the IR symbol ref via `mlir::cast<FlatSymbolRefAttr>(...).getValue()`
// (UNQUALIFIED — the IR-level conduit name), and looks up the channel
// maps with that unqualified key. The lookup MISSES, falls through to
// `state.tileNextS2MMChannel[memtileVal]++` / `tileNextMM2SChannel[...]++`
// which return the post-Phase-4-bump value (1 instead of the intended 0).
// linkPhase then emits `aie.dma_start(S2MM, 1, ...)` on the memtile
// while Phase 4a's `aie.flow(...DMA : 0, memtile, DMA : 0)` declares ch 0.
// Channel mismatch -> data delivered on ch 0 to unprogrammed S2MM, the
// memtile S2MM ch 1 BD chain blocks forever waiting on ch 1 data
// -> ert_cmd_state.ERT_CMD_STATE_TIMEOUT (Llama decode hang).
//
// The bug is multi-device-only: in single-device mode makeConduitKey
// returns name as-is, so qualified key == unqualified key and the lookup
// hits. Most existing Conduit lit fixtures are single-device.
//
// Topology (minimum reproducer of Llama decode op5_StridedCopy /
// op6_Repeat / op10_Transpose pattern):
//
//   aie.device(npu2) @first {
//     shim(0,0) ──[op_X, depth=2]──> memtile(0,1) ──[op_Y, depth=2]──> shim(0,0)
//     aie.objectfifo.link [@op_X] -> [@op_Y] ([] [])
//   }
//   aie.device(npu2) @second {        // identical, second device triggers
//     ...                              // multi-device qualification.
//   }
//
// 1->1 ObjectFifo Link lowers to ScatterOp with single dst. linkPhase's
// memtile distribute path runs (ingestS2MMCh + distMM2SChannels[0]).
//
// CHECK discipline (USER-LOCKED 2026-04-27): fail-now / pass-after-fix.
// CHECK lines pin the CORRECT post-fix behavior:
//   - aie.dma_start(S2MM, 0, ...) on memtile (matches Phase 4a flow ch 0)
//   - aie.dma_start(MM2S, 0, ...) on memtile (matches Phase 4b flow ch 0)
// On current HEAD this FAILS — linkPhase emits ch 1 / ch 1 (the bug).
// Post-fix it PASSES — qualified-key lookup hits, returns the Phase-4
// allocation (ch 0), and linkPhase emits matching ch 0 / ch 0.
// Both devices must be repaired by any fix; both are pinned below.
//
// Discovered 2026-04-26 during Llama decode hang investigation
// (Task #3, post the multi-device-rotation fix c09973b389).

// CHECK-LABEL: module

// FIRST device (deviceIndex = 0). Phase 4a/4b emit the flows at ch 0/0
// on memtile (these CHECKs pass today and post-fix; document the
// expected wiring that linkPhase must match).
// CHECK: aie.device(npu2) @first
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
// linkPhase emits memtile_dma. Ordering: S2MM block first, MM2S second
// (matches current emission order; preserved post-fix).
// CHECK: aie.memtile_dma(%mem_tile_0_1)
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_start(MM2S, 0

// SECOND device (deviceIndex = 1). Same pin with `__d1` qualification
// in the bug; same correct-channel post-fix.
// CHECK: aie.device(npu2) @second
// CHECK-DAG: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
// CHECK-DAG: aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
// CHECK: aie.memtile_dma(%mem_tile_0_1)
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_start(MM2S, 0

module @link_lookup_unqualified_key_multi_device {
  aie.device(npu2) @first {
    %shim = aie.tile(0, 0)
    %mem  = aie.tile(0, 1)

    aie.objectfifo @op_X (%shim, {%mem},  2 : i32) : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @op_Y (%mem,  {%shim}, 2 : i32) : !aie.objectfifo<memref<32xbf16>>

    // 1->1 link: shim -> memtile -> shim. Pass A lowers to single-dst
    // ScatterOp. linkPhase memtile distribute path fires, exhibits the
    // bug.
    aie.objectfifo.link [@op_X] -> [@op_Y] ([] [])

    aie.runtime_sequence(%arg0: memref<32xbf16>, %arg1: memref<32xbf16>) {
      %t_in = aiex.dma_configure_task_for @op_X {
        aie.dma_bd(%arg0 : memref<32xbf16>, 0, 32) {burst_length = 0 : i32}
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @op_Y {
        aie.dma_bd(%arg1 : memref<32xbf16>, 0, 32) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_in)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_in)
      aiex.dma_await_task(%t_out)
    }
  }

  aie.device(npu2) @second {
    %shim = aie.tile(0, 0)
    %mem  = aie.tile(0, 1)

    aie.objectfifo @op_X (%shim, {%mem},  2 : i32) : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @op_Y (%mem,  {%shim}, 2 : i32) : !aie.objectfifo<memref<32xbf16>>

    aie.objectfifo.link [@op_X] -> [@op_Y] ([] [])

    aie.runtime_sequence(%arg0: memref<32xbf16>, %arg1: memref<32xbf16>) {
      %t_in = aiex.dma_configure_task_for @op_X {
        aie.dma_bd(%arg0 : memref<32xbf16>, 0, 32) {burst_length = 0 : i32}
        aie.end
      }
      %t_out = aiex.dma_configure_task_for @op_Y {
        aie.dma_bd(%arg1 : memref<32xbf16>, 0, 32) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_in)
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_in)
      aiex.dma_await_task(%t_out)
    }
  }
}
