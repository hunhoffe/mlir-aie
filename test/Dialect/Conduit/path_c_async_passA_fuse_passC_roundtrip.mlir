//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-channels --conduit-to-dma %s | FileCheck %s
// Metafix Candidate 1 (Path C async, full pipeline): the second RUN line
// pushes the same input through the entire downstream legalization stack so
// any per-channel BD-pool exhaustion or dialect-verifier failure surfaces
// here, not on first NPU contact.
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-channels --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// Path C async end-to-end roundtrip pin (Task #33, design from
// path-c-test-matrix.md §1.7 + cross-fusion §1.5).
//
// Goal: full Path A → fuse-channels → Path C composite, demonstrating that
// IRON-emitted dma_await_task / dma_free_task at the source level survive
// every stage of the pipeline as the per-launch release boundary.  This is
// the roundtrip-identity-as-lit pin called out in CLAUDE.md "Working
// Conventions" and design doc §1.7.
//
// Input shape:
//   - Two MM2S channels (@ext_a, @ext_b) on the same shim tile, each with
//     IRON-emitted per-iteration dma_free_task in source — the canonical
//     "preserve per-task_group release boundaries" pattern (CLAUDE.md
//     "Path C reference design" + commit e2ea0ab450 message).
//
// Expected output (final IR after --conduit-to-dma):
//   - aiex.dma_free_task ops appear in source-relative position (NOT all
//     trailing per Path B).  This pins that wait_all{token=false} survived
//     fuse-channels and lowered back to inline dma_free_task in Step 8g.
//   - aiex.dma_configure_task_for / dma_start_task / dma_free_task triples
//     interleave per channel — each free immediately after its start, NOT
//     batched at end of rt-seq.

// CHECK-LABEL: module @path_c_async_passA_fuse_passC_roundtrip

// Both channels' configures and starts must survive lowering.
// CHECK:       aie.runtime_sequence

// Channel A: configure → start → INLINE free (the per-launch release
// boundary that wait_all{token=false} preserves through fuse-channels).
// CHECK:           [[A0:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_a
// CHECK:           aiex.dma_start_task([[A0]])
// CHECK:           aiex.dma_free_task([[A0]])

// Channel B: configure → start → INLINE free.
// CHECK:           [[B0:%[a-zA-Z0-9_]+]] = aiex.dma_configure_task_for @ext_b
// CHECK:           aiex.dma_start_task([[B0]])
// CHECK:           aiex.dma_free_task([[B0]])

// No additional frees, awaits, or configures after the per-channel pairs —
// every configured task is released exactly once, inline, in source-relative
// position (no trailing-batch fallback per Path B, no missed releases).
// CHECK-NOT:       aiex.dma_free_task
// CHECK-NOT:       aiex.dma_await_task
// CHECK-NOT:       aiex.dma_configure_task_for

module @path_c_async_passA_fuse_passC_roundtrip {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // depth=1 so --conduit-fuse-channels actually triggers fusion (depth>1
    // emits "skipping S2MM fusion" and the wait_all{token=false} preservation
    // through fuse-channels + Pass C roundtrip this test pins isn't reachable).
    aie.objectfifo @ext_a(%shim, {%tile}, 1 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @ext_b(%shim, {%tile}, 1 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @kernel(memref<128xbf16>)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        // Sequential consume of @ext_a then @ext_b — non-overlapping live
        // intervals on the same compute tile (the precondition fuse-channels
        // analysis would use to color them into one group).
        %ina = aie.objectfifo.acquire @ext_a(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %ina_buf = aie.objectfifo.subview.access %ina[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @kernel(%ina_buf) : (memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_a(Consume, 1)

        %inb = aie.objectfifo.acquire @ext_b(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %inb_buf = aie.objectfifo.subview.access %inb[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @kernel(%inb_buf) : (memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_b(Consume, 1)
      }
      aie.end
    } {link_with = "kernel.a"}

    aie.runtime_sequence(%arg0: memref<256xbf16>) {
      // Per-task_group release for @ext_a (becomes wait_all{token=false}
      // through Pass A; lowers back to inline aiex.dma_free_task through
      // Pass C Step 8g).
      %ta = aiex.dma_configure_task_for @ext_a {
        aie.dma_bd(%arg0 : memref<256xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta)
      aiex.dma_free_task(%ta)

      // Per-task_group release for @ext_b.
      %tb = aiex.dma_configure_task_for @ext_b {
        aie.dma_bd(%arg0 : memref<256xbf16>, 128, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tb)
      aiex.dma_free_task(%tb)
    }
  }
}
