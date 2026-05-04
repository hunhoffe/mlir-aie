//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --verify-diagnostics --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-channels --conduit-to-dma %s
// Metafix Candidate 1 (Path C async, full pipeline): the second RUN line
// pushes the same input through the entire downstream legalization stack so
// any per-channel BD-pool exhaustion or dialect-verifier failure surfaces
// here, not on first NPU contact.
// RUN: aie-opt --verify-diagnostics --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-channels --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// SCOPE NOTE (#99 closure, 2026-05-01): the cross-producer-MM2S → shared-
// S2MM IR shape this fixture uses (two same-shim-tile MM2S channels both
// landing on (0,2) DMA:0 after fuse-channels folds the consumer-side S2MM
// port) is now diagnosed at Pass C as a duplicate-dst circuit-route
// infeasibility — see emitFlow in ConduitToDMACommon.cpp + the dedicated
// pin passc_dup_dst_feasibility_error.mlir.  Both RUN lines therefore now
// expect-error rather than producing post-Pass-C IR.  The original
// FileCheck-pinned roundtrip (inline frees in source-relative position)
// is no longer reachable via this IR shape — the FileCheck pins below are
// retained as documentation of the design intent for a single-source rt-
// seq variant that would still exercise the same wait_all{token=false}
// preservation invariant; a future fixture should re-pin that with a
// shape that doesn't trip the #99 check (e.g., a single MM2S channel, or
// distinct producer tiles on packet-routed conduits once the packet-flow
// path lands in Sprint N+4).  The error-pin below is intentional and
// preserves regression coverage for the IR shape's documented outcome.

// Path C async end-to-end roundtrip pin (Task #33, design from
// path-c-test-matrix.md §1.7 + cross-fusion §1.5).
//
// ORIGINAL goal (pre-#99): full Path A → fuse-channels → Path C composite,
// demonstrating that IRON-emitted dma_await_task / dma_free_task at the
// source level survive every stage of the pipeline as the per-launch
// release boundary (the roundtrip-identity-as-lit pin called out in
// CLAUDE.md "Working Conventions" and design doc §1.7).
//
// CURRENT scope (post-#99 closure): the chosen IR shape — two MM2S
// channels (@ext_a, @ext_b) on the SAME shim tile, sequential consumes
// on a shared compute tile — is exactly the cross-producer-MM2S → shared-
// S2MM duplicate-dst case that Pass C's emitFlow now diagnoses cleanly.
// The fixture is therefore retained as an error-pin for the documented
// infeasibility outcome on this shape.  Re-pinning the original wait_all
// {token=false} preservation invariant on a shape that does NOT trip the
// #99 check is tracked separately (single-MM2S variant or packet-routed
// distinct-producer variant once the packet-flow path lands).

module @path_c_async_passA_fuse_passC_roundtrip {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    // expected-error @below {{conduit-to-dma: cannot circuit-route distinct sources to (0,2)}}
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
