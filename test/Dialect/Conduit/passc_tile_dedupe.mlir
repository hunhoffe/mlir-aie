//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --conduit-to-dma %s | FileCheck %s

// Pass C regression pin (#97 Fix B / H(c) verdict, Task #45): when the input
// IR contains DUPLICATE `aie.tile` SSA ops at the same (col, row) coord
// inside one `aie.device` (the post-`aie-combine-device same-tile=true` shape
// after fuse-core-bodies / fuse-operators erase one of the cores but leave
// the orphan tile SSA + ops referencing it behind), `--conduit-to-dma` MUST
// dedupe them BEFORE building its tileCache.  Pre-fix: orphan SSA wins the
// last-writer-wins tileCache (`ConduitToDMACollect.cpp:328`) → buffer-
// creation walk (`ConduitToDMAAlloc.cpp:324`) misses the surviving core via
// SSA inequality → rotation buffer never allocated → lowering emits
// "depth>1 buffer rotation requires a rotation counter" at the
// subview_access in the surviving merged core (#97).
//
// Post-fix: a pre-pass dedupe step (`dedupeTileOps` in `ConduitToDMAPass.cpp`)
// elects the FIRST-DEFINED tile op as canonical (matching `aie-combine-
// device`'s splice order: bodyA first → devA's tiles first → surviving cores
// were originally bound to devA), `replaceAllUsesWith(canonical)` on each
// orphan, and erases them.  Covers both compute and shim tiles uniformly
// since `aie.tile` is one op type for both.
//
// Fixture authored directly in post-Pass-A conduit IR (no objectfifo) so the
// duplicate-tile shape can survive (Pass A's `--objectfifo-to-conduit`
// verifier rejects asymmetric core/conduit tile bindings before Pass C even
// sees them).  Two `aie.tile(0, 0)` SSAs and two `aie.tile(0, 2)` SSAs are
// declared in canonical-then-orphan order; `aie.shim_dma_allocation` for the
// tile→shim conduit references the orphan shim (modeling
// `aie-combine-device`'s splice leaving an orphan shim_dma_allocation
// pointing at the orphan shim SSA — see forensics §"Pre-Pass-C IR shape").
// The orphan compute tile is unreferenced; its mere existence pollutes the
// tileCache via the unconditional last-writer-wins overwrite at
// ConduitToDMACollect.cpp:328.

module @passc_tile_dedupe {
  aie.device(npu2) {
    // ---- Canonical (first-defined) tile SSAs ----
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)

    // depth=2 shim→tile conduit.  Rotation needed (depth > 1).
    conduit.create @ext_in {
      element_type = memref<64xbf16>,
      depth = 2 : i64
    }
    aie.shim_dma_allocation @ext_in_shim_alloc(%shim_a, MM2S, 0)
        {conduit_channel = @ext_in}

    // ---- Orphan (post-aie-combine-device-same-tile=true) duplicates ----
    %shim_b = aie.tile(0, 0)
    %tile_b = aie.tile(0, 2)

    // depth=2 tile→shim conduit.  Models the second-device conduit whose
    // shim_dma_allocation got spliced into devA still pointing at the
    // orphan shim SSA.
    conduit.create @ext_out {
      element_type = memref<64xbf16>,
      depth = 2 : i64
    }
    aie.shim_dma_allocation @ext_out_shim_alloc(%shim_b, S2MM, 0)
        {conduit_channel = @ext_out}

    // Surviving merged core bound to the FIRST tile(0, 2) — matches the
    // post-fuse-core-bodies / post-fuse-operators state where the merged
    // core uses devA's original tile SSA.
    %core = aie.core(%tile_a) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      %cone = arith.constant 1.0 : bf16
      scf.for %niter = %c0 to %c4 step %c1 {
        %win_in = conduit.acquire {name = @ext_in,
            port = #conduit.port<Consume>, count = 1 : i64}
            : !conduit.window<memref<64xbf16>>
        %elem_in = conduit.subview_access %win_in {index = 0 : i64}
            : !conduit.window<memref<64xbf16>> -> memref<64xbf16>

        %win_out = conduit.acquire {name = @ext_out,
            port = #conduit.port<Produce>, count = 1 : i64}
            : !conduit.window<memref<64xbf16>>
        %elem_out = conduit.subview_access %win_out {index = 0 : i64}
            : !conduit.window<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_in[%j] : memref<64xbf16>
          %r = arith.addf %v, %cone : bf16
          memref.store %r, %elem_out[%j] : memref<64xbf16>
        }

        conduit.release %win_out {port = #conduit.port<Produce>, count = 1 : i64}
            : !conduit.window<memref<64xbf16>>
        conduit.release %win_in {port = #conduit.port<Consume>, count = 1 : i64}
            : !conduit.window<memref<64xbf16>>
      }
      aie.end
    }
  }
}

// CHECK-LABEL: aie.device(npu2)

// Exactly ONE aie.tile(0, 2) and ONE aie.tile(0, 0) survive post-dedupe.
// The CHECK directives below pin exact counts (the trailing CHECK-NOT
// directives at the bottom forbid any further matches anywhere in the IR).
// Pattern follows commit e3a8c0b139's dma_start exhaustiveness pin.
// CHECK-COUNT-1: aie.tile(0, 0)
// CHECK-COUNT-1: aie.tile(0, 2)

// Buffer + rotation counter must be allocated on the canonical compute
// tile.  Pass C names the rotation counter buffer with a generated suffix;
// the i32 memref shape distinguishes it from the bf16 data buffers.  Pre-
// fix the rotation counter buffer is missing entirely (the bug shape this
// pin guards against); post-fix it is present.
// CHECK-DAG: aie.buffer({{.*}}){{.*}}: memref<{{[0-9]+}}xi32>

// Hard negative on the diagnostic — the bug shape this fix prevents.
// CHECK-NOT: depth>1 buffer rotation requires a rotation counter

// File-scope tile-count exhaustiveness: forbid any further aie.tile(0, 2)
// or aie.tile(0, 0) op anywhere in the post-pass IR.
// CHECK-NOT: aie.tile(0, 2)
// CHECK-NOT: aie.tile(0, 0)
