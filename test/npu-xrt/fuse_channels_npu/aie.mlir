//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// HW smoke for `--conduit-fuse-channels` (channel-fusion annotation pass).
// Lit-style e2e companion to lit-only pin
//   test/Dialect/Conduit/conduit_fuse_channels.mlir
// and Python-harness sibling
//   iron_operator_mlir/fusion_npu_regression/fuse_channels_smoke.py.
//
// SCOPE NOTE (post-#99 closure, 2026-05-01): this fixture's HW coverage is
// the @inter / @ext_out producer-side fold ONLY (single shim input + single
// shim output, with one self-loop intermediate).  The earlier two-shim-input
// shape (@ext_in_a + @ext_in_d both shim→tile(0,2)) was migrated to the
// pure-lit pin
//   test/Dialect/Conduit/passc_dup_dst_feasibility_error.mlir
// because its consumer-side S2MM fold tripped Pass C's new duplicate-dst
// feasibility check (#99) — the two shim-MM2S sources colliding at the same
// (tile(0,2), DMA:0) S2MM port now error at Pass C, before HW.  Pure-lit
// pin covers that error path; this HW smoke covers the producer-side fold
// that still lowers cleanly.
//
// The fuse-channels pass is ANNOTATION-ONLY: it walks producer-side conduits
// on the same tile that occupy non-overlapping windows in the same basic
// block, stamps `dma_channel_group="groupN"` + `fuse_mode="static"` on each
// member of a group, and otherwise leaves IR shape unchanged.  Pass C
// downstream consumes the annotation to fold the grouped conduits onto a
// single HW DMA channel.  Output bytes MUST be byte-identical to a bare
// `--use-conduit` pipeline (no fuse-channels flag) on the same input — the
// reference is computed in test.cpp and the pass-vs-bypass diff is what the
// HW smoke proves.
//
// IR shape required by the pass: TWO producer-side conduits on the SAME tile
// in the SAME basic block, with non-overlapping acquire/release windows.
// This fixture uses a single device with a single core on tile(0,2) running
// an unbounded outer `scf.for` whose body contains two sequential acquire/
// compute/release sections:
//   * Add section:  acquire @ext_in_a (Consume), acquire @inter (Produce),
//                   inline `add 1.0`, release both.
//   * Mul section:  acquire @inter (Consume), acquire @ext_out (Produce),
//                   inline `multiply by 2.0`, release.
// The two PRODUCE-side conduits on tile(0,2) — `@inter` and `@ext_out` —
// are the fuse-channels candidates: same producer tile, same parent block,
// disjoint live windows (Add's `@inter` release happens before Mul's
// `@ext_out` acquire).  `@inter` is a self-loop (tile→tile) so its consumer
// path also lives on the same tile, but only the producer-side window is
// what the pass groups.  Single-column smoke uses `npu2_1col`.
//
// Why one device (not two like #73/#74): `--conduit-fuse-channels` runs
// after `--use-conduit` lowers ObjectFifos to Conduit IR; it does NOT
// require the spatial-fusion infrastructure that collapses separate
// `aie.device` blocks.  A single device with co-located producer conduits
// is the minimal IR shape that exercises the annotation pass.  No
// `fusion_group=` attribute is set anywhere — fuse-channels groups by
// producer-tile + parent-block + window-disjointness, NOT by user-supplied
// fusion_group tags (those are for fuse-operators / fuse-core-bodies).
//
// Compute is inlined (no `func.call` to external kernels) so the test is
// self-contained — no peano/chess link step required.
//
// Reference (test.cpp computes this):
//   in_a[j] = (j % 16),  out[j] = (in_a[j] + 1.0) * 2.0.
// Choice keeps every value bf16-exact: max value (15+1)*2 = 32 ≤ 256, and
// integers in [0, 256) are exact in bf16.
//
// PASS expectation: every dispatch (4 invocations) completes; output bytes
// match reference.  Failure modes specific to the annotation pass (per
// fuse_channels_smoke.py docstring) are detailed in conduit.lit header.

module {
  // One device, one compute tile, three conduits.  The two producer-side
  // conduits ON tile(0,2) — @inter and @ext_out — are the fuse-channels
  // grouping candidates.  @ext_in_a is produced by the shim (different tile),
  // so it is not part of this group.
  aie.device(NPUDEVICE) {

    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // Input from shim (consumer-side on tile(0,2)):
    // Depth=1 per fuse-channels' Tier-3-depth=1 design intent (commit
    // 14eb385272: depth>1 + Tier 3 puts/gets is declined by the pass +
    // would violate BD-ring ordering when chained).
    aie.objectfifo @ext_in_a (%shim, {%tile}, 1 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Intermediate self-loop on tile(0,2) — Add produces, Mul consumes.
    // Producer-side window: Add section's acquire@inter(Produce).
    aie.objectfifo @inter (%tile, {%tile}, 1 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Output to shim (producer-side on tile(0,2)).
    // Producer-side window: Mul section's acquire@ext_out(Produce).
    aie.objectfifo @ext_out (%tile, {%shim}, 1 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    %core = aie.core(%tile) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      // NUM_INVOCATIONS in test.cpp is 4 (line 63); core acquires must
      // match host dispatch count or core blocks waiting for buffers
      // that never arrive (XRT timeout / status 8). Fixture as authored
      // at 4c0611e4e6 had %cmax = 0xFFFFFE = 16,777,214 — Pass A's
      // dma_repeat inference (ObjectFifoToConduit.cpp:1200-1203) reads
      // this directly into @inter's dma_repeat, blocking core on the
      // 5th iteration. Pre-#99 the fixture failed at compile time
      // (aie-routing duplicate-dst on the multi-source-S2MM case);
      // #99 closure (commit 4012ed56be) exposed the latent runtime
      // bound bug. Bound to NUM_INVOCATIONS via constant.
      %cmax = arith.constant 4 : index
      %cone = arith.constant 1.0 : bf16
      %ctwo = arith.constant 2.0 : bf16
      scf.for %niter = %c0 to %cmax step %c1 {

        // ---- Add section: ext_in_a + 1.0 -> inter ----
        %sv_a = aie.objectfifo.acquire @ext_in_a (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_a = aie.objectfifo.subview.access %sv_a[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_int_p = aie.objectfifo.acquire @inter (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_int_p = aie.objectfifo.subview.access %sv_int_p[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_a[%j]      : memref<64xbf16>
          %r = arith.addf %v, %cone         : bf16
          memref.store %r, %elem_int_p[%j]  : memref<64xbf16>
        }

        aie.objectfifo.release @inter     (Produce, 1)
        aie.objectfifo.release @ext_in_a  (Consume, 1)

        // ---- Mul section: inter * 2.0 -> ext_out ----
        // Sequential after Add section's releases, so @inter (consume side)
        // and @ext_out (produce side) windows are disjoint from the
        // preceding @inter (produce side) window — fuse-channels eligibility
        // is decided on producer-side windows in the parent block.
        %sv_int_c = aie.objectfifo.acquire @inter (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_int_c = aie.objectfifo.subview.access %sv_int_c[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_o = aie.objectfifo.acquire @ext_out (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_o = aie.objectfifo.subview.access %sv_o[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %vi = memref.load %elem_int_c[%j] : memref<64xbf16>
          %r  = arith.mulf %vi, %ctwo       : bf16
          memref.store %r, %elem_o[%j]      : memref<64xbf16>
        }

        aie.objectfifo.release @ext_out   (Produce, 1)
        aie.objectfifo.release @inter     (Consume, 1)
      }
      aie.end
    }

    aie.runtime_sequence @addmul_seq(%a_in  : memref<64xbf16>,
                                     %o_out : memref<64xbf16>) {
      %ta_a = aiex.dma_configure_task_for @ext_in_a {
        aie.dma_bd(%a_in : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta_a)
      %ta_o = aiex.dma_configure_task_for @ext_out {
        aie.dma_bd(%o_out : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ta_o)
      aiex.dma_await_task(%ta_o)
      aiex.dma_free_task(%ta_a)
    }
  }
}
