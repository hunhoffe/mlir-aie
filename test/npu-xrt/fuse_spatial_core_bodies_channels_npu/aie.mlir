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
// HW smoke for the FULL HYBRID COMPOSITION `--conduit-fuse-spatial` +
// `--conduit-fuse-core-bodies` + `--conduit-fuse-channels`.  Sibling-pair to
// the single-pass smokes:
//   test/npu-xrt/fuse_operators_basic_npu/   (--conduit-fuse-spatial alone,
//                                             1:1 Add→Mul, single output per
//                                             core)
//   test/npu-xrt/fuse_core_bodies_npu/       (--conduit-fuse-core-bodies
//                                             alone, 1:1 Add→Mul, single
//                                             output per core)
//   test/npu-xrt/fuse_channels_npu/          (--conduit-fuse-channels alone,
//                                             hand-authored single-device
//                                             single-core IR shape)
// And companion to the two-pass composition fixtures:
//   test/npu-xrt/fuse_channels_after_core_bodies_npu/  (compose-A: core-bodies
//                                             + channels — landed with
//                                             documented WEAKER no-op outcome
//                                             on cloned core-bodies source IR)
//   test/npu-xrt/fuse_spatial_and_channels_npu/        (compose-B: spatial +
//                                             channels — landed with stronger
//                                             "fuse-channels-with-work" signal
//                                             on a Section A + Section B
//                                             two-output topology, same-tile
//                                             =false routing offsets devMul to
//                                             column 1)
//
// Why a separate full-hybrid fixture: the two-pass companions each leave
// ONE pass un-exercised in the same pipeline.  Compose-A
// (`fuse_channels_after_core_bodies_npu`) doesn't run `--conduit-fuse-
// spatial`; compose-B (`fuse_spatial_and_channels_npu`) doesn't run
// `--conduit-fuse-core-bodies`.  Neither answers the production-pipeline
// question: does a topology where ALL THREE passes do meaningful work in
// one pipeline actually compile + run on hardware?  This fixture extends
// compose-B's source IR (Section A + Section B in devMul's core) with the
// SAME `fusion_group` annotation that compose-A used to drive
// `--conduit-fuse-core-bodies`'s Step 0 cross-device channel pair unify
// (verified by `ConduitFuseCoreBodyPass.cpp:74-155` — the intermediate pair
// is matched by fusion_group regardless of which pass injects the
// aie-combine-device step).
//
// Direct precursor to swiglu functional NPU smoke (Task #12 / fuse_hybrid_
// swiglu_npu) which adds K=2 convergent merge + 4-op shape on top of this
// 2-op minimum-viable hybrid.
//
// Source IR shape (clone of compose-B's Section A + Section B topology with
// fusion_group key renamed `addmul_spatial_chan` → `addmul_hybrid` to match
// per-fixture naming style; two `aie.device` blocks, both targeting
// `tile(0,2)`, intermediate channel pair tagged matching `fusion_group=
// "addmul_hybrid"`):
//
//   devAdd:  shim(0,0) → @ext_in_add → core_a on tile(0,2) → @inter_add
//                                                            (tile→shim,
//                                                             depth=2,
//                                                             fusion_group=
//                                                              "addmul_hybrid")
//   devMul:  shim(0,0) → @consume_add (shim→tile, depth=2,
//                                       fusion_group="addmul_hybrid")
//                       → core_m on tile(0,2):
//                              Section A: consume → mul → @ext_out_mul (tile→shim)
//                              Section B: write 7.0 const → @ext_out_aux (tile→shim)
//
// Pipeline behavior (RUN line: `--conduit-fuse-spatial
// --conduit-fuse-core-bodies-flag --conduit-fuse-channels-flag`):
//
//   1. `--use-conduit` lowers ObjectFifo → Conduit IR (Pass A).
//   2. `--conduit-fuse-spatial` injects `--conduit-fuse-operators` per
//      `aiecc.cpp:332`.  Critically, because `--conduit-fuse-core-bodies-
//      flag` is ALSO set, `aiecc.cpp:1509-1511` routes `--aie-combine-
//      device` with `same-tile=true` (NOT compose-B's `same-tile=false`),
//      so devMul's tiles are NOT offset to column 1 — instead BOTH cores
//      collapse onto tile(0,2) on column 0.  `--conduit-fuse-operators`
//      then erases the intermediate conduit pair (`@inter_add` /
//      `@consume_add`, both tagged `fusion_group="addmul_hybrid"`) and
//      replaces it with a single `@fused_intermediate_0` on-tile channel.
//   3. `--conduit-fuse-core-bodies` runs.  Step 0 (cross-device fusion_group
//      unify) is a no-op since fuse-operators in step 2 already erased the
//      pair; Steps 2-5 merge the two per-iteration loop bodies into ONE
//      core on tile(0,2): consume @ext_in_add → add 1.0 (L1 alloc) →
//      mul 2.0 → release @ext_out_mul → write 7.0 → release @ext_out_aux.
//      The intermediate L1 alloc replaces the on-tile channel.
//   4. `--conduit-fuse-channels` runs on the merged-device merged-core IR.
//      Producer-side conduits on tile(0,2): `@ext_out_mul` and
//      `@ext_out_aux`, sequential in the same outer-`scf.for` body, with
//      disjoint windows.  EMPIRICALLY (recorded post aie-opt validation):
//      fuse-channels is a NO-OP here — it skips both with the diagnostic
//      "Tier 3 channel with depth>1 not supported in fuse groups".  The
//      depth=2 producer-side conduits are not currently eligible under
//      Tier 3 grouping in the merged-onto-same-tile shape.  See conduit.lit
//      OBSERVED OUTCOME for the full record.
//   5. `--conduit-to-dma` lowers to AIE dialect ops the rest of aiecc consumes.
//
// The value is in proving all three passes can run in sequence on the SAME
// source IR without crashing, mis-routing, or producing wrong bytes — even
// when only ONE pass (fuse-core-bodies) does meaningful structural work on
// this minimum-viable shape.  See the conduit.lit OBSERVED OUTCOME for the
// full empirical record (post-aie-opt validation 2026-05-01).
//
// Why no `conduit.wait_all` consumers: the related MED bug at
//   test/Dialect/Conduit/path_c_async_fuse_corebody_blocks_at_wait_all.mlir
// fires when fusion-then-Pass-C IR carries `wait_all` consumers.  This
// fixture uses ordinary `aie.objectfifo.acquire` / `.release` /
// `.subview.access` (which Pass A lowers to `conduit.acquire` /
// `conduit.release` / `conduit.subview_access`), so the wait_all path is
// not exercised here.
//
// Why no cross-element-type fusion: cross-element-type fan-in is scoped out
// per the user-locked design (CLAUDE.md §"Locked design decisions").  All
// channels here carry `memref<64xbf16>`.
//
// Compute is inlined (no `func.call` to external kernels) so the test is
// self-contained — no peano/chess link step required for the kernel side.
// (The peano REQUIRES line still applies because Chess cannot select
// `fp_to_bf16` for the inline core; see conduit.lit header.)
//
// Reference (test.cpp computes this):
//   in[j] = (j % 16),  out_mul[j] = (in[j] + 1.0) * 2.0,
//                       out_aux[j] = 7.0 (constant, all elements).
// Choice keeps every value bf16-exact: max value (15+1)*2 = 32 ≤ 256 and
// 7.0 ≤ 256, integers in [0, 256) are exact in bf16.
//
// PASS expectation: every dispatch (4 invocations) completes; output bytes
// for both outputs match reference, and (separately, by construction) the
// out_mul bytes match the basic-spatial sibling on the same input.  Failure
// mode classes:
//   * Compile-time crash inside fuse-core-bodies triggered by fuse-spatial's
//     same-tile=true output IR shape.
//   * Compile-time crash inside fuse-channels triggered by the
//     fuse-core-bodies-merged single-core output (the question this fixture
//     is designed to answer).
//   * Hang on the second invocation = Pattern E forward-chain mis-fusion
//     across the merged body in multi-invocation, OR same-tile lock
//     interaction at the merged-body boundary.
//   * Wrong bytes mid-buffer = fuse-channels mis-grouping the surviving
//     producer-side conduits on the merged tile (e.g., reading
//     out_mul-destined data into out_aux's BO via a swapped DMA channel).
//
// NUM_INVOCATIONS bound on `%cmax` (NOT `0xFFFFFE` per the
// fuse_channels_npu #25-fix lesson): Pass A's `dma_repeat` inference reads
// the outer `scf.for` upper bound directly into the conduit's `dma_repeat`
// attribute (`ObjectFifoToConduit.cpp:1200-1203`).  An unbounded core loop
// blocks the core on iteration NUM_INVOCATIONS+1 waiting for buffers the
// host never sends.  Bound to NUM_INVOCATIONS=4.

module {
  // Pre-fuse: devAdd produces `inter_add`, devMul consumes `consume_add` and
  // additionally produces `ext_out_aux` in a sequential second section of
  // its core body.  Matching `fusion_group="addmul_hybrid"` on the
  // intermediate pair lets `--conduit-fuse-operators` (injected by
  // `--conduit-fuse-spatial`) pair them and erase the intermediate after
  // `--aie-combine-device same-tile=true` (gated by
  // `--conduit-fuse-core-bodies-flag` per `aiecc.cpp:1509-1511`) collapses
  // the two devices onto tile(0,2).  `--conduit-fuse-core-bodies` then
  // merges the two per-iteration loop bodies into one core; the
  // `@ext_out_mul` + `@ext_out_aux` producer-side conduits become the
  // fuse-channels grouping eligibility surface on tile(0,2).

  aie.device(NPUDEVICE) @devAdd {

    // ---- Producer: Add (in[j] + 1.0 -> inter[j]) ----
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)

    aie.objectfifo @ext_in_add (%shim_a, {%tile_a}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @inter_add (%tile_a, {%shim_a}, 2 : i32)
        {fusion_group = "addmul_hybrid"}
        : !aie.objectfifo<memref<64xbf16>>

    %core_a = aie.core(%tile_a) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      // NUM_INVOCATIONS in test.cpp is 4; bound here matches host dispatch
      // count.  See header comment for the dma_repeat-inflation hazard.
      %cmax = arith.constant 4 : index
      %cone = arith.constant 1.0 : bf16
      scf.for %niter = %c0 to %cmax step %c1 {
        %sv_in = aie.objectfifo.acquire @ext_in_add (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_in = aie.objectfifo.subview.access %sv_in[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_out = aie.objectfifo.acquire @inter_add (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_out = aie.objectfifo.subview.access %sv_out[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_in[%j]  : memref<64xbf16>
          %r = arith.addf %v, %cone      : bf16
          memref.store %r, %elem_out[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @inter_add  (Produce, 1)
        aie.objectfifo.release @ext_in_add (Consume, 1)
      }
      aie.end
    }

    aie.runtime_sequence @add_seq(%ain  : memref<64xbf16>,
                                  %aout : memref<64xbf16>) {
      %ta_in = aiex.dma_configure_task_for @ext_in_add {
        aie.dma_bd(%ain : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta_in)
      %ta_out = aiex.dma_configure_task_for @inter_add {
        aie.dma_bd(%aout : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ta_out)
      aiex.dma_await_task(%ta_out)
      aiex.dma_free_task(%ta_in)
    }
  }

  aie.device(NPUDEVICE) @devMul {

    // ---- Consumer + Aux producer: Mul section (consume * 2.0 -> ext_out_mul)
    //                               followed by const-write 7.0 -> ext_out_aux.
    // Two producer-side conduits on tile(0,2) (same parent block, disjoint
    // windows) are the fuse-channels grouping candidates AFTER both spatial
    // fusion (same-tile=true via the core-bodies-flag routing) and core-body
    // fusion collapse the two cores onto tile(0,2) and merge their bodies.
    %shim_m = aie.tile(0, 0)
    %tile_m = aie.tile(0, 2)

    aie.objectfifo @consume_add (%shim_m, {%tile_m}, 2 : i32)
        {fusion_group = "addmul_hybrid"}
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @ext_out_mul (%tile_m, {%shim_m}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Section B's producer-side conduit — second fuse-channels candidate.
    aie.objectfifo @ext_out_aux (%tile_m, {%shim_m}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    %core_m = aie.core(%tile_m) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 4 : index
      %ctwo = arith.constant 2.0 : bf16
      // Section B writes a constant 7.0 to every element of @ext_out_aux.
      // 7.0 is bf16-exact (≤ 256, integer), test.cpp verifies byte-for-byte.
      %cseven = arith.constant 7.0 : bf16
      scf.for %niter = %c0 to %cmax step %c1 {

        // ---- Section A: consume * 2.0 -> ext_out_mul ----
        %sv_in = aie.objectfifo.acquire @consume_add (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_in = aie.objectfifo.subview.access %sv_in[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_out = aie.objectfifo.acquire @ext_out_mul (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_out = aie.objectfifo.subview.access %sv_out[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_in[%j]  : memref<64xbf16>
          %r = arith.mulf %v, %ctwo      : bf16
          memref.store %r, %elem_out[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @ext_out_mul (Produce, 1)
        aie.objectfifo.release @consume_add (Consume, 1)

        // ---- Section B: const 7.0 -> ext_out_aux ----
        // Sequential after Section A's releases, so @ext_out_mul's
        // (producer-side) window strictly precedes @ext_out_aux's
        // (producer-side) window — fuse-channels eligibility (interval
        // disjointness in the same parent block) is satisfied.
        %sv_aux = aie.objectfifo.acquire @ext_out_aux (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_aux = aie.objectfifo.subview.access %sv_aux[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          memref.store %cseven, %elem_aux[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @ext_out_aux (Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence @mul_seq(%min  : memref<64xbf16>,
                                  %mout : memref<64xbf16>,
                                  %maux : memref<64xbf16>) {
      %tm_in = aiex.dma_configure_task_for @consume_add {
        aie.dma_bd(%min : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tm_in)
      %tm_out = aiex.dma_configure_task_for @ext_out_mul {
        aie.dma_bd(%mout : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tm_out)
      aiex.dma_await_task(%tm_out)
      %tm_aux = aiex.dma_configure_task_for @ext_out_aux {
        aie.dma_bd(%maux : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tm_aux)
      aiex.dma_await_task(%tm_aux)
      aiex.dma_free_task(%tm_in)
    }
  }
}
