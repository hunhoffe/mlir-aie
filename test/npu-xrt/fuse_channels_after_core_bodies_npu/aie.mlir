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
// HW smoke for the COMPOSITION `--conduit-fuse-core-bodies-flag` →
// `--conduit-fuse-channels-flag`.  Sibling-pair to the single-pass smokes:
//   test/npu-xrt/fuse_core_bodies_npu/        (--conduit-fuse-core-bodies-flag
//                                              --conduit-fuse-spatial alone)
//   test/npu-xrt/fuse_channels_npu/           (--conduit-fuse-channels-flag
//                                              alone, hand-authored single-tile
//                                              IR shape)
//
// Why a separate composition fixture: fuse_channels_npu hand-authors the
// single-device single-core Add+Mul shape that fuse-channels needs as INPUT.
// fuse_core_bodies_npu hand-authors the two-device pre-fuse shape that
// fuse-core-bodies needs as INPUT.  Neither answers the production-pipeline
// question: does fuse-core-bodies' OUTPUT actually satisfy fuse-channels'
// INPUT eligibility?  This fixture wires both flags through the SAME
// `aie.mlir` source and verifies HW byte-equivalence with the single-pass
// references.
//
// Source IR (clone of fuse_core_bodies_npu shape — two `aie.device` blocks,
// both targeting `tile(0,2)`, intermediate channel pair tagged
// `fusion_group="addmul_corebody"` so `aie-combine-device same-tile=true`
// can pair them across devices):
//
//   devAdd:  shim(0,0) → @ext_in_add → core_a on tile(0,2) → @inter_add
//                                                            (tile→shim,
//                                                             depth=2,
//                                                             fusion_group=
//                                                              "addmul_corebody")
//   devMul:  shim(0,0) → @consume_add (shim→tile, depth=2,
//                                       fusion_group="addmul_corebody")
//                       → core_m on tile(0,2) → @ext_out_mul (tile→shim,
//                                                             depth=2)
//
// Pipeline behavior (RUN line: `--conduit-fuse-core-bodies-flag
// --conduit-fuse-channels-flag`):
//
//   1. `--use-conduit` lowers ObjectFifo → Conduit IR (Pass A).
//   2. `--conduit-fuse-core-bodies-flag` triggers `aie-combine-device
//      same-tile=true` (gated by this flag, see `aiecc.cpp:1509-1511`),
//      collapsing devAdd + devMul into ONE merged device on tile(0,2).
//   3. `--conduit-fuse-core-bodies` Step 0 unifies the
//      `@inter_add`/`@consume_add` channel pair via fusion_group; Steps
//      2–5 then attempt full body merge (intermediate routing decision +
//      body composition + intermediate erase).
//   4. `--conduit-fuse-channels` runs on the merged-device IR.  If
//      fuse-core-bodies fully merged bodies, only `@ext_out_mul` remains as
//      a producer-side conduit on tile(0,2) and fuse-channels is a no-op
//      (still an interesting smoke: it verifies fuse-channels does not
//      choke on the post-fuse-core-bodies IR shape).  If fuse-core-bodies
//      skipped body merge, `@inter` survives as a tile→tile self-loop on
//      tile(0,2) and fuse-channels groups `@inter` + `@ext_out_mul` (both
//      producer-side, same parent block, disjoint windows) — same shape as
//      fuse_channels_npu's hand-authored source IR, reached from the
//      OPPOSITE direction (composed pipeline vs. hand-authored).
//   5. `--conduit-to-dma` lowers to the AIE dialect ops the rest of aiecc
//      consumes.
//
// EITHER outcome compiles clean and exercises the composition; the value
// is in proving the two passes can run in sequence on the SAME source IR
// without crashing or producing wrong bytes.
//
// Why no `conduit.wait_all` consumers: the related MED bug at
//   test/Dialect/Conduit/path_c_async_fuse_corebody_blocks_at_wait_all.mlir
// fires when `--conduit-fuse-core-bodies` precedes `--conduit-to-dma` on
// IR carrying `wait_all` consumers.  This fixture uses ordinary
// `aie.objectfifo.acquire` / `.release` / `.subview.access` (which Pass A
// lowers to `conduit.acquire` / `conduit.release` / `conduit.subview_access`),
// so the wait_all path is not exercised here.
//
// Why no cross-element-type fusion: cross-element-type fan-in is scoped
// out per the user-locked design (CLAUDE.md §"Locked design decisions").
// All channels here carry `memref<64xbf16>`.
//
// Compute is inlined (no `func.call` to external kernels) so the test is
// self-contained — no peano/chess link step required for the kernel side.
// (The peano REQUIRES line still applies because Chess cannot select
// `fp_to_bf16` for the inline core; see conduit.lit header.)
//
// Reference (test.cpp computes this):
//   in[j] = (j % 16),  out[j] = (in[j] + 1.0) * 2.0.
// Choice keeps every value bf16-exact: max value (15+1)*2 = 32 ≤ 256, and
// integers in [0, 256) are exact in bf16.
//
// PASS expectation: every dispatch (4 invocations) completes; output bytes
// match reference, and (separately, by construction) match the
// single-pass siblings on the same input.  Failure mode classes:
//   * Compile-time crash inside fuse-channels triggered by fuse-core-bodies'
//     output IR shape (the question this fixture is designed to answer).
//   * Hang on the second invocation = Pattern E forward-chain mis-fusion
//     or merged-tile lock interaction (sibling to fuse_core_bodies_npu
//     failure modes).
//   * Wrong bytes mid-buffer = fuse-channels mis-grouping the surviving
//     producer-side conduits on the merged tile.
//
// NUM_INVOCATIONS bound on `%cmax` (NOT `0xFFFFFE` per the
// fuse_channels_npu #25-fix lesson): Pass A's `dma_repeat` inference reads
// the outer `scf.for` upper bound directly into the conduit's `dma_repeat`
// attribute (`ObjectFifoToConduit.cpp:1200-1203`).  An unbounded core
// loop blocks the core on iteration NUM_INVOCATIONS+1 waiting for buffers
// the host never sends.  Bound to NUM_INVOCATIONS=4.

module {

  aie.device(NPUDEVICE) @devAdd {

    // ---- Producer: Add (in[j] + 1.0 -> inter[j]) ----
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)

    aie.objectfifo @ext_in_add (%shim_a, {%tile_a}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @inter_add (%tile_a, {%shim_a}, 2 : i32)
        {fusion_group = "addmul_corebody"}
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

    // ---- Consumer: Mul (consume[j] * 2.0 -> ext_out[j]) ----
    %shim_m = aie.tile(0, 0)
    %tile_m = aie.tile(0, 2)

    aie.objectfifo @consume_add (%shim_m, {%tile_m}, 2 : i32)
        {fusion_group = "addmul_corebody"}
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @ext_out_mul (%tile_m, {%shim_m}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    %core_m = aie.core(%tile_m) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 4 : index
      %ctwo = arith.constant 2.0 : bf16
      scf.for %niter = %c0 to %cmax step %c1 {
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
      }
      aie.end
    }

    aie.runtime_sequence @mul_seq(%min  : memref<64xbf16>,
                                  %mout : memref<64xbf16>) {
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
      aiex.dma_free_task(%tm_in)
    }
  }
}
