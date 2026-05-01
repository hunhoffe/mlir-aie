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
// HW smoke for the HYBRID Conduit fusion pipeline:
//   --conduit-fuse-spatial + --conduit-fuse-core-bodies-flag +
//   --conduit-fuse-channels-flag together.
//
// Exercises ALL THREE Conduit fusion passes simultaneously — the actual
// Llama-decode hybrid path.  PASS confirms fused-decode capability at small
// blast radius before Llama-scale verify.  Sibling per-pass smokes
// (fuse_operators_convergent_npu / fuse_core_bodies_npu / fuse_channels_npu)
// already verify each pass in isolation; this fixture verifies their
// pairwise + three-way interaction surface that single-pass smokes cannot.
//
// Pipeline composition (in the order the passes run inside aiecc.cpp under
// the three flags):
//   1. --conduit-fuse-spatial → --conduit-fuse-operators:
//      a. K=2 convergent merge of devGate + devUp → devMul (paired via
//         fusion_group="swiglu_fg0", fusion_index=0/1) — produces two
//         @fused_intermediate_N channels per Track 3 design (NOT one
//         multi-producer channel; preserves Pass C single-producer
//         per-channel invariant).
//      b. 1:1 spatial merge of devMul → devSink (paired via shared
//         fusion_group="mulsink_cb").
//   2. --conduit-fuse-core-bodies-flag → routes --aie-combine-device with
//      same-tile=true (per fuse_core_bodies_npu/conduit.lit).  Mul + sink
//      cores land on the same compute tile and core-body fusion merges
//      their per-iteration loop bodies into one core; the intermediate
//      conduit pair @mul_inter / @consume_mul is erased.
//   3. --conduit-fuse-channels-flag → fuse-channels (annotation-only)
//      groups producer-side conduits on the same tile in the same parent
//      block with disjoint live windows.  After (1)+(2) the merged mul+
//      sink core produces TWO external outputs (@ext_out_a + @ext_out_b)
//      on the SAME tile in the SAME parent block — these are the
//      channel-fusion candidates.  Pass C downstream consumes the
//      annotation to fold the grouped conduits onto a single HW DMA
//      channel.
//
// Pre-fuse 4 devices, all on tile(0,2) col 0:
//   devGate: identity copy ext_in_gate → inter_gate (fusion_group="swiglu_fg0",
//            fusion_index=0).
//   devUp:   identity copy ext_in_up   → inter_up   (fusion_group="swiglu_fg0",
//            fusion_index=1).
//   devMul:  consume_gate (fusion_group="swiglu_fg0", fusion_index=0) ×
//            consume_up   (fusion_group="swiglu_fg0", fusion_index=1) →
//            elementwise multiply → mul_inter (fusion_group="mulsink_cb").
//   devSink: consume_mul (fusion_group="mulsink_cb") → produces TWO
//            external outputs ext_out_a (= mul + 1.0) and ext_out_b
//            (= mul + 2.0).  Distinct constants on the two outputs let
//            channel-fusion mis-routing surface as a byte mismatch
//            (vs. silent passthrough if both outputs computed the same
//            value).
//
// Post-spatial-fuse expected placement (column-spread under npu2_4col
// budget; --conduit-fuse-spatial spreads K=2 producers per
// fuse_operators_convergent_npu/conduit.lit, while --conduit-fuse-core-
// bodies-flag's same-tile=true keeps the mul+sink pair co-located):
//   col 0 tile(0,2): gate producer
//   col 1 tile(0,2): up producer
//   col 2 tile(0,2): merged mul+sink (post core-body fuse)
//
// Compute is inlined (no `func.call` to external kernels) so the test is
// self-contained — no peano/chess link step required.
//
// Reference (test.cpp computes this):
//   gate_in[j] = up_in[j] = (j % 8)         -- bf16-exact, symmetric
//   mul[j]     = gate_in[j] * up_in[j]      -- max 7*7 = 49
//   ext_out_a[j] = mul[j] + 1.0             -- max 50
//   ext_out_b[j] = mul[j] + 2.0             -- max 51
// All values are bf16-exact (≤ 256, integer).  Symmetric inputs make
// verification independent of post-fusion arg-reprojection order on the
// PRODUCER side (gate vs up arg slots interchangeable).  Distinct +1.0/
// +2.0 constants on the SINK side make channel-fusion mis-routing between
// ext_out_a and ext_out_b surface as a byte mismatch.
//
// PASS expectation: every dispatch (4 invocations) completes; output
// bytes match reference.  Failure-mode interpretation:
//   * Convergent-merge bug → wrong-bytes on BOTH out_a and out_b
//     (gate/up producer-side intermediates mis-paired during
//     fused_intermediate_N erasure; both downstream sinks see corrupted
//     mul_inter).
//   * Core-body merge interaction bug → hang on the SECOND invocation
//     (Pattern E forward-chain mis-fusion class; merged mul+sink core
//     wedged across reuse), OR truncated output from iter-count
//     mis-alignment between mul and sink loops.
//   * Channel-grouping race → out_a and out_b SWAPPED in the output
//     (mis-routed across the grouped HW DMA channel — out_a holds
//     mul+2.0 values, out_b holds mul+1.0), OR all-zero on one of the
//     two outputs (BD-ring ordering violation post-grouping).
// PASS = ALL THREE pass interactions correct simultaneously.

module {
  // Each aie.device is a SymbolOp; explicit @names avoid the @main collision
  // at parse time across the 4 pre-fuse devices.  Post-fusion --aie-combine-
  // device collapses them into one device (with same-tile=true under
  // --conduit-fuse-core-bodies-flag).

  aie.device(NPUDEVICE) @devGate {

    // ---- Producer 1: gate (identity copy ext_in_gate -> inter_gate) ----
    %shim_g = aie.tile(0, 0)
    %tile_g = aie.tile(0, 2)

    aie.objectfifo @ext_in_gate (%shim_g, {%tile_g}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @inter_gate (%tile_g, {%shim_g}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<64xbf16>>

    %core_g = aie.core(%tile_g) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 0xFFFFFE : index
      scf.for %niter = %c0 to %cmax step %c1 {
        %sv_in = aie.objectfifo.acquire @ext_in_gate (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_in = aie.objectfifo.subview.access %sv_in[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_out = aie.objectfifo.acquire @inter_gate (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_out = aie.objectfifo.subview.access %sv_out[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_in[%j]  : memref<64xbf16>
          memref.store %v, %elem_out[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @inter_gate   (Produce, 1)
        aie.objectfifo.release @ext_in_gate  (Consume, 1)
      }
      aie.end
    }

    aie.runtime_sequence @gate_seq(%gin  : memref<64xbf16>,
                                   %gout : memref<64xbf16>) {
      %tg_in = aiex.dma_configure_task_for @ext_in_gate {
        aie.dma_bd(%gin : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tg_in)
      %tg_out = aiex.dma_configure_task_for @inter_gate {
        aie.dma_bd(%gout : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tg_out)
      aiex.dma_await_task(%tg_out)
      aiex.dma_free_task(%tg_in)
    }
  }

  aie.device(NPUDEVICE) @devUp {

    // ---- Producer 2: up (identity copy ext_in_up -> inter_up) ----
    %shim_u = aie.tile(0, 0)
    %tile_u = aie.tile(0, 2)

    aie.objectfifo @ext_in_up (%shim_u, {%tile_u}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @inter_up (%tile_u, {%shim_u}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<64xbf16>>

    %core_u = aie.core(%tile_u) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 0xFFFFFE : index
      scf.for %niter = %c0 to %cmax step %c1 {
        %sv_in = aie.objectfifo.acquire @ext_in_up (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_in = aie.objectfifo.subview.access %sv_in[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_out = aie.objectfifo.acquire @inter_up (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_out = aie.objectfifo.subview.access %sv_out[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_in[%j]  : memref<64xbf16>
          memref.store %v, %elem_out[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @inter_up   (Produce, 1)
        aie.objectfifo.release @ext_in_up  (Consume, 1)
      }
      aie.end
    }

    aie.runtime_sequence @up_seq(%uin  : memref<64xbf16>,
                                 %uout : memref<64xbf16>) {
      %tu_in = aiex.dma_configure_task_for @ext_in_up {
        aie.dma_bd(%uin : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tu_in)
      %tu_out = aiex.dma_configure_task_for @inter_up {
        aie.dma_bd(%uout : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tu_out)
      aiex.dma_await_task(%tu_out)
      aiex.dma_free_task(%tu_in)
    }
  }

  aie.device(NPUDEVICE) @devMul {

    // ---- Consumer 1: eltmul (consume_gate × consume_up -> mul_inter) ----
    // mul_inter is fusion_group="mulsink_cb" — paired with devSink's
    // consume_mul; spatial fuse merges devMul + devSink, then core-body
    // fuse erases the mul_inter / consume_mul intermediate.
    %shim_c = aie.tile(0, 0)
    %tile_c = aie.tile(0, 2)

    aie.objectfifo @consume_gate (%shim_c, {%tile_c}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @consume_up (%shim_c, {%tile_c}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @mul_inter (%tile_c, {%shim_c}, 2 : i32)
        {fusion_group = "mulsink_cb"}
        : !aie.objectfifo<memref<64xbf16>>

    %core_c = aie.core(%tile_c) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 0xFFFFFE : index
      scf.for %niter = %c0 to %cmax step %c1 {
        %sv_g = aie.objectfifo.acquire @consume_gate (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_g = aie.objectfifo.subview.access %sv_g[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_u = aie.objectfifo.acquire @consume_up (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_u = aie.objectfifo.subview.access %sv_u[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_o = aie.objectfifo.acquire @mul_inter (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_o = aie.objectfifo.subview.access %sv_o[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %a = memref.load %elem_g[%j] : memref<64xbf16>
          %b = memref.load %elem_u[%j] : memref<64xbf16>
          %p = arith.mulf %a, %b : bf16
          memref.store %p, %elem_o[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @mul_inter    (Produce, 1)
        aie.objectfifo.release @consume_up   (Consume, 1)
        aie.objectfifo.release @consume_gate (Consume, 1)
      }
      aie.end
    }

    aie.runtime_sequence @mul_seq(%cg  : memref<64xbf16>,
                                  %cu  : memref<64xbf16>,
                                  %mout : memref<64xbf16>) {
      %tc_g = aiex.dma_configure_task_for @consume_gate {
        aie.dma_bd(%cg : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tc_g)
      %tc_u = aiex.dma_configure_task_for @consume_up {
        aie.dma_bd(%cu : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tc_u)
      %tc_o = aiex.dma_configure_task_for @mul_inter {
        aie.dma_bd(%mout : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tc_o)
      aiex.dma_await_task(%tc_o)
      aiex.dma_free_task(%tc_u)
      aiex.dma_free_task(%tc_g)
    }
  }

  aie.device(NPUDEVICE) @devSink {

    // ---- Sink: consume mul, produce TWO external outputs ----
    // ext_out_a + ext_out_b are channel-fusion candidates after spatial+
    // core-body fusion lands them on the same tile in the same parent
    // block.  Both depth=1 per fuse-channels Tier-3-depth=1 design intent
    // (commit 14eb385272: depth>1 + Tier 3 puts/gets is declined by the
    // pass; matches fuse_channels_npu fixture).  consume_mul is depth=2
    // — it gets erased post core-body fuse so its depth doesn't matter
    // for fuse-channels eligibility.
    %shim_s = aie.tile(0, 0)
    %tile_s = aie.tile(0, 2)

    aie.objectfifo @consume_mul (%shim_s, {%tile_s}, 2 : i32)
        {fusion_group = "mulsink_cb"}
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @ext_out_a (%tile_s, {%shim_s}, 1 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @ext_out_b (%tile_s, {%shim_s}, 1 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    %core_s = aie.core(%tile_s) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 0xFFFFFE : index
      %cone = arith.constant 1.0 : bf16
      %ctwo = arith.constant 2.0 : bf16
      scf.for %niter = %c0 to %cmax step %c1 {
        %sv_in = aie.objectfifo.acquire @consume_mul (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_in = aie.objectfifo.subview.access %sv_in[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        // ---- A section: write mul + 1.0 into ext_out_a ----
        // Disjoint-window producer-side conduit #1 on tile(0,2).
        %sv_a = aie.objectfifo.acquire @ext_out_a (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_a = aie.objectfifo.subview.access %sv_a[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_in[%j]  : memref<64xbf16>
          %r = arith.addf %v, %cone      : bf16
          memref.store %r, %elem_a[%j]   : memref<64xbf16>
        }

        aie.objectfifo.release @ext_out_a (Produce, 1)

        // ---- B section: write mul + 2.0 into ext_out_b ----
        // Disjoint-window producer-side conduit #2 on tile(0,2).  Sequential
        // after the A section's release → fuse-channels eligibility met
        // (same producer tile, same parent block, disjoint live windows).
        %sv_b = aie.objectfifo.acquire @ext_out_b (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_b = aie.objectfifo.subview.access %sv_b[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_in[%j]  : memref<64xbf16>
          %r = arith.addf %v, %ctwo      : bf16
          memref.store %r, %elem_b[%j]   : memref<64xbf16>
        }

        aie.objectfifo.release @ext_out_b   (Produce, 1)
        aie.objectfifo.release @consume_mul (Consume, 1)
      }
      aie.end
    }

    aie.runtime_sequence @sink_seq(%sin  : memref<64xbf16>,
                                   %sa   : memref<64xbf16>,
                                   %sb   : memref<64xbf16>) {
      %ts_in = aiex.dma_configure_task_for @consume_mul {
        aie.dma_bd(%sin : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ts_in)
      %ts_a = aiex.dma_configure_task_for @ext_out_a {
        aie.dma_bd(%sa : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ts_a)
      %ts_b = aiex.dma_configure_task_for @ext_out_b {
        aie.dma_bd(%sb : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ts_b)
      aiex.dma_await_task(%ts_b)
      aiex.dma_await_task(%ts_a)
      aiex.dma_free_task(%ts_in)
    }
  }
}
