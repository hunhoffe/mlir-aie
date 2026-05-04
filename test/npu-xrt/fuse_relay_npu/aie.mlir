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
// HW smoke for `--conduit-fuse-relay` (gather→scatter relay folding).
// Lit-style e2e companion to lit-only pin
//   test/Dialect/Conduit/fuse_relay_basic.mlir
// and SKIP-77 Python-harness sibling
//   iron_operator_mlir/fusion_npu_regression/fuse_relay_smoke.py
// (the Python harness skips because aiecc lacked the
// `conduit-fuse-relay-flag` cl::opt; that flag is added in lockstep
// with this fixture so the smoke can compile end-to-end).
//
// Single-column scenario at column 0: input data flows shim(0,0) →
// memtile(0,1) → compute(0,2) via two `aie.objectfifo` channels chained
// by `aie.objectfifo.link`.  After `--objectfifo-to-conduit` (Pass A) the
// linked pair lowers to a `conduit.gather` + `conduit.scatter` pair on
// memtile(0,1) with an intermediate channel that has no other users —
// exactly the IR shape `--conduit-fuse-relay` matches.  The pass folds
// the gather/scatter pair into a single `conduit.transpose`, eliminates
// the intermediate channel, and halves memtile DMA-channel usage on the
// input path (one S2MM+MM2S pair instead of two).  The output path
// (compute → shim direct, no relay) is unchanged by fuse-relay and
// serves as a cross-check that non-relay channels are untouched.
//
// Compute is inlined (no `func.call` to external kernels) so the test is
// self-contained — no peano/chess link step required for the kernel.
//
// Reference (test.cpp computes this):
//   in[j] = (j % 16),  out[j] = in[j] + 1.0.
// Choice keeps every value bf16-exact: max value 15+1 = 16 ≤ 256, and
// integers in [0, 256) are exact in bf16.
//
// PASS expectation: every dispatch (4 invocations) completes; output
// bytes match reference.  Failure modes that lit-only fixtures cannot
// catch (per fuse_relay_smoke.py docstring):
//   * Relay-elimination produces compile-clean IR but the post-fuse
//     direct flow violates a tile-routing constraint → HW deadlock /
//     timeout symptom that lit-only checks miss.
//   * Eliminated relay's lock state survives somewhere in the merged IR
//     → silent corruption visible only as wrong output bytes after
//     dispatch completes.
//   * Multi-relay chains reduce wrong (covered structurally by single-
//     relay shape here; multi-relay variants are deferred to a follow-on
//     fixture).
// Multi-invocation (NUM_INVOCATIONS=4 in test.cpp) catches second-
// invocation lock-state leftovers that single-dispatch would hide.

module {
  aie.device(NPUDEVICE) {
    %t00 = aie.tile(0, 0)        // shim
    %t01 = aie.tile(0, 1)        // memtile (relay; folded by fuse-relay)
    %t02 = aie.tile(0, 2)        // compute

    // ---- Input path: shim → memtile → compute (relayed; fuse-relay folds) ----
    aie.objectfifo @in_shim    (%t00, {%t01}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @in_compute (%t01, {%t02}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo.link [@in_shim] -> [@in_compute] ([] [])

    // ---- Output path: compute → shim direct (no relay; fuse-relay no-op) ----
    aie.objectfifo @out_shim (%t02, {%t00}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.core(%t02) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 0xFFFFFE : index
      %cone = arith.constant 1.0 : bf16
      scf.for %niter = %c0 to %cmax step %c1 {
        %sv_in = aie.objectfifo.acquire @in_compute (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_in = aie.objectfifo.subview.access %sv_in[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_out = aie.objectfifo.acquire @out_shim (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_out = aie.objectfifo.subview.access %sv_out[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_in[%j]  : memref<64xbf16>
          %r = arith.addf %v, %cone      : bf16
          memref.store %r, %elem_out[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @out_shim   (Produce, 1)
        aie.objectfifo.release @in_compute (Consume, 1)
      }
      aie.end
    }

    aie.runtime_sequence @relay_seq(%in  : memref<64xbf16>,
                                    %out : memref<64xbf16>) {
      %t_in = aiex.dma_configure_task_for @in_shim {
        aie.dma_bd(%in : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t_in)
      %t_out = aiex.dma_configure_task_for @out_shim {
        aie.dma_bd(%out : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t_out)
      aiex.dma_await_task(%t_out)
      aiex.dma_free_task(%t_in)
    }
  }
}
