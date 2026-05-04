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
// HW smoke for `--conduit-fuse-operators` 1:1 spatial fusion (basic).
// Lit-style e2e companion to lit-only pin
//   test/Dialect/Conduit/fuse_operators_basic.mlir
// and Python-harness sibling
//   iron_operator_mlir/fusion_npu_regression/fuse_operators_smoke.py.
//
// Chained Add → Mul on TWO columns: two producer/consumer devices (devAdd +
// devMul) feed each other through one paired intermediate channel
// (`inter_add` ↔ `consume_add`) tagged with matching
// `fusion_group="addmul_smoke_op"`.  Pre-fuse devAdd lives at tile(0,2) and
// devMul lives at tile(0,2); after `--conduit-fuse-spatial` (which routes
// through `--aie-combine-device same-tile=false`, since
// `--conduit-fuse-core-bodies-flag` is NOT set in this fixture) the two
// devices collapse into ONE merged device with devMul's tiles offset to
// column 1 — so the merged producer core sits at tile(0,2) and the merged
// consumer core at tile(1,2).  `--conduit-fuse-operators` then erases the
// intermediate conduit pair and replaces it with `@fused_intermediate_0`,
// rerouting cross-column on-chip.  Two-column smoke uses `npu2_2col`.
//
// Pre-fuse vs post-fuse delta vs the sibling fixture
//   test/npu-xrt/fuse_core_bodies_npu/aie.mlir:
// IR is structurally identical (same 2 devices, same compute, same fusion
// _group, same coefficients, same buffer shape, same compute tile coords);
// the only run-time delta is the pipeline flags (this fixture: spatial-
// only; that fixture: spatial + core-bodies hybrid) and the device target
// (this fixture: 2col, since spatial-only spreads ops across columns).
//
// Compute is inlined (no `func.call` to external kernels) so the test is
// self-contained — no peano/chess link step required.
//
// Reference (test.cpp computes this):
//   in[j] = (j % 16),  out[j] = (in[j] + 1.0) * 2.0.
// Choice keeps every value bf16-exact: max value (15+1)*2 = 32 ≤ 256, and
// integers in [0, 256) are exact in bf16.
//
// PASS expectation: every dispatch (4 invocations) completes; output bytes
// match reference.  Failure modes specific to spatial-only fusion (per
// fuse_operators_smoke.py docstring) are detailed in conduit.lit header.

module {
  // Pre-fuse: devAdd produces `inter_add`, devMul consumes `consume_add`.
  // Matching `fusion_group="addmul_smoke_op"` on the intermediate pair lets
  // `--conduit-fuse-operators` (injected by `--conduit-fuse-spatial`) pair
  // them and erase the intermediate after `--aie-combine-device` collapses
  // the two devices and offsets devMul to column 1.

  aie.device(NPUDEVICE) @devAdd {

    // ---- Producer: Add (in[j] + 1.0 -> inter[j]) ----
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)

    aie.objectfifo @ext_in_add (%shim_a, {%tile_a}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @inter_add (%tile_a, {%shim_a}, 2 : i32)
        {fusion_group = "addmul_smoke_op"}
        : !aie.objectfifo<memref<64xbf16>>

    %core_a = aie.core(%tile_a) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 0xFFFFFE : index
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
        {fusion_group = "addmul_smoke_op"}
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @ext_out_mul (%tile_m, {%shim_m}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    %core_m = aie.core(%tile_m) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 0xFFFFFE : index
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
