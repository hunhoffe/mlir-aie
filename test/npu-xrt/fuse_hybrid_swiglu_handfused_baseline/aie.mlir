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
// HAND-FUSED single-device baseline for the SwiGLU-shaped 4-device fixture
// `test/npu-xrt/fuse_hybrid_swiglu_npu/aie.mlir`.  This fixture performs the
// SAME end-to-end compute as that fixture, but written directly as one
// already-fused device with one core that fuses gate * up + 1.0 / + 2.0 in
// a single inline body.  No Conduit fusion passes run; the lit RUN line
// uses the stateful aiecc pipeline (no --use-conduit, no --conduit-fuse-*-flag).
//
// PURPOSE: gives a known-working post-AIEObjectFifoStatefulTransform IR
// shape for the same compute as the auto-fused conduit version.  The
// auto-fused conduit fixture currently HW-FAILS with all-zero outputs;
// IR-diffing this baseline against the post-Pass-C conduit IR isolates
// the bug surface introduced by the fusion passes.
//
// Compute (matches fuse_hybrid_swiglu_npu test.cpp):
//   gate_in[j] = up_in[j] = (j % 8)                   - bf16 ramp
//   mul[j]     = gate_in[j] * up_in[j]                - bf16 multiply
//   ext_out_a[j] = mul[j] + 1.0                       - bf16 add
//   ext_out_b[j] = mul[j] + 2.0                       - bf16 add
// All values bf16-exact integers.  IO_LEN = 64.  4 invocations.
//
// Structural choices:
//   * Single aie.device (NPUDEVICE -> npu2 / npu2_4col via sed).
//   * Single aie.core on tile(0,2) - matches the converged target tile of
//     the auto-fused fixture's mul+sink merged core.
//   * 4 ObjectFifos - all depth=2 (no fuse-channels grouping needed since
//     no fusion runs):
//       ext_in_gate (shim->tile)
//       ext_in_up   (shim->tile)
//       ext_out_a   (tile->shim)
//       ext_out_b   (tile->shim)
//   * Outer scf.for bound = 4 (= NUM_INVOCATIONS in test.cpp).  Per
//     fuse_channels_npu fixture comment: stateful core acquires must
//     match host dispatch count.  cmax = 0xFFFFFE works only for the
//     conduit pipeline (Pass A reads loop bound into dma_repeat); under
//     stateful lowering the core would block on the 5th iteration.
//   * Compute inlined as bf16 mul + add (no func.call to external kernels).
//     Chess (xchesscc) cannot select fp_to_bf16; conduit.lit uses
//     --no-xchesscc --no-xbridge to route kernel compile through Peano
//     (sibling pattern: fuse_channels_npu, fuse_core_bodies_npu).
//   * `aie.runtime_sequence` uses `aiex.dma_configure_task_for` style
//     (matching the auto-fused fixture and fuse_channels_npu) with 4
//     args in source order: gate_in, up_in, out_a, out_b - matches the
//     test.cpp arg layout (arg3=gate, arg4=up, arg5=out_a, arg6=out_b).

module {
  aie.device(NPUDEVICE) {

    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    aie.objectfifo @ext_in_gate (%shim, {%tile}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @ext_in_up   (%shim, {%tile}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @ext_out_a   (%tile, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @ext_out_b   (%tile, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    %core = aie.core(%tile) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      // Bound to NUM_INVOCATIONS (test.cpp line `constexpr int NUM_INVOCATIONS = 4`).
      %cmax = arith.constant 4 : index
      %cone = arith.constant 1.0 : bf16
      %ctwo = arith.constant 2.0 : bf16
      scf.for %niter = %c0 to %cmax step %c1 {

        // Acquire all 4 buffers for this invocation.
        %sv_g = aie.objectfifo.acquire @ext_in_gate (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_g = aie.objectfifo.subview.access %sv_g[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_u = aie.objectfifo.acquire @ext_in_up (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_u = aie.objectfifo.subview.access %sv_u[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_a = aie.objectfifo.acquire @ext_out_a (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_a = aie.objectfifo.subview.access %sv_a[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_b = aie.objectfifo.acquire @ext_out_b (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_b = aie.objectfifo.subview.access %sv_b[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        // mul[j] = gate[j] * up[j];  out_a[j] = mul + 1.0;  out_b[j] = mul + 2.0.
        scf.for %j = %c0 to %c64 step %c1 {
          %g  = memref.load %elem_g[%j] : memref<64xbf16>
          %u  = memref.load %elem_u[%j] : memref<64xbf16>
          %m  = arith.mulf %g, %u : bf16
          %ra = arith.addf %m, %cone : bf16
          %rb = arith.addf %m, %ctwo : bf16
          memref.store %ra, %elem_a[%j] : memref<64xbf16>
          memref.store %rb, %elem_b[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @ext_out_b   (Produce, 1)
        aie.objectfifo.release @ext_out_a   (Produce, 1)
        aie.objectfifo.release @ext_in_up   (Consume, 1)
        aie.objectfifo.release @ext_in_gate (Consume, 1)
      }
      aie.end
    }

    // Runtime sequence: 4 host args in source-order matching test.cpp
    // arg layout (arg3=gate, arg4=up, arg5=out_a, arg6=out_b).  Pattern
    // mirrors fuse_channels_npu's runtime_sequence: dma_configure_task_for
    // each channel, start tasks, await output tasks, free input tasks.
    aie.runtime_sequence @swiglu_seq(%g_in  : memref<64xbf16>,
                                     %u_in  : memref<64xbf16>,
                                     %a_out : memref<64xbf16>,
                                     %b_out : memref<64xbf16>) {
      %tg = aiex.dma_configure_task_for @ext_in_gate {
        aie.dma_bd(%g_in : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tg)

      %tu = aiex.dma_configure_task_for @ext_in_up {
        aie.dma_bd(%u_in : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tu)

      %ta = aiex.dma_configure_task_for @ext_out_a {
        aie.dma_bd(%a_out : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%ta)

      %tb = aiex.dma_configure_task_for @ext_out_b {
        aie.dma_bd(%b_out : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tb)

      aiex.dma_await_task(%tb)
      aiex.dma_await_task(%ta)
      aiex.dma_free_task(%tu)
      aiex.dma_free_task(%tg)
    }
  }
}
