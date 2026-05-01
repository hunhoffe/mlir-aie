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
// HW smoke for `--conduit-fuse-operators` convergent (K=2) merge.  E2E
// companion to the lit-only pin
//   test/Dialect/Conduit/fuse_operators_convergent_basic.mlir.
//
// SwiGLU-shaped 2:1 fan-in: two producer devices (devGate + devUp) feed a
// single consumer device (devMul) via paired `fusion_group="swiglu_fg0"`
// channels distinguished by `fusion_index = 0` (gate) and `fusion_index = 1`
// (up).  Pre-fuse all three devices live on tile(0,2) of column 0; after
// `--conduit-fuse-operators` they are placed at columns 0/1/2 of one merged
// device (npu2_4col gives the column budget headroom).  Per locked Track 3
// design the consumer-side IR shape post-fuse is K=2 separate
// `@fused_intermediate_N` channels (NOT one multi-producer channel) — Pass C
// then sees a single-producer per-channel for both fused intermediates.
//
// Compute is inlined (no `func.call` to external kernels) so the test is
// self-contained — no peano/chess link step required.  Producer cores copy
// input verbatim to their intermediate ("identity") so the surviving
// intermediates carry the input bytes; consumer core multiplies the two
// intermediates element-wise and writes to the external output.
//
// Reference (test.cpp computes this):
//   gate_in[j] = up_in[j] = (j % 16),  out[j] = gate_in[j] * up_in[j].
// Choice keeps every value bf16-exact (integers ≤ 256 are bf16-exact;
// 15² = 225 ≤ 256) AND symmetric across the two producer args, so output
// verification does not depend on which surviving runtime-sequence arg
// position holds gate vs up after fusion's arg reprojection.
//
// PASS expectation: dispatch completes; output matches reference bytes.
// Hang/timeout = post-fuse runtime sequence drives wrong DMA sequence.
// Wrong-bytes = arg reprojection (Step 8c-bis) or convergent-aware
// intermediate-erase mis-paired producer/consumer halves.

module {
  // Each `aie.device` is a SymbolOp; without an explicit `@name` it defaults
  // to `@main`, which collides at parse time across the 3 pre-fuse devices
  // (`error: redefinition of symbol named 'main'`).  Names are arbitrary --
  // post-fusion `aie-combine-device` collapses these into one device.
  aie.device(NPUDEVICE) @devGate {

    // ---- Producer 1: "gate" (identity copy ext_in_gate -> inter_gate) ----
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

    // ---- Producer 2: "up" (identity copy ext_in_up -> inter_up) ----
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

    // ---- Consumer: eltmul (consume_gate × consume_up -> ext_out_mul) ----
    %shim_c = aie.tile(0, 0)
    %tile_c = aie.tile(0, 2)

    aie.objectfifo @consume_gate (%shim_c, {%tile_c}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @consume_up (%shim_c, {%tile_c}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @ext_out_mul (%tile_c, {%shim_c}, 2 : i32)
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

        %sv_o = aie.objectfifo.acquire @ext_out_mul (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_o = aie.objectfifo.subview.access %sv_o[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %a = memref.load %elem_g[%j] : memref<64xbf16>
          %b = memref.load %elem_u[%j] : memref<64xbf16>
          %p = arith.mulf %a, %b : bf16
          memref.store %p, %elem_o[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @ext_out_mul  (Produce, 1)
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
      %tc_o = aiex.dma_configure_task_for @ext_out_mul {
        aie.dma_bd(%mout : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tc_o)
      aiex.dma_await_task(%tc_o)
      aiex.dma_free_task(%tc_u)
      aiex.dma_free_task(%tc_g)
    }
  }
}
