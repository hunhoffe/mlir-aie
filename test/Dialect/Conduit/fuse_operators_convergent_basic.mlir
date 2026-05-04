//===- fuse_operators_convergent_basic.mlir -----------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Track 3 — convergent merge in `--conduit-fuse-operators`.
//
// Smallest 2:1 SwiGLU-shaped fan-in. Two producer devices (gate + up) feed
// a single consumer device (eltmul). Both producer intermediates carry the
// SAME `fusion_group = "swiglu_fg0"` tag, and the consumer device has TWO
// matching consumer intermediates (one per producer) carrying the same tag
// AND distinguished by a per-producer `fusion_index` (0 = gate, 1 = up).
//
// Per the locked Track 3 design (CLAUDE.md USER-LOCKED 2026-04-26):
//   * Consumer-side IR shape after fuse = K SEPARATE `fused_intermediate_N`
//     channels, NOT one multi-producer channel (preserves Pass C single-
//     producer assumption).
//   * Same element-type only for initial landing.
//   * Iterative pairwise N-way merge (here K=2, so one pairwise step).
//
// Phase 2 (Sprint N+2) flips this fixture from XFAIL → PASS:
// `--conduit-fuse-operators` now matches by (fusion_group, fusion_index)
// and pre-allocates K stable `fused_intermediate_K` names per convergent
// group, so the iterate-pairwise device-merge driver collapses
// (devGate, devUp, devMul) into a single merged device with two distinct
// fused intermediates (one per producer).
//
// Sibling pattern reference: fuse_operators_basic.mlir (1:1 case).
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
// Metafix Candidate 1: also smoke through full Pass C + downstream
// legalization so any per-channel BD-pool exhaustion or dialect-verifier
// failure surfaces here, not on first NPU contact.
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// CHECK-LABEL: module @fuse_operators_convergent_basic

// Only ONE merged device should remain after fusion (gate + up + mul → one):
// CHECK:       aie.device(npu2)
// CHECK-NOT:   aie.device(npu2)

// Two distinct fused intermediates — one per producer — are emitted
// (consumer side stays single-producer per-channel by design):
// CHECK-DAG:   conduit.create @fused_intermediate_0
// CHECK-DAG:   conduit.create @fused_intermediate_1

// External I/O channels survive (gate input, up input, mul output):
// CHECK-DAG:   conduit.create @ext_in_gate
// CHECK-DAG:   conduit.create @ext_in_up
// CHECK-DAG:   conduit.create @ext_out_mul

// Pre-fuse intermediates do NOT survive:
// CHECK-NOT:   conduit.create @inter_gate
// CHECK-NOT:   conduit.create @inter_up
// CHECK-NOT:   conduit.create @consume_gate
// CHECK-NOT:   conduit.create @consume_up

module @fuse_operators_convergent_basic {
  // Producer 1 — "gate" GEMV-like.
  aie.device(npu2) @devGate {
    %shim   = aie.tile(0, 0)
    %tile_g = aie.tile(0, 2)

    aie.objectfifo @ext_in_gate(%shim, {%tile_g}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    aie.objectfifo @inter_gate(%tile_g, {%shim}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @gate_kernel(memref<128xbf16>, memref<128xbf16>)

    %core_g = aie.core(%tile_g) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_gate(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_gate(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @gate_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @inter_gate(Produce, 1)
        aie.objectfifo.release @ext_in_gate(Consume, 1)
      }
      aie.end
    } {link_with = "gate.a"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_gate {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_gate {
        aie.dma_bd(%arg1 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  // Producer 2 — "up" GEMV-like.
  aie.device(npu2) @devUp {
    %shim   = aie.tile(0, 0)
    // Compute tile at row 3 (distinct from devGate's row-2 compute) so
    // devUp's tile set is not a subset of devGate's; this preserves the
    // offset path the original CHECK-DAG channel lookups assume.
    %tile_u = aie.tile(0, 3)

    aie.objectfifo @ext_in_up(%shim, {%tile_u}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    aie.objectfifo @inter_up(%tile_u, {%shim}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @up_kernel(memref<128xbf16>, memref<128xbf16>)

    %core_u = aie.core(%tile_u) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_up(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_up(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @up_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @inter_up(Produce, 1)
        aie.objectfifo.release @ext_in_up(Consume, 1)
      }
      aie.end
    } {link_with = "up.a"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_up {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_up {
        aie.dma_bd(%arg1 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  // Consumer — eltmul. TWO matching intermediates (one per producer),
  // distinguished by `fusion_index`. Per locked design, post-fuse this
  // becomes TWO @fused_intermediate_N channels (NOT one multi-producer).
  aie.device(npu2) @devMul {
    %shim   = aie.tile(0, 0)
    // Compute tile at row 4 (distinct from devGate row-2 + devUp row-3
    // compute) so devMul's tile set is not a subset of the merged
    // (devGate+devUp) device's tiles after the first pairwise merge step.
    %tile_c = aie.tile(0, 4)

    aie.objectfifo @consume_gate(%shim, {%tile_c}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    aie.objectfifo @consume_up(%shim, {%tile_c}, 2 : i32)
        {fusion_group = "swiglu_fg0", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    aie.objectfifo @ext_out_mul(%tile_c, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @mul_kernel(memref<128xbf16>, memref<128xbf16>, memref<128xbf16>)

    %core_c = aie.core(%tile_c) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in_g = aie.objectfifo.acquire @consume_gate(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_g_buf = aie.objectfifo.subview.access %in_g[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %in_u = aie.objectfifo.acquire @consume_up(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_u_buf = aie.objectfifo.subview.access %in_u[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @ext_out_mul(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @mul_kernel(%in_g_buf, %in_u_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_out_mul(Produce, 1)
        aie.objectfifo.release @consume_up(Consume, 1)
        aie.objectfifo.release @consume_gate(Consume, 1)
      }
      aie.end
    } {link_with = "mul.a"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>, %arg2: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @consume_gate {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @consume_up {
        aie.dma_bd(%arg1 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @ext_out_mul {
        aie.dma_bd(%arg2 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }
}
