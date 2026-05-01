//===- fuse_operators_convergent_lcm_rate_align.mlir --------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Track 3 — convergent merge LCM rate-align regression.
//
// Smallest case: BOTH producers fire N=4 times per consumer fire (consumer
// also fires N=4 times in its own loop). LCM(4,4,4) = 4, so the rate-align
// step is a no-op on partition counts but must still preserve the existing
// per-channel `producer_rates` / `consumer_rates` consistency post-merge.
//
// Per the locked Track 3 design (CLAUDE.md USER-LOCKED 2026-04-26): LCM
// rate-align with memory check; for this fixture the memory budget fits
// trivially (depth=2 × 128 bf16 = 512 bytes per intermediate, well under
// any reasonable L1 budget).
//
// Phase 2 of Track 3 (Sprint N+2) lands the convergent matcher + N-way
// pairwise driver but does NOT implement LCM rate-align with the locked
// memory check.  Phase 3+ owns rate-align; this fixture stays XFAIL until
// then so the LCM-aligned `producer_rates` / `consumer_rates` invariant
// is pinned ahead of the implementation.  The conduit-dev landing the
// rate-align step removes the XFAIL line.
//
// The aim of this fixture: confirm that after merge, both fused
// intermediates carry the SAME LCM-aligned partition count (here 4) on
// the producer-side AND consumer-side rate attributes — no off-by-one,
// no asymmetry.
//
// Sibling pattern reference: fuse_operators_convergent_basic.mlir.
//===----------------------------------------------------------------------===//

// XFAIL: *

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-infer-rates --conduit-fuse-operators %s | FileCheck %s
// Metafix Candidate 1.
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-infer-rates --conduit-fuse-operators --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// CHECK-LABEL: module @fuse_operators_convergent_lcm_rate_align

// Only ONE merged device after fusion:
// CHECK:       aie.device(npu2)
// CHECK-NOT:   aie.device(npu2)

// Two fused intermediates with rate=4 on both sides (LCM-aligned):
// CHECK-DAG:   conduit.create @fused_intermediate_0
// CHECK-SAME:    producer_rates = {{.*}}4
// CHECK-SAME:    consumer_rates = {{.*}}4
// CHECK-DAG:   conduit.create @fused_intermediate_1
// CHECK-SAME:    producer_rates = {{.*}}4
// CHECK-SAME:    consumer_rates = {{.*}}4

module @fuse_operators_convergent_lcm_rate_align {
  // Producer 0 — fires 4× per outer step.
  aie.device(npu2) @devGate {
    %shim = aie.tile(0, 0)
    %t    = aie.tile(0, 2)

    aie.objectfifo @ext_in_gate(%shim, {%t}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @inter_gate(%t, {%shim}, 2 : i32)
        {fusion_group = "rate_fg", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @gate_kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %cmax step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
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
      }
      aie.end
    } {link_with = "gate.a"}

    aie.runtime_sequence(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_gate {
        aie.dma_bd(%arg0 : memref<512xbf16>, 0, 512,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 4, stride = 128>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_gate {
        aie.dma_bd(%arg1 : memref<512xbf16>, 0, 512,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 4, stride = 128>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  // Producer 1 — also fires 4× per outer step (same trip count).
  aie.device(npu2) @devUp {
    %shim = aie.tile(0, 0)
    %t    = aie.tile(0, 2)

    aie.objectfifo @ext_in_up(%shim, {%t}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @inter_up(%t, {%shim}, 2 : i32)
        {fusion_group = "rate_fg", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @up_kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %cmax step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
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
      }
      aie.end
    } {link_with = "up.a"}

    aie.runtime_sequence(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_up {
        aie.dma_bd(%arg0 : memref<512xbf16>, 0, 512,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 4, stride = 128>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_up {
        aie.dma_bd(%arg1 : memref<512xbf16>, 0, 512,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 4, stride = 128>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  // Consumer — also fires 4× per outer step (same trip count).
  aie.device(npu2) @devMul {
    %shim = aie.tile(0, 0)
    %tc   = aie.tile(0, 2)

    aie.objectfifo @consume_gate(%shim, {%tc}, 2 : i32)
        {fusion_group = "rate_fg", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @consume_up(%shim, {%tc}, 2 : i32)
        {fusion_group = "rate_fg", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @ext_out(%tc, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @mul_kernel(memref<128xbf16>, memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%tc) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %cmax step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %g = aie.objectfifo.acquire @consume_gate(Consume, 1)
              : !aie.objectfifosubview<memref<128xbf16>>
          %g_buf = aie.objectfifo.subview.access %g[0]
              : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
          %u = aie.objectfifo.acquire @consume_up(Consume, 1)
              : !aie.objectfifosubview<memref<128xbf16>>
          %u_buf = aie.objectfifo.subview.access %u[0]
              : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
          %out = aie.objectfifo.acquire @ext_out(Produce, 1)
              : !aie.objectfifosubview<memref<128xbf16>>
          %out_buf = aie.objectfifo.subview.access %out[0]
              : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
          func.call @mul_kernel(%g_buf, %u_buf, %out_buf)
              : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
          aie.objectfifo.release @ext_out(Produce, 1)
          aie.objectfifo.release @consume_up(Consume, 1)
          aie.objectfifo.release @consume_gate(Consume, 1)
        }
      }
      aie.end
    } {link_with = "mul.a"}

    aie.runtime_sequence(%arg0: memref<512xbf16>, %arg1: memref<512xbf16>, %arg2: memref<512xbf16>) {
      %t0 = aiex.dma_configure_task_for @consume_gate {
        aie.dma_bd(%arg0 : memref<512xbf16>, 0, 512,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 4, stride = 128>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @consume_up {
        aie.dma_bd(%arg1 : memref<512xbf16>, 0, 512,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 4, stride = 128>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @ext_out {
        aie.dma_bd(%arg2 : memref<512xbf16>, 0, 512,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 4, stride = 128>, <size = 128, stride = 1>])
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
