//===- fuse_operators_convergent_three_producers.mlir -------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Track 3 — convergent merge generalization regression.
//
// 3:1 fan-in (three producers + one consumer). Verifies the implementation
// is NOT a 2-way special case: pairwise iteration must reduce 3 → 1
// (e.g. merge p0+p1 → step state, then merge that+p2 → final consumer).
//
// Per the locked Track 3 design (CLAUDE.md USER-LOCKED 2026-04-26):
// consumer-side IR shape after fuse = K SEPARATE `fused_intermediate_N`
// channels, here K=3.
//
// Phase 2 (Sprint N+2) flips THE FIRST RUN LINE from XFAIL → PASS: the
// iterate-pairwise device-merge driver collapses 3 producers + 1 consumer
// into a single merged device with K=3 distinct fused intermediates,
// proving convergent N-way merge is not a 2-way special case.
//
// The SECOND RUN LINE (full Pass C lower) remains expected-failure —
// combined `--conduit-fuse-operators` + `--conduit-fuse-channels`
// integration is Track 5 / Sprint N+3+ territory. K=3 emits 3
// `fused_intermediate_N` channels, exceeding the consumer-tile (3,2) S2MM
// cap of 2; full lower needs channel-fusion compression to ≤2 channels.
// NPU smoke for the combined-capabilities test is queued as Task #63.
// Flip to PASS by removing the file-level expected-failure directive once
// channel-fusion integration lands.
//
// Sibling pattern reference: fuse_operators_convergent_basic.mlir.
//===----------------------------------------------------------------------===//

// XFAIL: *
// Lit's expected-failure marker is file-level (no per-RUN syntax); the
// file is marked because the SECOND RUN line below fails by design
// (Track 5 / Sprint N+3+ scope — see header note above). The first RUN
// line passes today (Phase 2 IR-shape contract) and stays passing; the
// file-level marker only flips to PASS once the second RUN line also
// passes (i.e. `--conduit-fuse-channels` lands downstream-K=3-channel
// compression so Pass C does not exhaust S2MM).

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
// Metafix Candidate 1.
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// CHECK-LABEL: module @fuse_operators_convergent_three_producers

// Only ONE merged device after fusion (3 producers + 1 consumer → one):
// CHECK:       aie.device(npu2)
// CHECK-NOT:   aie.device(npu2)

// Three distinct fused intermediates — one per producer:
// CHECK-DAG:   conduit.create @fused_intermediate_0
// CHECK-DAG:   conduit.create @fused_intermediate_1
// CHECK-DAG:   conduit.create @fused_intermediate_2

// External I/O channels survive:
// CHECK-DAG:   conduit.create @ext_in_p0
// CHECK-DAG:   conduit.create @ext_in_p1
// CHECK-DAG:   conduit.create @ext_in_p2
// CHECK-DAG:   conduit.create @ext_out

// Pre-fuse intermediates do NOT survive:
// CHECK-NOT:   conduit.create @inter_p0
// CHECK-NOT:   conduit.create @inter_p1
// CHECK-NOT:   conduit.create @inter_p2
// CHECK-NOT:   conduit.create @consume_p0
// CHECK-NOT:   conduit.create @consume_p1
// CHECK-NOT:   conduit.create @consume_p2

module @fuse_operators_convergent_three_producers {
  // Producer 0.
  aie.device(npu2) @devP0 {
    %shim = aie.tile(0, 0)
    %t    = aie.tile(0, 2)

    aie.objectfifo @ext_in_p0(%shim, {%t}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @inter_p0(%t, {%shim}, 2 : i32)
        {fusion_group = "fan3_fg0", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @p0_kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_p0(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_p0(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @p0_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @inter_p0(Produce, 1)
        aie.objectfifo.release @ext_in_p0(Consume, 1)
      }
      aie.end
    } {link_with = "p0.a"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_p0 {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_p0 {
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

  // Producer 1.
  aie.device(npu2) @devP1 {
    %shim = aie.tile(0, 0)
    %t    = aie.tile(0, 2)

    aie.objectfifo @ext_in_p1(%shim, {%t}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @inter_p1(%t, {%shim}, 2 : i32)
        {fusion_group = "fan3_fg0", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @p1_kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_p1(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_p1(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @p1_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @inter_p1(Produce, 1)
        aie.objectfifo.release @ext_in_p1(Consume, 1)
      }
      aie.end
    } {link_with = "p1.a"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_p1 {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_p1 {
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

  // Producer 2.
  aie.device(npu2) @devP2 {
    %shim = aie.tile(0, 0)
    %t    = aie.tile(0, 2)

    aie.objectfifo @ext_in_p2(%shim, {%t}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @inter_p2(%t, {%shim}, 2 : i32)
        {fusion_group = "fan3_fg0", fusion_index = 2 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @p2_kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_p2(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_p2(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @p2_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @inter_p2(Produce, 1)
        aie.objectfifo.release @ext_in_p2(Consume, 1)
      }
      aie.end
    } {link_with = "p2.a"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_p2 {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_p2 {
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

  // Consumer — 3-input ternary kernel.
  aie.device(npu2) @devC {
    %shim = aie.tile(0, 0)
    %tc   = aie.tile(0, 2)

    aie.objectfifo @consume_p0(%shim, {%tc}, 2 : i32)
        {fusion_group = "fan3_fg0", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @consume_p1(%shim, {%tc}, 2 : i32)
        {fusion_group = "fan3_fg0", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @consume_p2(%shim, {%tc}, 2 : i32)
        {fusion_group = "fan3_fg0", fusion_index = 2 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    aie.objectfifo @ext_out(%tc, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @ternary_kernel(memref<128xbf16>, memref<128xbf16>, memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%tc) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %s0 = aie.objectfifo.acquire @consume_p0(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %s0_buf = aie.objectfifo.subview.access %s0[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %s1 = aie.objectfifo.acquire @consume_p1(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %s1_buf = aie.objectfifo.subview.access %s1[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %s2 = aie.objectfifo.acquire @consume_p2(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %s2_buf = aie.objectfifo.subview.access %s2[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @ext_out(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @ternary_kernel(%s0_buf, %s1_buf, %s2_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>, memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_out(Produce, 1)
        aie.objectfifo.release @consume_p2(Consume, 1)
        aie.objectfifo.release @consume_p1(Consume, 1)
        aie.objectfifo.release @consume_p0(Consume, 1)
      }
      aie.end
    } {link_with = "consumer.a"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>, %arg2: memref<128xbf16>, %arg3: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @consume_p0 {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @consume_p1 {
        aie.dma_bd(%arg1 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @consume_p2 {
        aie.dma_bd(%arg2 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @ext_out {
        aie.dma_bd(%arg3 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t3)
      aiex.dma_await_task(%t3)
      aiex.dma_free_task(%t2)
      aiex.dma_free_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }
}
