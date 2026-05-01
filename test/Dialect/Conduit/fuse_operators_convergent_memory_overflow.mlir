//===- fuse_operators_convergent_memory_overflow.mlir -------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Track 3 — convergent merge L1-overflow → memtile-relay regression.
//
// Synthetic large-depth fixture: each producer/consumer carries a depth-8
// 8192-element bf16 intermediate (= 128 KiB per-channel). With K=2
// producers fanning into one consumer compute tile, the consumer's L1
// staging budget would be exceeded if both fused intermediates are pinned
// to L1. Track 3's L1→L2 demotion path must detect this and insert a
// MemTile relay (per Q5: Track 3 owns placement; depth-promote does NOT
// re-delegate to it).
//
// Phase 2 of Track 3 (Sprint N+2) lands the convergent matcher + N-way
// pairwise driver in case-A only (per-channel L1 budget assumed to fit;
// no L2 fallback).  Phase 3+ owns the L1→L2 demotion path and the
// MemTile-relay materialization the CHECK lines below pin.  This fixture
// stays XFAIL until then so the post-impl shape is pinned ahead of the
// implementation.  The conduit-dev landing the demotion path removes the
// XFAIL line.
//
// CHECK lines below describe the post-impl shape: a memtile (row 1) is
// materialized in the merged device, AND the fused intermediates carry a
// memtile-staging marker (`shows up as a MemTile aie.tile in the fused
// device`). The exact memtile-relay attribute spelling is left to the
// implementer; the load-bearing CHECK is `aie.tile(<col>, 1)` appearing
// in the fused device.
//
// Sibling pattern reference: fuse_operators_convergent_basic.mlir,
// path_c_relay_tile_dropped_by_pass_a.mlir (memtile relay materialization).
//===----------------------------------------------------------------------===//

// XFAIL: *

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
// Metafix Candidate 1.
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// CHECK-LABEL: module @fuse_operators_convergent_memory_overflow

// Only ONE merged device:
// CHECK:       aie.device(npu2)
// CHECK-NOT:   aie.device(npu2)

// Track 3 must materialize a memtile (row 1) for L2 staging because the
// per-tile L1 budget is exceeded by 2 × depth=8 × 8192 bf16 buffers:
// CHECK-DAG:   aie.tile({{[0-9]+}}, 1)

// Two fused intermediates still distinct (one per producer):
// CHECK-DAG:   conduit.create @fused_intermediate_0
// CHECK-DAG:   conduit.create @fused_intermediate_1

module @fuse_operators_convergent_memory_overflow {
  // Producer 0 — large-depth output.
  aie.device(npu2) @devGate {
    %shim = aie.tile(0, 0)
    %t    = aie.tile(0, 2)

    aie.objectfifo @ext_in_gate(%shim, {%t}, 2 : i32)
        : !aie.objectfifo<memref<8192xbf16>>
    aie.objectfifo @inter_gate(%t, {%shim}, 8 : i32)
        {fusion_group = "big_fg", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<8192xbf16>>

    func.func private @gate_kernel(memref<8192xbf16>, memref<8192xbf16>)

    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_gate(Consume, 1)
            : !aie.objectfifosubview<memref<8192xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<8192xbf16>> -> memref<8192xbf16>
        %out = aie.objectfifo.acquire @inter_gate(Produce, 1)
            : !aie.objectfifosubview<memref<8192xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<8192xbf16>> -> memref<8192xbf16>
        func.call @gate_kernel(%in_buf, %out_buf)
            : (memref<8192xbf16>, memref<8192xbf16>) -> ()
        aie.objectfifo.release @inter_gate(Produce, 1)
        aie.objectfifo.release @ext_in_gate(Consume, 1)
      }
      aie.end
    } {link_with = "gate.a"}

    aie.runtime_sequence(%arg0: memref<8192xbf16>, %arg1: memref<8192xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_gate {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 0, 8192,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 8192, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_gate {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 8192,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 8192, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  // Producer 1 — large-depth output.
  aie.device(npu2) @devUp {
    %shim = aie.tile(0, 0)
    %t    = aie.tile(0, 2)

    aie.objectfifo @ext_in_up(%shim, {%t}, 2 : i32)
        : !aie.objectfifo<memref<8192xbf16>>
    aie.objectfifo @inter_up(%t, {%shim}, 8 : i32)
        {fusion_group = "big_fg", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<8192xbf16>>

    func.func private @up_kernel(memref<8192xbf16>, memref<8192xbf16>)

    %core = aie.core(%t) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in_up(Consume, 1)
            : !aie.objectfifosubview<memref<8192xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<8192xbf16>> -> memref<8192xbf16>
        %out = aie.objectfifo.acquire @inter_up(Produce, 1)
            : !aie.objectfifosubview<memref<8192xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<8192xbf16>> -> memref<8192xbf16>
        func.call @up_kernel(%in_buf, %out_buf)
            : (memref<8192xbf16>, memref<8192xbf16>) -> ()
        aie.objectfifo.release @inter_up(Produce, 1)
        aie.objectfifo.release @ext_in_up(Consume, 1)
      }
      aie.end
    } {link_with = "up.a"}

    aie.runtime_sequence(%arg0: memref<8192xbf16>, %arg1: memref<8192xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_up {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 0, 8192,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 8192, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_up {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 8192,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 8192, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  // Consumer — large depth on both inputs forces L2 staging.
  aie.device(npu2) @devMul {
    %shim = aie.tile(0, 0)
    %tc   = aie.tile(0, 2)

    aie.objectfifo @consume_gate(%shim, {%tc}, 8 : i32)
        {fusion_group = "big_fg", fusion_index = 0 : i32}
        : !aie.objectfifo<memref<8192xbf16>>
    aie.objectfifo @consume_up(%shim, {%tc}, 8 : i32)
        {fusion_group = "big_fg", fusion_index = 1 : i32}
        : !aie.objectfifo<memref<8192xbf16>>
    aie.objectfifo @ext_out(%tc, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<8192xbf16>>

    func.func private @mul_kernel(memref<8192xbf16>, memref<8192xbf16>, memref<8192xbf16>)

    %core = aie.core(%tc) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %g = aie.objectfifo.acquire @consume_gate(Consume, 1)
            : !aie.objectfifosubview<memref<8192xbf16>>
        %g_buf = aie.objectfifo.subview.access %g[0]
            : !aie.objectfifosubview<memref<8192xbf16>> -> memref<8192xbf16>
        %u = aie.objectfifo.acquire @consume_up(Consume, 1)
            : !aie.objectfifosubview<memref<8192xbf16>>
        %u_buf = aie.objectfifo.subview.access %u[0]
            : !aie.objectfifosubview<memref<8192xbf16>> -> memref<8192xbf16>
        %out = aie.objectfifo.acquire @ext_out(Produce, 1)
            : !aie.objectfifosubview<memref<8192xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<8192xbf16>> -> memref<8192xbf16>
        func.call @mul_kernel(%g_buf, %u_buf, %out_buf)
            : (memref<8192xbf16>, memref<8192xbf16>, memref<8192xbf16>) -> ()
        aie.objectfifo.release @ext_out(Produce, 1)
        aie.objectfifo.release @consume_up(Consume, 1)
        aie.objectfifo.release @consume_gate(Consume, 1)
      }
      aie.end
    } {link_with = "mul.a"}

    aie.runtime_sequence(%arg0: memref<8192xbf16>, %arg1: memref<8192xbf16>, %arg2: memref<8192xbf16>) {
      %t0 = aiex.dma_configure_task_for @consume_gate {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 0, 8192,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 8192, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @consume_up {
        aie.dma_bd(%arg1 : memref<8192xbf16>, 0, 8192,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 8192, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @ext_out {
        aie.dma_bd(%arg2 : memref<8192xbf16>, 0, 8192,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 8192, stride = 1>])
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
