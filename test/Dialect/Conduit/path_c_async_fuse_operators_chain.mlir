//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
// Metafix Candidate 1: also smoke through full Pass C + downstream
// legalization.  This is the highest-risk fuse pass for Path C — it is the
// rt-seq orchestrator merger (ConduitFuseOperators.cpp:349-455 / 1136-1200).
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// Path C async fuse-pass interaction pin (Task #33, design from
// path-c-test-matrix.md §1.2).
//
// Goal: a producer-consumer operator pair whose intermediate is fusible
// (`fusion_group` attr) MUST be merged into a single device by
// --conduit-fuse-operators while preserving the producer's WaitAll{token=true}
// (await) as the producer→consumer sync point inside the fused rt-seq.  This
// is V2 invariant pinning (Path C design §4.3): WaitAll{token=true} operands
// must all be defined by `get_memref_async` (S2MM await) and {token=false}
// operands must all be defined by `put_memref_async` (MM2S free).
//
// Input shape:
//   - DevA (producer): @ext_in (MM2S, free) → kernel → @inter_out (S2MM,
//     marked fusion_group="fg0", IRON-emitted dma_await_task).
//   - DevB (consumer): @inter_in (MM2S, fusion_group="fg0", free) → kernel →
//     @ext_out (S2MM, IRON-emitted dma_await_task).
//
// Expected output (after --conduit-fuse-operators):
//   - One aie.device only.
//   - @inter_out / @inter_in replaced by @fused_intermediate_0.
//   - Both producer and consumer WaitAll{token=true} (the awaits) survive in
//     source-relative position inside the merged rt-seq (Step 6/8 orchestrator
//     must list async-variants and route WaitAlls into the right arg-group
//     per ConduitFuseOperators.cpp:374-376 / 448).
//   - WaitAll{token=false} releases survive likewise.

// CHECK-LABEL: module @path_c_async_fuse_operators_chain

// One device after fusion (devA + devB merged).  Existing
// fuse_operators_basic.mlir convention: only one CHECK on aie.device(npu2).
// CHECK:       aie.device(npu2)

// Fused intermediate replaces the per-device intermediates; both external
// I/O channels survive on both sides of the fused chain.  Single DAG group
// so order of emission within Pass --conduit-fuse-operators is unconstrained.
// CHECK-DAG:   conduit.create @ext_in
// CHECK-DAG:   conduit.create @ext_out
// CHECK-DAG:   conduit.create @fused_intermediate_0
// CHECK-NOT:   conduit.create @inter_out
// CHECK-NOT:   conduit.create @inter_in

// In the fused rt-seq: producer-side put_memref_async for @ext_in must be
// followed by its WaitAll{token=false} release.  Consumer-side
// get_memref_async for @ext_out must be followed by its WaitAll (await,
// token=true elided to default).  Both sync points survive merging.
// CHECK:       conduit.put_memref_async
// CHECK-SAME:  name = @ext_in
// CHECK:       conduit.get_memref_async
// CHECK-SAME:  name = @ext_out

// At least two WaitAll ops survive — one release (token=false) and one await
// (token=true, default elided).  Without explicit ordering pins (Step 6
// orchestrator merge order is impl detail) we just count survival.
// CHECK:       conduit.wait_all
// CHECK:       conduit.wait_all

// No second aie.device after fusion.
// CHECK-NOT:   aie.device(npu2)

module @path_c_async_fuse_operators_chain {
  // DevA: producer (@ext_in MM2S free, @inter_out S2MM await).
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @ext_in(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @inter_out(%tile_0_2, {%shim_0}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @producer_kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_out(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @producer_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @inter_out(Produce, 1)
        aie.objectfifo.release @ext_in(Consume, 1)
      }
      aie.end
    } {link_with = "producer.a"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %inter: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_out {
        aie.dma_bd(%inter : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      // Producer await (becomes WaitAll{token=true}).
      aiex.dma_await_task(%t1)
      // Producer release (becomes WaitAll{token=false}).
      aiex.dma_free_task(%t0)
    }
  }

  // DevB: consumer (@inter_in MM2S free, @ext_out S2MM await).
  aie.device(npu2) @devB {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @inter_in(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @ext_out(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @consumer_kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @inter_in(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @ext_out(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @consumer_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_out(Produce, 1)
        aie.objectfifo.release @inter_in(Consume, 1)
      }
      aie.end
    } {link_with = "consumer.a"}

    aie.runtime_sequence(%inter: memref<128xbf16>, %arg1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @inter_in {
        aie.dma_bd(%inter : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_out {
        aie.dma_bd(%arg1 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      // Consumer await on @ext_out (becomes WaitAll{token=true}).
      aiex.dma_await_task(%t1)
      // Consumer release on @inter_in (becomes WaitAll{token=false}).
      aiex.dma_free_task(%t0)
    }
  }
}
