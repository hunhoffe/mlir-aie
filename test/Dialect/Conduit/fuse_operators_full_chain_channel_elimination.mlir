//===- fuse_operators_full_chain_channel_elimination.mlir ----*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Channel-merge regression for the spatial-fusion path (task #57).
//
// Background: in aiecc.cpp the spatial-fusion path was originally
//   `aie-combine-device,conduit-fuse-operators`.
// `aie-combine-device` runs FIRST and physically merges devB into devA, so
// `conduit-fuse-operators` then bails at its `devices.size() < 2` guard
// before its channel-merging logic (Steps 5-7: emit fused internal channel,
// rewrite acquires, delete the intermediate runtime DMA tasks) ever runs.
// The device merge fired but channel merge did NOT — the intermediate L3
// round-trip survived.
//
// Fix: drop the redundant `aie-combine-device` from the spatial path;
// `conduit-fuse-operators` does its own device-body merge via
// DeviceMergeUtils (FS3 helper) at Step 8.
//
// This test exercises the FULL fusion chain end-to-end and asserts that the
// intermediate channel becomes truly internal — i.e. NO
// `aie.shim_dma_allocation` is emitted for the fused intermediate after
// `conduit-to-dma`.  This is the lit coverage gap flagged in task #54
// (channel-merge investigation).
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit \
// RUN:         --conduit-fuse-operators --conduit-depth-promote \
// RUN:         --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: module @fuse_operators_full_chain_channel_elimination

// Only ONE device remains after fusion (devB merged into devA via
// DeviceMergeUtils inside conduit-fuse-operators):
// CHECK:       aie.device(npu2)
// CHECK-NOT:   aie.device(npu2)

// External-IO channels keep their shim allocations:
// CHECK-DAG:   aie.shim_dma_allocation @ext_in
// CHECK-DAG:   aie.shim_dma_allocation @ext_out

// The intermediate channels are fully internalized — NO shim allocation for
// either of the original inter_* names or the fused_intermediate symbol.
// (This is the load-bearing assertion: it would FAIL on the pre-fix
// pipeline because the surviving runtime DMA tasks for the intermediate
// would force conduit-to-dma to emit a shim_dma_allocation for it.)
// CHECK-NOT:   aie.shim_dma_allocation @inter_out
// CHECK-NOT:   aie.shim_dma_allocation @inter_in
// CHECK-NOT:   aie.shim_dma_allocation @fused_intermediate

module @fuse_operators_full_chain_channel_elimination {
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // External input: LPDDR5 → compute tile.
    aie.objectfifo @ext_in(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    // Intermediate output: compute tile → LPDDR5 (fusible).
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

    aie.runtime_sequence(%arg0: memref<128xbf16>, %intermediate: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_out {
        aie.dma_bd(%intermediate : memref<128xbf16>, 0, 128,
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

  aie.device(npu2) @devB {
    // shim at column 1 (distinct from devA's column-0 shim) so devB's
    // tile set is NOT a subset of devA's; this preserves the offset path
    // (devB → +colMaxA+1 = +1) the original fixture assumed.
    %shim_0 = aie.tile(1, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Intermediate input: LPDDR5 → compute tile (fusible, matching fusion_group).
    aie.objectfifo @inter_in(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

    // External output: compute tile → LPDDR5.
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

    aie.runtime_sequence(%intermediate: memref<128xbf16>, %arg1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @inter_in {
        aie.dma_bd(%intermediate : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_out {
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
}
