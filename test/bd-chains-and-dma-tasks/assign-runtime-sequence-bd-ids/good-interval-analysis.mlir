//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// Validates the interval-analysis bd_id assigner:
//   (a) Two configure_task intervals whose lifetimes do NOT overlap on the
//       same (tile, channel) get the SAME bd_id (recycled on free).
//   (b) Configure_task intervals whose lifetimes DO overlap on the same
//       (tile, channel) get DIFFERENT bd_ids.
//
// dma_await_task auto-inserts a dma_free_task after itself, so the await on
// %t1 below ends t1's interval before %t2 is configured.

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<32xi16>, %arg1: memref<32xi16>) {

      // ===== (a) Sequential, non-overlapping: bd_id reused. =====

      // CHECK: aie.dma_bd(%arg0 : memref<32xi16>, 0, 32) {bd_id = 0 : i32}
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<32xi16>, 0, 32)
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)

      // %t2's interval starts after the auto-inserted free of %t1, so the
      // generator has bd_id 0 available again. This proves invariant (a).
      // CHECK: aie.dma_bd(%arg1 : memref<32xi16>, 0, 32) {bd_id = 0 : i32}
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg1 : memref<32xi16>, 0, 32)
        aie.end
      }
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)

      // ===== (b) Two overlapping intervals: distinct bd_ids. =====
      //
      // %t3 and %t4 are both configured before either is awaited, so their
      // intervals overlap and they must receive different bd_ids. After both
      // are awaited (and their auto-inserted frees fire), bd_ids are
      // recycled, so %t5 picks up id 0 again.

      // CHECK: aie.dma_bd(%arg0 : memref<32xi16>, 0, 32) {bd_id = 0 : i32}
      %t3 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<32xi16>, 0, 32)
        aie.end
      }
      aiex.dma_start_task(%t3)

      // CHECK: aie.dma_bd(%arg1 : memref<32xi16>, 0, 32) {bd_id = 1 : i32}
      %t4 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg1 : memref<32xi16>, 0, 32)
        aie.end
      }
      aiex.dma_start_task(%t4)

      aiex.dma_await_task(%t3)
      aiex.dma_await_task(%t4)

      // CHECK: aie.dma_bd(%arg0 : memref<32xi16>, 0, 32) {bd_id = 0 : i32}
      %t5 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<32xi16>, 0, 32)
        aie.end
      }
      aiex.dma_start_task(%t5)
      aiex.dma_await_task(%t5)
    }
  }
}
