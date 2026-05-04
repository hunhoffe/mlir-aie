//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s

// Validates that the bd_id allocator uses the configure_task's actual
// channel index (rather than always allocating from channel 0). On an npu2
// mem-tile the AIETargetModel partitions BDs by even/odd channel:
//
//   even channel -> BDs 0..23
//   odd  channel -> BDs 24..47
//
// So configures on channel 0 and channel 1 must allocate from disjoint
// ranges, and back-to-back overlapping intervals on the same channel must
// stay within their own range without ever crossing into the other pool.

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf_e0 = aie.buffer(%tile_0_1) {sym_name = "buf_e0"} : memref<32xi8>
    %buf_e1 = aie.buffer(%tile_0_1) {sym_name = "buf_e1"} : memref<32xi8>
    %buf_o0 = aie.buffer(%tile_0_1) {sym_name = "buf_o0"} : memref<32xi8>
    %buf_o1 = aie.buffer(%tile_0_1) {sym_name = "buf_o1"} : memref<32xi8>

    aie.runtime_sequence() {
      // Even-channel pool: BDs 0..23. Two overlapping intervals must pick
      // distinct IDs from this pool only (lowest available is 0, then 1).

      // CHECK: aie.dma_bd(%{{.*}} : memref<32xi8>, 0, 32) {bd_id = 0 : i32}
      %t_e0 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf_e0 : memref<32xi8>, 0, 32)
        aie.end
      }
      aiex.dma_start_task(%t_e0)

      // CHECK: aie.dma_bd(%{{.*}} : memref<32xi8>, 0, 32) {bd_id = 1 : i32}
      %t_e1 = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
        aie.dma_bd(%buf_e1 : memref<32xi8>, 0, 32)
        aie.end
      }
      aiex.dma_start_task(%t_e1)

      // Odd-channel pool: BDs 24..47. The lowest accessible BD on channel 1
      // is 24. With the previous (buggy) hardcoded channelIndex=0 this would
      // have allocated id 2 -- which the hardware cannot submit on channel
      // 1. The fix forces nextBdId(1) to skip BDs 0..23 entirely.

      // CHECK: aie.dma_bd(%{{.*}} : memref<32xi8>, 0, 32) {bd_id = 24 : i32}
      %t_o0 = aiex.dma_configure_task(%tile_0_1, MM2S, 1) {
        aie.dma_bd(%buf_o0 : memref<32xi8>, 0, 32)
        aie.end
      }
      aiex.dma_start_task(%t_o0)

      // CHECK: aie.dma_bd(%{{.*}} : memref<32xi8>, 0, 32) {bd_id = 25 : i32}
      %t_o1 = aiex.dma_configure_task(%tile_0_1, MM2S, 1) {
        aie.dma_bd(%buf_o1 : memref<32xi8>, 0, 32)
        aie.end
      }
      aiex.dma_start_task(%t_o1)

      aiex.dma_await_task(%t_e0)
      aiex.dma_await_task(%t_e1)
      aiex.dma_await_task(%t_o0)
      aiex.dma_await_task(%t_o1)
    }
  }
}
