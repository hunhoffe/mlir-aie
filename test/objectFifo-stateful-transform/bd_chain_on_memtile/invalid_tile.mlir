//===- bd_chain_on_memtile/invalid_tile.mlir ---*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// iter_count is now supported on all tile types (compute, MemTile, ShimTile).
// This test verifies that iter_count on a compute tile objectfifo compiles and
// produces a finite BD chain with the correct repeat_count on aie.dma_start.

// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK: aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 4)

module @objectfifo_iter_count_on_compute_tile {
 aie.device(npu1) {
    %tile13 = aie.tile(1, 2)
    %tile14 = aie.tile(1, 3)

    // iter_count=5: 5 passes × depth 2 = 10 BD slots; dma_start repeat_count = 4
    aie.objectfifo @of_0 (%tile13, {%tile14}, 2 : i32) {iter_count = 5 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}
