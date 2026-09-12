//===- objectfifo_transport.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// Every mode round-trips, and the stream mode carries the ends and port that
// only it accepts.

// CHECK-LABEL: @transport_modes
// CHECK: aie.objectfifo @auto_path
// CHECK-NOT: transport
// CHECK: aie.objectfifo @dma_path{{.*}}transport = #aie.transport<dma>
// CHECK: aie.objectfifo @shared_path{{.*}}transport = #aie.transport<shared_mem>
// CHECK: aie.objectfifo @stream_path{{.*}}transport = #aie.transport<stream, ends = both, port = 1>
module @transport_modes {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @auto_path (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @dma_path (%tile12, {%tile22}, 2 : i32) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @shared_path (%tile22, {%tile23}, 2 : i32) {transport = #aie.transport<shared_mem>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @stream_path (%tile13, {%tile23}, 2 : i32) {transport = #aie.transport<stream, ends = both, port = 1>} : !aie.objectfifo<memref<16xi32>>
 }
}
