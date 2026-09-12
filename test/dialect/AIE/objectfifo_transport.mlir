//===- objectfifo_transport.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt %s | FileCheck %s

// Every mode round-trips, the stream mode carries the ends and port that only
// it accepts, and the DMA path carries a packet header, open or pinned.

// CHECK-LABEL: @transport_modes
// CHECK: aie.objectfifo @auto_path
// CHECK-NOT: transport
// CHECK: aie.objectfifo @dma_path{{.*}}transport = #aie.transport<dma>
// CHECK: aie.objectfifo @shared_path{{.*}}transport = #aie.transport<shared_mem>
// CHECK: aie.objectfifo @stream_path{{.*}}transport = #aie.transport<stream, ends = both, port = 1>
// CHECK: aie.objectfifo @packet_path{{.*}}transport = #aie.transport<dma, packet = <>>
// CHECK: aie.objectfifo @pinned_path{{.*}}transport = #aie.transport<auto, packet = <pkt_id = 7>>
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
    aie.objectfifo @packet_path (%tile12, {%tile23}, 2 : i32) {transport = #aie.transport<dma, packet = <>>} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @pinned_path (%tile13, {%tile22}, 2 : i32) {transport = #aie.transport<auto, packet = <pkt_id = 7>>} : !aie.objectfifo<memref<16xi32>>
 }
}
