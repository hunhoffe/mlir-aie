//===- bad_pin_with_stream_transport.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// A stream transport routes through stream ports, which bypass DMA channels, so
// a DMA channel on the same endpoint is contradictory and is rejected.

// CHECK: error: 'aie.objectfifo' op cannot pin a DMA channel on an objectfifo that also uses a stream transport (stream ports bypass DMA channels)

module @bad_pin_with_stream_transport {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @of (%tile12, {%tile33}, 2 : i32) {prod_dma_channel = 0 : i32, transport = #aie.transport<stream, ends = producer, port = 0>} : !aie.objectfifo<memref<16xi32>>
  }
}
