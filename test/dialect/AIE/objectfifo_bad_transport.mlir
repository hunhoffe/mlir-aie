//===- objectfifo_bad_transport.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// The switches this replaced would otherwise be carried along as unknown
// attributes and silently ignored, so each is named and refused.
module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`via_DMA` has been replaced by `transport`}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {via_DMA = true} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`aie_stream` has been replaced by `transport`}}
    aie.objectfifo @of (%tile12, {%tile13}, 2 : i32) {aie_stream = 0 : i32} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

// A stream reaches exactly one consumer, because a stream port is a wire.
module {
 aie.device(xcve2302) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)
    %tile23 = aie.tile(2, 3)
    // expected-error@+1 {{a stream transport can only be used in 1-to-1 object FIFOs}}
    aie.objectfifo @of (%tile12, {%tile13, %tile23}, 2 : i32) {transport = #aie.transport<stream, ends = both, port = 0>} : !aie.objectfifo<memref<16xi32>>
 }
}

// -----

// Only compute tiles have stream ports.
module {
 aie.device(xcve2302) {
    %tile11 = aie.tile(1, 1)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{a stream transport is not available for shim and mem tiles}}
    aie.objectfifo @of (%tile11, {%tile13}, 2 : i32) {transport = #aie.transport<stream, ends = producer, port = 0>} : !aie.objectfifo<memref<16xi32>>
 }
}
