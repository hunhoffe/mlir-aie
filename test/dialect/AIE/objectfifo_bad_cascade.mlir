//===- objectfifo_bad_cascade.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A cascade is a register handed between two neighbouring cores, so most of
// what a fifo can ask for has nowhere to land.

module {
 aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)
    %tile23 = aie.tile(2, 3)
    // expected-error@+1 {{a cascade transport is point to point, so it takes exactly one consumer tile}}
    aie.objectfifo @cas (%tile03, {%tile13, %tile23}, 1 : i32) {transport = #aie.transport<cascade>}
        : !aie.objectfifo<memref<1xvector<16xi32>>>
 }
}

// -----

module {
 aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{a cascade transport holds no objects of its own, so its depth is 1}}
    aie.objectfifo @cas (%tile03, {%tile13}, 2 : i32) {transport = #aie.transport<cascade>}
        : !aie.objectfifo<memref<1xvector<16xi32>>>
 }
}

// -----

module {
 aie.device(npu1) {
    %mem01 = aie.tile(0, 1)
    %tile03 = aie.tile(0, 3)
    // expected-error@+1 {{shim and mem tiles have no cascade interface}}
    aie.objectfifo @cas (%mem01, {%tile03}, 1 : i32) {transport = #aie.transport<cascade>}
        : !aie.objectfifo<memref<1xvector<16xi32>>>
 }
}

// -----

module {
 aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile33 = aie.tile(3, 3)
    // expected-error@+1 {{a cascade transport runs between neighbouring tiles, and these are not adjacent}}
    aie.objectfifo @cas (%tile03, {%tile33}, 1 : i32) {transport = #aie.transport<cascade>}
        : !aie.objectfifo<memref<1xvector<16xi32>>>
 }
}

// -----

// The wire carries one accumulator-width value, 512 bits on this target.
module {
 aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{a cascade transport carries one 512-bit value on this target}}
    aie.objectfifo @cas (%tile03, {%tile13}, 1 : i32) {transport = #aie.transport<cascade>}
        : !aie.objectfifo<memref<1xi32>>
 }
}

// -----

module {
 aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{a cascade transport carries one value, so its element type holds one element}}
    aie.objectfifo @cas (%tile03, {%tile13}, 1 : i32) {transport = #aie.transport<cascade>}
        : !aie.objectfifo<memref<4xvector<16xi32>>>
 }
}

// -----

module {
 aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`repeat_count` has nothing to act on in a cascade transport}}
    aie.objectfifo @cas (%tile03, {%tile13}, 1 : i32) {transport = #aie.transport<cascade>, repeat_count = 2 : i32}
        : !aie.objectfifo<memref<1xvector<16xi32>>>
 }
}

// -----

module {
 aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)
    // expected-error@+1 {{`packet` belongs to a dma or auto transport, not cascade}}
    aie.objectfifo @cas (%tile03, {%tile13}, 1 : i32) {transport = #aie.transport<cascade, packet = <>>}
        : !aie.objectfifo<memref<1xvector<16xi32>>>
 }
}
