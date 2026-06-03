//===- objectfifo_bank_aware_bad.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Verifier rejects malformed producer_mem_bank / consumer_mem_banks attrs.
//
// NPU2 compute tiles have 4 DM banks; anything in [4, ∞) is out of range.

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: 'aie.objectfifo' op `producer_mem_bank` length (3) must equal producer depth (2)
aie.device(npu2) {
  %prod = aie.tile(0, 2)
  %cons = aie.tile(0, 3)
  aie.objectfifo @bad_prod_len(%prod, {%cons}, 2 : i32)
    {producer_mem_bank = [0 : i32, 1 : i32, 2 : i32]}
    : !aie.objectfifo<memref<16xi32>>
}

// -----

// CHECK: 'aie.objectfifo' op `producer_mem_bank[1]` (7) out of range [0, 4)
aie.device(npu2) {
  %prod = aie.tile(0, 2)
  %cons = aie.tile(0, 3)
  aie.objectfifo @prod_oob(%prod, {%cons}, 2 : i32)
    {producer_mem_bank = [0 : i32, 7 : i32]}
    : !aie.objectfifo<memref<16xi32>>
}

// -----

// CHECK: 'aie.objectfifo' op `consumer_mem_banks` outer length must equal number of consumer tiles
aie.device(npu2) {
  %prod = aie.tile(0, 2)
  %cons_a = aie.tile(0, 3)
  %cons_b = aie.tile(0, 4)
  // Two consumers but only one inner array supplied.
  aie.objectfifo @bad_cons_outer(%prod, {%cons_a, %cons_b}, 2 : i32)
    {consumer_mem_banks = [[0 : i32, 1 : i32]]}
    : !aie.objectfifo<memref<16xi32>>
}

// -----

// CHECK: 'aie.objectfifo' op `consumer_mem_banks[0]` length (1) must equal consumer depth (2)
aie.device(npu2) {
  %prod = aie.tile(0, 2)
  %cons = aie.tile(0, 3)
  aie.objectfifo @bad_cons_inner_len(%prod, {%cons}, 2 : i32)
    {consumer_mem_banks = [[0 : i32]]}
    : !aie.objectfifo<memref<16xi32>>
}

// -----

// CHECK: 'aie.objectfifo' op `consumer_mem_banks[0][1]` (8) out of range [0, 4)
aie.device(npu2) {
  %prod = aie.tile(0, 2)
  %cons = aie.tile(0, 3)
  aie.objectfifo @cons_oob(%prod, {%cons}, 2 : i32)
    {consumer_mem_banks = [[0 : i32, 8 : i32]]}
    : !aie.objectfifo<memref<16xi32>>
}

// -----

// CHECK: 'aie.objectfifo' op `producer_mem_bank[0]` (-1) out of range [0, 4)
aie.device(npu2) {
  %prod = aie.tile(0, 2)
  %cons = aie.tile(0, 3)
  aie.objectfifo @prod_negative(%prod, {%cons}, 2 : i32)
    {producer_mem_bank = [-1 : i32, 0 : i32]}
    : !aie.objectfifo<memref<16xi32>>
}

// -----

// Per-endpoint depths: producer depth=3, consumer depth=2. Producer-side
// bank list must match producer depth, not consumer depth.
// CHECK: 'aie.objectfifo' op `producer_mem_bank` length (2) must equal producer depth (3)
aie.device(npu2) {
  %prod = aie.tile(0, 2)
  %cons = aie.tile(0, 3)
  aie.objectfifo @per_endpoint_bad(%prod, {%cons}, [3 : i32, 2 : i32])
    {producer_mem_bank = [0 : i32, 1 : i32]}
    : !aie.objectfifo<memref<16xi32>>
}

// -----

// Empty inner array is the explicit "this consumer opts out" form and must
// be accepted regardless of producer/other-consumer pinning.
// CHECK-NOT: error
// CHECK: aie.objectfifo @cons_opt_out
aie.device(npu2) {
  %prod = aie.tile(0, 1)
  %cons_a = aie.tile(0, 2)
  %cons_b = aie.tile(0, 3)
  aie.objectfifo @cons_opt_out(%prod, {%cons_a, %cons_b}, 2 : i32)
    {consumer_mem_banks = [[0 : i32, 1 : i32], []]}
    : !aie.objectfifo<memref<16xi32>>
}
