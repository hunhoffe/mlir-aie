//===- objectfifo_bank_aware_roundtrip.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Round-trip an aie.objectfifo with per-slot producer_mem_bank /
// consumer_mem_banks. The consumer_mem_banks outer array may contain an
// empty inner array to opt one consumer out of pinning while another
// consumer pins its slots.

// RUN: aie-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @prod_only
// CHECK: aie.objectfifo @of_prod_pinned
// CHECK-SAME: producer_mem_bank = [0 : i32, 1 : i32]

module @prod_only {
  aie.device(npu2) {
    %prod = aie.tile(0, 2)
    %cons = aie.tile(0, 3)
    aie.objectfifo @of_prod_pinned(%prod, {%cons}, 2 : i32)
      {producer_mem_bank = [0 : i32, 1 : i32]}
      : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// CHECK-LABEL: @cons_only
// CHECK: aie.objectfifo @of_cons_pinned
// CHECK-SAME: consumer_mem_banks = {{\[\[}}2 : i32, 3 : i32]]

module @cons_only {
  aie.device(npu2) {
    %prod = aie.tile(0, 2)
    %cons = aie.tile(0, 3)
    aie.objectfifo @of_cons_pinned(%prod, {%cons}, 2 : i32)
      {consumer_mem_banks = [[2 : i32, 3 : i32]]}
      : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// CHECK-LABEL: @mixed_pinning
// CHECK: aie.objectfifo @of_mixed
// CHECK-SAME: consumer_mem_banks = {{\[\[}}1 : i32, 2 : i32], []]

module @mixed_pinning {
  aie.device(npu2) {
    %prod = aie.tile(0, 1)
    %cons_a = aie.tile(0, 2)
    %cons_b = aie.tile(0, 3)
    // cons_a pins slots [1, 2]; cons_b opts out (empty inner array).
    aie.objectfifo @of_mixed(%prod, {%cons_a, %cons_b}, 2 : i32)
      {consumer_mem_banks = [[1 : i32, 2 : i32], []]}
      : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// CHECK-LABEL: @per_endpoint_depths
// CHECK: aie.objectfifo @of_per_endpoint_depths
// CHECK-SAME: [3 : i32, 2 : i32]
// CHECK-SAME: consumer_mem_banks = {{\[\[}}3 : i32, 0 : i32]]
// CHECK-SAME: producer_mem_bank = [0 : i32, 1 : i32, 2 : i32]

module @per_endpoint_depths {
  // Per-endpoint depths ([prod, cons0]); per-slot bank lengths match each.
  aie.device(npu2) {
    %prod = aie.tile(0, 2)
    %cons = aie.tile(0, 3)
    aie.objectfifo @of_per_endpoint_depths(%prod, {%cons}, [3 : i32, 2 : i32])
      {producer_mem_bank = [0 : i32, 1 : i32, 2 : i32],
       consumer_mem_banks = [[3 : i32, 0 : i32]]}
      : !aie.objectfifo<memref<16xi32>>
  }
}
