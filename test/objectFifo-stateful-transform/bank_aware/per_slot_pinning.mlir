//===- per_slot_pinning.mlir ------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-objectFifo-stateful-transform %s | FileCheck %s

// The stateful-transform pass propagates producer_mem_bank /
// consumer_mem_banks (per-slot) onto the generated aie.buffer ops in the
// same depth-loop order: buff_0 -> bank[0], buff_1 -> bank[1], ...
//
// We use MemTile-producer → CoreTile-consumer to ensure buffers land on
// both tiles (no shared-memory shortcut that places the buffer pool on a
// single tile).

// CHECK-LABEL: @producer_pinned
// CHECK: aie.buffer(%mem_tile_0_1) {mem_bank = 0 : i32, sym_name = "of_prod_pinned_buff_0"} : memref<16xi32>
// CHECK: aie.buffer(%mem_tile_0_1) {mem_bank = 1 : i32, sym_name = "of_prod_pinned_buff_1"} : memref<16xi32>
module @producer_pinned {
  aie.device(npu2) {
    %prod = aie.tile(0, 1)
    %cons = aie.tile(0, 2)
    aie.objectfifo @of_prod_pinned(%prod, {%cons}, 2 : i32)
      {producer_mem_bank = [0 : i32, 1 : i32]}
      : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Consumer pin lands on the consumer-side buffers (memTile producer + core
// consumer means the consumer-side buffer pool is distinct from the
// producer's MemTile-resident pool).
// CHECK-LABEL: @consumer_pinned
// CHECK: aie.buffer(%tile_0_2) {mem_bank = 2 : i32, sym_name = "of_cons_pinned_cons_buff_0"} : memref<16xi32>
// CHECK: aie.buffer(%tile_0_2) {mem_bank = 3 : i32, sym_name = "of_cons_pinned_cons_buff_1"} : memref<16xi32>
module @consumer_pinned {
  aie.device(npu2) {
    %prod = aie.tile(0, 1)
    %cons = aie.tile(0, 2)
    aie.objectfifo @of_cons_pinned(%prod, {%cons}, 2 : i32)
      {consumer_mem_banks = [[2 : i32, 3 : i32]]}
      : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// Mixed: cons_a pins, cons_b opts out via empty inner array. cons_a's
// buffers carry mem_bank; cons_b's must not.
// CHECK-LABEL: @mixed_pinning
// CHECK-DAG: aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "of_mixed_0_cons_buff_0"} : memref<16xi32>
// CHECK-DAG: aie.buffer(%tile_0_2) {mem_bank = 1 : i32, sym_name = "of_mixed_0_cons_buff_1"} : memref<16xi32>
// CHECK-DAG: aie.buffer(%tile_0_3) {sym_name = "of_mixed_1_cons_buff_0"} : memref<16xi32>
// CHECK-DAG: aie.buffer(%tile_0_3) {sym_name = "of_mixed_1_cons_buff_1"} : memref<16xi32>
module @mixed_pinning {
  aie.device(npu2) {
    %prod = aie.tile(0, 1)
    %cons_a = aie.tile(0, 2)
    %cons_b = aie.tile(0, 3)
    aie.objectfifo @of_mixed(%prod, {%cons_a, %cons_b}, 2 : i32)
      {consumer_mem_banks = [[0 : i32, 1 : i32], []]}
      : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// No pinning at all → no mem_bank attr on any buffer (byte-identical to
// pre-bank-aware lowering).
// CHECK-LABEL: @no_pinning
// CHECK-NOT: mem_bank
module @no_pinning {
  aie.device(npu2) {
    %prod = aie.tile(0, 1)
    %cons = aie.tile(0, 2)
    aie.objectfifo @of_default(%prod, {%cons}, 2 : i32)
      : !aie.objectfifo<memref<16xi32>>
  }
}
