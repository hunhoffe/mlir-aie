//===- fallback_routine_error.mlir -----------------------------*- MLIR -*-===//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Check that aiecc emits useful diagnostics when on-tile buffer allocation
// fails: bank-aware allocator tries first and warns on fall-through, sequential
// allocator runs and also fails, and aiecc surfaces the MemoryMap so the user
// can see what was placed.
//
// The fixture declares 6 buffers totaling 36,864 bytes on a 32,768-byte AIE1
// data-memory tile (xcvc1902). No objectfifo / conduit ops — this isolates the
// allocator/diagnostic path from the lowering pipeline, so the test exercises
// the diagnostic surface area directly.

// RUN: not %python aiecc.py %s 2>&1 | FileCheck %s
// CHECK: warning: Bank-aware allocation failed, trying basic sequential allocation.
// CHECK: error: 'aie.tile' op allocated buffers exceeded available memory
// CHECK: note: see current operation: %{{.*}} = "aie.tile"() <{col = 1 : i32, row = 2 : i32}>
// CHECK: note: MemoryMap:
// CHECK-DAG: a {{.*}} (8192 bytes)
// CHECK-DAG: b {{.*}} (8192 bytes)
// CHECK-DAG: c {{.*}} (8192 bytes)
// CHECK-DAG: d {{.*}} (4096 bytes)
// CHECK-DAG: e {{.*}} (4096 bytes)
// CHECK-DAG: f {{.*}} (4096 bytes)
// CHECK: error: 'aie.tile' op Basic sequential allocation also failed.

module @test {
 aie.device(xcvc1902) {
  %tile12 = aie.tile(1, 2)
  %1 = aie.buffer(%tile12) { sym_name = "a" } : memref<2048xi32>  // 8192 bytes
  %2 = aie.buffer(%tile12) { sym_name = "b" } : memref<2048xi32>  // 8192 bytes
  %3 = aie.buffer(%tile12) { sym_name = "c" } : memref<2048xi32>  // 8192 bytes
  %4 = aie.buffer(%tile12) { sym_name = "d" } : memref<1024xi32>  // 4096 bytes
  %5 = aie.buffer(%tile12) { sym_name = "e" } : memref<1024xi32>  // 4096 bytes
  %6 = aie.buffer(%tile12) { sym_name = "f" } : memref<1024xi32>  // 4096 bytes  -- total 36864 > 32768
 }
}
