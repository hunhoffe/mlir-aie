//===- rtp_passthrough_bare_conduit.mlir ----------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Pins the RTP machinery (aie.buffer marked use_write_rtp + aie.lock as
// host-released barrier + core-side aie.use_lock + memref.load of the RTP
// buffer + aiex.npu.rtp_write + aiex.set_lock) through bare `--use-conduit`
// unchanged.  None of the conduit passes own these ops, but the pipeline
// must accept them and emit them verbatim into the lowered IR.  This test
// is the regression net for Track 5 / Pattern D / RTP-aware `dma_repeat`
// work — if any future conduit pass starts mutating, dropping, or
// reordering RTP machinery, this test will catch it.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s

// CHECK-LABEL: aie.device(npu1)
// CHECK:       aie.buffer({{.*}}) {sym_name = "my_rtp", use_write_rtp = true} : memref<2xi32>
// CHECK:       %[[BARRIER:.*]] = aie.lock({{.*}}) {sym_name = "my_barrier"}
// CHECK:       aie.core
// CHECK:         aie.use_lock(%[[BARRIER]], Acquire, 1)
// CHECK:         memref.load %{{.*}}[%{{.*}}] : memref<2xi32>
// CHECK:         aie.end
// CHECK:       aie.runtime_sequence
// CHECK:         aiex.npu.rtp_write(@my_rtp, 0, 42)
// CHECK:         aiex.set_lock(%[[BARRIER]], 1)

module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)

    %my_rtp = aie.buffer(%tile_0_2) {sym_name = "my_rtp", use_write_rtp = true} : memref<2xi32>
    %my_barrier = aie.lock(%tile_0_2) {sym_name = "my_barrier"}

    %core_0_2 = aie.core(%tile_0_2) {
      // Wait for the host to release the barrier to value 1.
      aie.use_lock(%my_barrier, Acquire, 1)

      // Load the runtime parameter the host wrote.
      %c0 = arith.constant 0 : index
      %rtp_val = memref.load %my_rtp[%c0] : memref<2xi32>

      // Trivial use of the RTP value so it is not DCE'd: count from 0 to
      // %rtp_val by 1 (loop body is empty).
      %lo = arith.constant 0 : i32
      %step = arith.constant 1 : i32
      scf.for %i = %lo to %rtp_val step %step : i32 {
      }

      aie.end
    }

    aie.runtime_sequence() {
      aiex.npu.rtp_write(@my_rtp, 0, 42)
      aiex.set_lock(%my_barrier, 1)
    }
  }
}
