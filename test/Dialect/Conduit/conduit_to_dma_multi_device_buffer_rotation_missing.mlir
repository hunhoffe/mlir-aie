//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-depth-promote --conduit-to-dma %s | FileCheck %s

// Pass C multi-device buffer-rotation regression pin.
//
// Bug source: ConduitToDMALower.cpp:106 silent fallback to static
// buffer selection when `tileRotationBuf == nullptr`. Root cause is in
// `prescanAndCreateRotationBufs` (ConduitToDMAAlloc.cpp:70) which fails
// to allocate rotation buffers for all devices in multi-device fused
// MLIR — only one device gets the rotation infrastructure; the others
// silently fall back to static buff_0 selection.
//
// Minimum reproducer: TWO device blocks in one module, each with
// identical structure (single shim + single compute, depth=2 ObjectFifo,
// kernel call inside loop forcing subview_access). The FIRST device's
// core silently produces wrong code (only `data_a__d0_cons_buff_0`,
// no scf.index_switch). The SECOND device's core correctly emits
// scf.index_switch alternating between buff_0 and buff_1.
//
// Discovered 2026-04-25 during Llama decode hang investigation.
// Full Llama decode fused-ELF MLIR has 20 device blocks; multiple
// have missing rotation including op0_RMSNorm and op2_GEMV. At runtime
// the cores silently drop every other iteration's data, cascading
// into firmware deadlock at FusedMLIROperator dispatch
// (ert_cmd_state.ERT_CMD_STATE_TIMEOUT).
//
// Verified bug shape via: extracting op2_GEMV section into its own
// module produces CORRECT rotation; same section in the original
// 20-device fused module produces NO rotation. Bug is purely in
// multi-device handling.
//
// Fix direction: in `prescanAndCreateRotationBufs`, ensure rotation
// buffers are allocated per-device (currently appears to only allocate
// for one device's tiles). State-tracking across `switchToDeviceIndex`
// calls likely the issue. Defense-in-depth: change line 106 fallback
// to ERROR loudly instead of silently using static selection.

// CHECK-LABEL: module

// FIRST device — must have rotation in core (was the bug pre-fix):
// CHECK: aie.device(npu2) @first
// CHECK: aie.core
// CHECK-DAG: scf.yield %data_a{{.*}}_cons_buff_0
// CHECK-DAG: scf.yield %data_a{{.*}}_cons_buff_1
// CHECK: aie.end

// SECOND device — already has correct rotation pre-fix:
// CHECK: aie.device(npu2) @second
// CHECK: aie.core
// CHECK-DAG: scf.yield %data_b{{.*}}_cons_buff_0
// CHECK-DAG: scf.yield %data_b{{.*}}_cons_buff_1
// CHECK: aie.end

module {
  aie.device(npu2) @first {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    aie.objectfifo @data_a(%shim, {%tile}, 2 : i32)
        : !aie.objectfifo<memref<32xbf16>>

    func.func private @kernel_a(memref<32xbf16>)

    aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %acq = aie.objectfifo.acquire @data_a(Consume, 1)
            : !aie.objectfifosubview<memref<32xbf16>>
        %buf = aie.objectfifo.subview.access %acq[0]
            : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
        func.call @kernel_a(%buf) : (memref<32xbf16>) -> ()
        aie.objectfifo.release @data_a(Consume, 1)
      }
      aie.end
    } { link_with = "kernel_a.o" }

    aie.runtime_sequence(%arg0: memref<128xbf16>) {
      %t = aiex.dma_configure_task_for @data_a {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }

  aie.device(npu2) @second {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    aie.objectfifo @data_b(%shim, {%tile}, 2 : i32)
        : !aie.objectfifo<memref<32xbf16>>

    func.func private @kernel_b(memref<32xbf16>)

    aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %acq = aie.objectfifo.acquire @data_b(Consume, 1)
            : !aie.objectfifosubview<memref<32xbf16>>
        %buf = aie.objectfifo.subview.access %acq[0]
            : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
        func.call @kernel_b(%buf) : (memref<32xbf16>) -> ()
        aie.objectfifo.release @data_b(Consume, 1)
      }
      aie.end
    } { link_with = "kernel_b.o" }

    aie.runtime_sequence(%arg0: memref<128xbf16>) {
      %t = aiex.dma_configure_task_for @data_b {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}
