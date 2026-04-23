//===- passC_multi_device_shim_alloc.mlir --------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// FS2 regression: ConduitToDMAState::switchToDeviceIndex() updated
// activeDevIdx and deviceBody but did NOT update state.deviceOp.  Every
// SymbolTable::lookupSymbolIn() call in routePhase therefore checked the
// FIRST device's symbol table even when emitting into a non-first device.
// On the FuseMLIROperator 3-device input (op0 + op1 + host), the lookup for
// `<chan>_shim_alloc` on device @op1_* missed the pre-existing symbol
// (placed there by objectfifo-to-conduit) and Pass C emitted a duplicate
// shim_dma_allocation → "redefinition of symbol" verifier crash.
//
// This test exercises the multi-device path with shim conduits in TWO
// devices.  Before the FS2 fix, conduit-to-dma crashed during routePhase
// on @dev1.  After the fix, both devices lower cleanly with exactly one
// shim_dma_allocation per channel.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s

// Two devices survive Pass C, each with its own shim_dma_allocation; no
// duplicate symbol is emitted in either device.

// CHECK-LABEL: aie.device(npu2) @dev0
// CHECK:       aie.shim_dma_allocation @of0
// CHECK-NOT:   aie.shim_dma_allocation @of0
// CHECK-LABEL: aie.device(npu2) @dev1
// CHECK:       aie.shim_dma_allocation @of1
// CHECK-NOT:   aie.shim_dma_allocation @of1

module @fs2_multi_device_shim_alloc {
  aie.device(npu2) @dev0 {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @of0(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @of0(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %el = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      aie.objectfifo.release @of0(Consume, 1)
      aie.end
    }
  }

  aie.device(npu2) @dev1 {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @of1(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.core(%tile_0_2) {
      %sv = aie.objectfifo.acquire @of1(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %el = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      aie.objectfifo.release @of1(Consume, 1)
      aie.end
    }
  }
}
