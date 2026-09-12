//===- test_place_routes.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-place-tiles %s | FileCheck %s

// A route is a net like a flow: the shim lands in the column of the core its
// route reaches, and a pool on the core is memory there.

// CHECK-LABEL: @route_shim_near_core
module @route_shim_near_core {
  aie.device(npu1) {
    // CHECK-DAG: %[[CORE:.*]] = aie.tile(3, 2)
    %core = aie.logical_tile<CoreTile>(3, 2)
    // CHECK-DAG: %[[SHIM:.*]] = aie.tile(3, 0)
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.route_endpoint @in(%shim) DMA
    aie.objectfifo.pool @p(%core) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @p_dma(%core) fills @p
    aie.objectfifo.core_endpoint @p_core(%core) drains @p
    // CHECK: aie.route from @in to [@p_dma]
    aie.route from @in to [@p_dma]
    // CHECK-NOT: aie.logical_tile
  }
}

// -----

// A fan-in route makes the mem tile a peer of both cores, so it lands between
// them, and it spends one input channel on the mem tile, not one per source.

// CHECK-LABEL: @fan_in_memtile_between_cores
module @fan_in_memtile_between_cores {
  aie.device(npu1) {
    // CHECK-DAG: %[[C1:.*]] = aie.tile(1, 2)
    %c1 = aie.logical_tile<CoreTile>(1, 2)
    // CHECK-DAG: %[[C2:.*]] = aie.tile(3, 2)
    %c2 = aie.logical_tile<CoreTile>(3, 2)
    // CHECK-DAG: %[[MEM:.*]] = aie.tile(2, 1)
    %mem = aie.logical_tile<MemTile>(?, ?)
    aie.objectfifo.pool @a(%c1) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @a_core(%c1) fills @a
    aie.objectfifo.dma_endpoint @a_dma(%c1) drains @a
    aie.objectfifo.pool @b(%c2) {depth = 1 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.core_endpoint @b_core(%c2) fills @b
    aie.objectfifo.dma_endpoint @b_dma(%c2) drains @b
    aie.objectfifo.pool @m(%mem) {depth = 2 : i32} : memref<16xi32> {
      aie.objectfifo.segment @s0 {offset = 0 : i32, size = 16 : i32}
    }
    aie.objectfifo.dma_endpoint @m_in(%mem) fills @m
    aie.objectfifo.dma_endpoint @m_out(%mem) drains @m
    %shim = aie.logical_tile<ShimNOCTile>(?, ?)
    aie.route_endpoint @out(%shim) DMA
    // CHECK: aie.route from [@a_dma, @b_dma] to [@m_in] {packet = #aie.packet_info<>}
    aie.route from [@a_dma, @b_dma] to [@m_in] {packet = #aie.packet_info<>}
    aie.route from @m_out to [@out]
    // CHECK-NOT: aie.logical_tile
  }
}
