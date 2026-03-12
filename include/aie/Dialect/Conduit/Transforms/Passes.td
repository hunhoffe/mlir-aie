//===- Passes.td - Conduit dialect passes ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Declares the two Conduit transformation passes:
//   Pass A: --objectfifo-to-conduit  (ObjectFIFO → Conduit IR)
//   Pass C: --conduit-to-dma         (Conduit IR → aie.dma_bd / aie.lock)
//
// Pass B (--air-channel-to-conduit) is deferred until Pass A + C are
// validated.
//
//===----------------------------------------------------------------------===//

#ifndef CONDUIT_PASSES
#define CONDUIT_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Pass A — ObjectFIFO → Conduit
//===----------------------------------------------------------------------===//

def ObjectFifoToConduit : Pass<"objectfifo-to-conduit", "::mlir::ModuleOp"> {
  let summary = "Lift aie.objectfifo.* ops into Conduit IR";
  let description = [{
    Converts the ObjectFIFO abstraction layer into Conduit IR:

      aie.objectfifo @name (prod, {cons}, depth)
        → conduit.create + conduit.annotate (producer_tile, consumer_tile_N)

      aie.objectfifo.link [@srcs] -> [@dsts] ([jo][do])
        → conduit.objectfifo_link {srcs, dsts, mode, offsets, memtile}

      aie.objectfifo.acquire @name (Role, count)
        → conduit.acquire {name, count}

      aie.objectfifo.release @name (Role, count)
        → conduit.release {name, count}

      aie.objectfifo.subview.access %sv[i]
        → conduit.subview_access {name, index}

    Tile coordinate annotations are carried forward as conduit.annotate ops
    so that Pass C (--conduit-to-dma) can reconstruct the hardware placement.

    This pass is a text-level structural transformation; it does not require
    the AIE ObjectFIFO stateful transform to have run first.
  }];

  let constructor = "xilinx::conduit::createObjectFifoToConduitPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
    "::mlir::memref::MemRefDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Pass C — Conduit → BD/DMA
//===----------------------------------------------------------------------===//

def ConduitToDMA : Pass<"conduit-to-dma", "::mlir::ModuleOp"> {
  let summary = "Lower Conduit IR to aie.dma_bd / aie.lock / aie.buffer hardware ops";
  let description = [{
    Converts Conduit IR directly to the raw AIE hardware programming model,
    replacing the existing aie-objectFifoStatefulTransform path:

      conduit.create {name, capacity} + tile annotations
        → aie.buffer (depth-many) + aie.lock (prod + cons) per tile

      conduit.objectfifo_link {srcs, dsts, mode, memtile, offsets}
        → MemTile DMA BD chain: aie.dma_start, aie.dma_bd, aie.next_bd,
          aie.use_lock inside aie.memtile_dma

      conduit.acquire {name, count}
        → aie.use_lock(%cons_lock, AcquireGreaterEqual, count)

      conduit.release {name, count}
        → aie.use_lock(%prod_lock, Release, count)

      conduit.subview_access {name, index}
        → buffer reference via index into buffersPerConduit[name]

      conduit.create (shim-tile producer)
        → aie.shim_dma_allocation + aie.flow

    Implements the same LockAnalysis + DMAChannelAnalysis + buffer-allocation
    logic as AIEObjectFifoStatefulTransform, operating on Conduit IR instead
    of ObjectFIFO IR.

    A correct implementation produces identical aie.dma_bd / aie.lock /
    aie.buffer / aie.flow counts as the existing stateful transform pass.
    Differences reveal either bugs or optimization opportunities.
  }];

  let constructor = "xilinx::conduit::createConduitToDMAPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

#endif // CONDUIT_PASSES
