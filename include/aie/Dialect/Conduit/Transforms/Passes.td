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
// Declares the three Conduit transformation passes:
//   Pass A: --objectfifo-to-conduit  (ObjectFIFO → Conduit IR)
//   Pass B: --air-channel-to-conduit (AIR Channel → Conduit IR)
//   Pass C: --conduit-to-dma         (Conduit IR → aie.dma_bd / aie.lock)
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
// Pass B — AIR Channel → Conduit
//===----------------------------------------------------------------------===//

def AirChannelToConduit : Pass<"air-channel-to-conduit", "::mlir::ModuleOp"> {
  let summary = "Lift air.channel.put/get ops into Conduit IR (Tier 3 memref-DMA ops)";
  let description = [{
    Converts the AIR Channel abstraction layer into Conduit IR Tier 3 ops:

      air.channel @name [1, 1]
        → conduit.create {name, capacity=1, element_type=<inferred>,
                          depth=1, link_mode=<empty>}

      air.channel.put async [%deps] @chan(%buf[offsets][sizes][strides])
        → conduit.put_memref_async {name, num_elems, offsets, sizes, strides}
                : !conduit.async.token

      air.channel.get async [%deps] @chan(%buf[offsets][sizes][strides])
        → conduit.get_memref_async {name, num_elems, offsets, sizes, strides}
                : !conduit.async.token

      air.wait_all async [%deps]  →  conduit.wait_all_async %deps
      air.wait_all       [%deps]  →  conduit.wait_all %deps

    The AIR dialect operates on generic unregistered ops when parsed via
    --allow-unregistered-dialect.  This pass matches by op name string so it
    does not require the AIR dialect library to be linked into aie-opt.

    Known limitations:
    - Only scalar [1,1] channels are handled (Task #32 established that
      SPMD specialization always produces [1,1] before lowering).
    - Channel size/shape information is extracted from the first put/get
      memref operand type; if no put/get is found, capacity defaults to 1.
    - The air.async.token → !conduit.async.token mapping relies on SSA
      value threading; all users of the old token must be updated.
  }];

  let constructor = "xilinx::conduit::createAirChannelToConduitPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
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
