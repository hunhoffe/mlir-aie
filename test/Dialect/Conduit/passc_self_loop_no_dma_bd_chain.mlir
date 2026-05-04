//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s

// Pass C regression pin (H4 from Task #25 / #34): a same-tile depth=1
// objectfifo self-loop (`aie.objectfifo @inter (%tile, {%tile}, 1)`) MUST be
// lowered to pure shared memory (just `aie.buffer` + locks) — NO producer-side
// MM2S BD chain, NO consumer-side S2MM BD chain, NO `aie.flow` on the
// self-loop.  This matches `--aie-objectFifo-stateful-transform`'s reference
// emit on the same shape (captured 2026-05-01 per CLAUDE.md USER-LOCKED
// 2026-04-30 capture-stateful rule).
//
// Pre-fix Pass C (`ConduitToDMALink.cpp` Phase 5.5 generic loop) emitted
// FOUR `aie.dma_start` ops on tile(0,2):
//   - S2MM 0  ← @ext_in   (correct)
//   - MM2S 1  ← @inter producer (SPURIOUS, Case C path)
//   - S2MM 0  ← @inter consumer (SPURIOUS + S2MM-0 collision with @ext_in)
//   - MM2S 0  ← @ext_out  (correct)
// The S2MM-0 collision corrupted @ext_in's hardware programming → core
// blocked on @ext_in lock → XRT timeout (ABORT status 8) — observed in
// `test/npu-xrt/fuse_channels_npu/aie.mlir` failure.
//
// Post-fix: only TWO `aie.dma_start` (one S2MM, one MM2S).  `@inter`'s
// buffer + 2 locks remain (allocated by `ConduitToDMAAlloc.cpp` regardless,
// per its `sameTile` carve-out at line 128) — only the BD chain emit is
// suppressed.
//
// Predicate (in `ConduitToDMALink.cpp` Phase 5.5 generic loop, applied
// before link-source / sharedMemory / Case C / Case A handling):
//   sameTile = (producerTileCoord == consumerTileCoords[0])
//   AND consumerTileCoords.size() == 1
//   AND shimConsumerTileCoords.empty()
//   AND depth == 1
// Restricted to depth==1 to preserve the depth>1 self-loop rotation path
// fixed by #92 (rotation counters need real BD chains).

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    aie.objectfifo @ext_in (%shim, {%tile}, 1 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Same-tile depth=1 self-loop — the pin target.  Must lower to pure
    // shared memory: buffer + locks only, no DMA.
    aie.objectfifo @inter (%tile, {%tile}, 1 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.objectfifo @ext_out (%tile, {%shim}, 1 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      %cone = arith.constant 1.0 : bf16
      %ctwo = arith.constant 2.0 : bf16
      scf.for %niter = %c0 to %c4 step %c1 {
        // Add section: read @ext_in, produce @inter.
        %ina = aie.objectfifo.acquire @ext_in(Consume, 1) : !aie.objectfifosubview<memref<64xbf16>>
        %inb = aie.objectfifo.acquire @inter(Produce, 1) : !aie.objectfifosubview<memref<64xbf16>>
        %r_ina = aie.objectfifo.subview.access %ina[0] : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        %r_inb = aie.objectfifo.subview.access %inb[0] : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        scf.for %i = %c0 to %c64 step %c1 {
          %v = memref.load %r_ina[%i] : memref<64xbf16>
          %s = arith.addf %v, %cone : bf16
          memref.store %s, %r_inb[%i] : memref<64xbf16>
        }
        aie.objectfifo.release @ext_in(Consume, 1)
        aie.objectfifo.release @inter(Produce, 1)

        // Mul section: consume @inter, produce @ext_out.
        %m_in = aie.objectfifo.acquire @inter(Consume, 1) : !aie.objectfifosubview<memref<64xbf16>>
        %m_out = aie.objectfifo.acquire @ext_out(Produce, 1) : !aie.objectfifosubview<memref<64xbf16>>
        %r_min = aie.objectfifo.subview.access %m_in[0] : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        %r_mout = aie.objectfifo.subview.access %m_out[0] : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %r_min[%j] : memref<64xbf16>
          %p = arith.mulf %v, %ctwo : bf16
          memref.store %p, %r_mout[%j] : memref<64xbf16>
        }
        aie.objectfifo.release @inter(Consume, 1)
        aie.objectfifo.release @ext_out(Produce, 1)
      }
      aie.end
    }
  }
}

// CHECK-LABEL: aie.device(npu2)

// Capture the compute tile SSA value.
// CHECK: %[[TILE:.+]] = aie.tile(0, 2)

// @inter's buffer + locks are still allocated by Pass C (ConduitToDMAAlloc
// `sameTile` carve-out at line 128 leaves the consumer-side resources in
// place).  Only the spurious BD chain emit is what the H4 fix removes.
// CHECK-DAG: aie.buffer(%[[TILE]]) {sym_name = "inter_cons_buff_0"}
// CHECK-DAG: aie.lock(%[[TILE]], {{.*}}) {init = 1 : i32, sym_name = "inter_cons_prod_lock_0"}
// CHECK-DAG: aie.lock(%[[TILE]], {{.*}}) {init = 0 : i32, sym_name = "inter_cons_cons_lock_0"}

// Exactly two `aie.flow` ops survive — @ext_in (shim->tile) and @ext_out
// (tile->shim).  No self-loop flow.
// CHECK-DAG: aie.flow(%{{.+}}, DMA : 0, %[[TILE]], DMA : 0)
// CHECK-DAG: aie.flow(%[[TILE]], DMA : 0, %{{.+}}, DMA : 0)

// Defensive: forbid any self-loop flow on tile(0,2).  A stray
// `aie.flow(%tile, ..., %tile, ...)` would mean Phase 5 routePhase
// emitted a flow for the same-tile self-loop conduit (the bug shape this
// fix prevents).  Bound to end-of-file by the trailing CHECK-NOT after
// the dma_start COUNT-2 below.
// CHECK-NOT: aie.flow(%[[TILE]],{{.*}}, %[[TILE]],

// Tile(0,2)'s aie.mem must contain EXACTLY two dma_start ops total — one
// S2MM (for @ext_in's shim->tile flow) and one MM2S (for @ext_out's tile->
// shim flow).  Pre-fix this contained four dma_starts (extra MM2S 1 + S2MM 0
// for @inter producer/consumer sides).  The count directive plus a trailing
// negative directive enforces exhaustiveness — plain sequential CHECKs alone
// would skip over the spurious dma_starts and falsely pass.
// CHECK: aie.mem(%[[TILE]])
// CHECK-COUNT-2: aie.dma_start(
// CHECK-NOT: aie.dma_start
