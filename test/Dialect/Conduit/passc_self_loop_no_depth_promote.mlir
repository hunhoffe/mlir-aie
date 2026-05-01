//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --conduit-depth-promote %s | FileCheck %s

// Layer A regression pin (companion to passc_self_loop_no_dma_bd_chain.mlir
// / Layer B): `--conduit-depth-promote` MUST NOT promote a same-tile depth=1
// self-loop conduit (`aie.objectfifo @inter (%tile, {%tile}, 1)`) to depth=2,
// because the conduit lowers to pure shared-memory access — the core reads/
// writes its own tile-local buffer directly via the buffer + locks allocated
// in `ConduitToDMAAlloc.cpp`'s `sameTile` carve-out (line 128).  No DMA exists;
// rotation has no meaning.  Promoting would emit a useless rotation counter +
// 2 buffers + cf.switch infrastructure that Pass C would then need to
// special-case again.
//
// Mirrors `--aie-objectFifo-stateful-transform`'s emit shape on the same
// input (1 buffer per conduit, lock init=1, no rotation counter; captured
// 2026-05-01 per CLAUDE.md USER-LOCKED 2026-04-30 capture-stateful rule).
//
// Sibling Layer B fixture: `passc_self_loop_no_dma_bd_chain.mlir`.  Layer B
// suppresses the spurious BD-chain emit downstream in Pass C
// (`ConduitToDMALink.cpp:1543` predicate, commit `e3a8c0b139`).  Layer A
// (this fixture) keeps the IR clean BEFORE Pass C even sees it, so the depth-
// promote pass doesn't insert rotation infrastructure for self-loops in the
// first place.
//
// Predicate (in `ConduitDepthPromotion.cpp` Step 5 candidate loop, after
// Criterion 0 cascade check, before Criterion 1):
//   producerTile == consumerTiles[0]
//   AND consumerTiles.size() == 1
//   AND shimConsumerTiles.empty()
// (depth==1 implicit — the candidate set at Step 2 is depth==1 only; the
// matching Pass C predicate at `ConduitToDMALink.cpp:1543` spells it out
// explicitly.)
//
// Pre-fix `--conduit-depth-promote` would mutate `@inter` from depth=1 to
// depth=2, allocating a second buffer + rotation counter on tile(0,2).
// Post-fix: `@inter` retains depth=1 attribute (and Pass C's H4 fix then
// suppresses the BD chain emit downstream).

module {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // External shim->tile depth=2 (not a candidate; depth>1 conduits skip
    // Step 2 entirely).
    aie.objectfifo @ext_in (%shim, {%tile}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Same-tile depth=1 self-loop — the Layer A pin target.  Must STAY at
    // depth=1 after `--conduit-depth-promote`.
    aie.objectfifo @inter (%tile, {%tile}, 1 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // External tile->shim depth=2 (not a candidate).
    aie.objectfifo @ext_out (%tile, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Non-passthrough loop body — both Produce and Consume of @inter with
    // compute in between.  Required so @inter survives Criterion 4
    // (passthrough-only check) — without compute, the candidate is skipped
    // for an UNRELATED reason and we wouldn't be testing the new predicate.
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

// @inter MUST stay at depth = 1 (Layer A fix — same-tile self-loop is pure
// shared-memory; promotion has no benefit and pollutes the IR).  A
// pre-fix run would emit `depth = 2 : i64` here.  The depth attribute is
// emitted inline on the conduit.create op and must match exactly.
// CHECK-DAG: conduit.create @inter {{.*}}depth = 1 : i64

// @ext_in / @ext_out are depth=2 already (not candidates for promotion;
// depth>1 conduits skip Step 2 entirely).  Defensive checks: their depth
// attribute must remain 2 — confirms the new predicate doesn't accidentally
// touch external conduits.
// CHECK-DAG: conduit.create @ext_in {{.*}}depth = 2 : i64
// CHECK-DAG: conduit.create @ext_out {{.*}}depth = 2 : i64

// Defensive: the depth-promote pass emits a "promoted N conduit(s)" remark
// on the module op only when N > 0.  With the Layer A fix and no other
// candidates in this fixture, no promotion remark should appear.
// CHECK-NOT: conduit-depth-promote: promoted
