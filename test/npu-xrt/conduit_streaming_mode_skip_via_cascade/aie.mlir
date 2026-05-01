//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// HW smoke for the streaming-mode skip in
//   ObjectFifoToConduit.cpp::inferDmaRepeatForChannel
// (lit pin: test/Dialect/Conduit/infer_iter_count_skip_for_stream.mlir).
//
// Pass A (--objectfifo-to-conduit) MUST skip stamping `dma_repeat` on the
// produced `conduit.create` when the source `aie.objectfifo` is routed via
// Stream / Cascade — those routing modes emit no shim DMA BD chain, so a
// stamped `dma_repeat` would be wrong-by-construction.  This file exercises
// the `via_cascade = true` arm of that skip on real HW: producer at
// tile(0,3) → consumer at tile(1,3) over the cascade port (compatible
// cascade-adjacency mirrors test/npu-xrt/cascade_flows/aie.mlir geometry).
//
// Pass A behaviour we rely on (per
//   test/Dialect/Conduit/objectfifo_to_conduit_cascade.mlir):
//   - aie.objectfifo @cas {via_cascade = true} lowers to
//     conduit.create @cas {routing_mode = #conduit.routing_mode<cascade>}
//     WITHOUT a `dma_repeat` attribute (the streaming-mode skip).
//   - acquire(@cas Produce) + subview.access + memref.store + release  →
//     aie.put_cascade(<vector value>)
//   - acquire(@cas Consume) + subview.access + memref.load  + release  →
//     aie.get_cascade() : <vector type>
//
// Downstream (Pass C / aie-standard-lowering / peano):
//   - cascade-routed conduit.create lowers to aie.cascade_flow.
//   - aie.put_cascade / aie.get_cascade lower to llvm.aie2.mcd.write.vec /
//     llvm.aie2.scd.read.vec (per test/lower-to-standard/lower_cascade_put_get.mlir).
//
// Why HW smoke is interesting even though the change is silent at runtime:
//   Pass C today carries skip-branches that ignore stamped-but-irrelevant
//   `dma_repeat` on cascade/stream channels (per CLAUDE.md "Mechanism" note
//   on the smoke).  A future tightening of Pass C that ASSUMES `dma_repeat`
//   is present (or absent) on cascade-routed conduit.create would silently
//   regress the skip without any lit signal.  Driving the entire pipeline
//   end-to-end on HW gates against that — if Pass A re-stamps `dma_repeat`,
//   Pass C may produce a malformed cascade lowering, downstream peano /
//   firmware will refuse to dispatch (or dispatch + give wrong bytes),
//   and this test FAILs.
//
// Topology (single dispatch, single 16-i32 vector through one cascade hop):
//
//   shim t(0,0) ──@in0──▶ memtile t(0,1) ──@in1──▶ producer t(0,3)
//                                                          │
//                                          @cas {via_cascade = true}
//                                                          │
//                                                          ▼
//                                                  consumer t(1,3)
//                                                          │
//   shim t(0,0) ◀──@out0── memtile t(0,1) ◀──@out1─────────┘
//
// Cascade direction tile(0,3) → tile(1,3) mirrors the cascade adjacency
// already proven on HW by test/npu-xrt/cascade_flows/aie.mlir
// (`aie.cascade_flow(%t03, %t13)`), so we are not inventing new cascade
// routing — only swapping the source declaration from `aie.cascade_flow`
// (manual) to `aie.objectfifo {via_cascade=true}` (Pass-A-driven).
//
// Element type for the cascade objectfifo is `memref<1xvector<16xi32>>`
// (mirrors the lit pin and the AIE2 cascade stream width = 512 bits =
// vector<16xi32>); the input / output objectfifos use the natural flat
// `memref<16xi32>` so DMA byte counts are obvious to the runtime sequence.

module {
  aie.device(NPUDEVICE) {
    %shim_0_0    = aie.tile(0, 0)
    %memtile_0_1 = aie.tile(0, 1)
    %prod_0_3    = aie.tile(0, 3)
    %cons_1_3    = aie.tile(1, 3)

    // ---- Input: shim → memtile → producer (DMA via memtile bridge) ----
    aie.objectfifo @in0 (%shim_0_0,    {%memtile_0_1}, 1 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in1 (%memtile_0_1, {%prod_0_3},    1 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@in0] -> [@in1] ([] [])

    // ---- Cascade: producer → consumer (THE THING TESTED) ----
    aie.objectfifo @cas (%prod_0_3, {%cons_1_3}, 1 : i32)
        {via_cascade = true}
        : !aie.objectfifo<memref<1xvector<16xi32>>>

    // ---- Output: consumer → memtile → shim (DMA via memtile bridge) ----
    aie.objectfifo @out1 (%cons_1_3,    {%memtile_0_1}, 1 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out0 (%memtile_0_1, {%shim_0_0},    1 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@out1] -> [@out0] ([] [])

    // Producer core: read 16 i32 from input objfifo, transfer-read into
    // a vector<16xi32>, store the vector into the cascade element memref.
    // Pass A pattern-matches  acquire(@cas Produce) + subview.access +
    // memref.store(%v) + release  →  aie.put_cascade(%v).
    aie.core(%prod_0_3) {
      %c0    = arith.constant 0 : index
      %i32_0 = arith.constant 0 : i32

      %sv_in   = aie.objectfifo.acquire @in1 (Consume, 1)
          : !aie.objectfifosubview<memref<16xi32>>
      %elem_in = aie.objectfifo.subview.access %sv_in[0]
          : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %v = vector.transfer_read %elem_in[%c0], %i32_0 {in_bounds = [true]}
          : memref<16xi32>, vector<16xi32>

      %sv_cas   = aie.objectfifo.acquire @cas (Produce, 1)
          : !aie.objectfifosubview<memref<1xvector<16xi32>>>
      %elem_cas = aie.objectfifo.subview.access %sv_cas[0]
          : !aie.objectfifosubview<memref<1xvector<16xi32>>> -> memref<1xvector<16xi32>>
      memref.store %v, %elem_cas[%c0] : memref<1xvector<16xi32>>

      aie.objectfifo.release @cas (Produce, 1)
      aie.objectfifo.release @in1 (Consume, 1)
      aie.end
    }

    // Consumer core: load the vector<16xi32> from the cascade element
    // memref, transfer-write to output objfifo.
    // Pass A pattern-matches  acquire(@cas Consume) + subview.access +
    // memref.load + release  →  aie.get_cascade() : vector<16xi32>.
    aie.core(%cons_1_3) {
      %c0 = arith.constant 0 : index

      %sv_cas   = aie.objectfifo.acquire @cas (Consume, 1)
          : !aie.objectfifosubview<memref<1xvector<16xi32>>>
      %elem_cas = aie.objectfifo.subview.access %sv_cas[0]
          : !aie.objectfifosubview<memref<1xvector<16xi32>>> -> memref<1xvector<16xi32>>
      %v = memref.load %elem_cas[%c0] : memref<1xvector<16xi32>>
      aie.objectfifo.release @cas (Consume, 1)

      %sv_out   = aie.objectfifo.acquire @out1 (Produce, 1)
          : !aie.objectfifosubview<memref<16xi32>>
      %elem_out = aie.objectfifo.subview.access %sv_out[0]
          : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      vector.transfer_write %v, %elem_out[%c0] {in_bounds = [true]}
          : vector<16xi32>, memref<16xi32>
      aie.objectfifo.release @out1 (Produce, 1)
      aie.end
    }

    // Single dispatch: one 16-i32 (= 64 byte = 1 cascade vector) transfer.
    // Convention mirrors test/npu-xrt/add_one_objFifo/aie.mlir runtime
    // sequence (aiex.npu.dma_memcpy_nd + dma_wait, single shot).
    aie.runtime_sequence(%in : memref<16xi32>, %out : memref<16xi32>) {
      %c0  = arith.constant  0 : i64
      %c1  = arith.constant  1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd
          (%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1])
          { metadata = @out0, id = 1 : i64 } : memref<16xi32>
      aiex.npu.dma_memcpy_nd
          (%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1])
          { metadata = @in0, id = 0 : i64, issue_token = true } : memref<16xi32>
      aiex.npu.dma_wait { symbol = @out0 }
    }
  }
}
