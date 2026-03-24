// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s --check-prefix=PASSA
// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s --check-prefix=PASSC
//
// Regression test: Phase 4.5 shim rename scope.
//
// Bug: replaceAllSymbolUses renames @ch → @ch_shim_alloc for ALL
// FlatSymbolRefAttr occurrences, including conduit.distribute/join/forward
// srcs and dsts.  Pass A must revert these conduit op references back to
// @ch so that Pass C's conduitMap lookup (keyed on conduit.create sym_name)
// succeeds.
//
// Test setup:
//   - shim tile (row 0) is the producer of @link_in  → triggers Phase 4.5
//   - @link_in is the src of an objectfifo.link (1→2 distribute)
//   - After --objectfifo-to-conduit:
//       * conduit.distribute srcs must remain [@link_in]  (NOT @link_in_shim_alloc)
//       * aie.shim_dma_allocation @link_in_shim_alloc IS created (correct)
//   - After --objectfifo-to-conduit --conduit-to-dma:
//       * Pass C must not emit "src buffers not allocated" error
//       * No conduit.* ops survive
//
// Also covers join: shim tile is consumer of @join_out, which is the dst of
// a join.  Phase 4.5 must not rename @join_out in conduit.join dsts.

// PASSA-LABEL: module @phase45_shim_rename_scope
// PASSA:       aie.device(xcve2302) {

// Phase 3: distribute link emitted with original conduit.create names.
// conduit.distribute srcs must be the conduit.create symbol, NOT
// the Phase 4.5 shim_alloc symbol (@link_in_shim_alloc).
// PASSA:       conduit.create @link_in
// PASSA:       conduit.create @link_out_a
// PASSA:       conduit.create @link_out_b
// PASSA:       conduit.distribute
// PASSA-SAME:  dsts = [@link_out_a, @link_out_b]
// PASSA-SAME:  srcs = [@link_in]
// PASSA-NOT:   srcs = [@link_in_shim_alloc]

// Phase 3: join link emitted with original conduit.create names.
// conduit.join dsts must be the conduit.create symbol, NOT
// the Phase 4.5 shim_alloc symbol (@join_out_shim_alloc).
// PASSA:       conduit.create @join_src_a
// PASSA:       conduit.create @join_src_b
// PASSA:       conduit.create @join_out
// PASSA:       conduit.join
// PASSA-SAME:  dsts = [@join_out]
// PASSA-SAME:  srcs = [@join_src_a, @join_src_b]
// PASSA-NOT:   dsts = [@join_out_shim_alloc]

// Phase 4.5 must emit shim_dma_allocation ops with the _shim_alloc suffix.
// PASSA:       aie.shim_dma_allocation @link_in_shim_alloc
// PASSA:       aie.shim_dma_allocation @join_out_shim_alloc

// No objectfifo ops survive.
// PASSA-NOT:   aie.objectfifo
// PASSA-NOT:   aie.objectfifo.link

// Pass C must succeed: no conduit.* ops survive (all lowered to DMA/lock).
// PASSC-LABEL: module @phase45_shim_rename_scope
// PASSC-NOT:   conduit.create
// PASSC-NOT:   conduit.distribute
// PASSC-NOT:   conduit.join
// PASSC:       aie.shim_dma_allocation @link_in_shim_alloc
// PASSC:       aie.shim_dma_allocation @join_out_shim_alloc

module @phase45_shim_rename_scope {
  aie.device(xcve2302) {
    // Distribute: shim producer (row 0) → link_in → memtile → two compute tiles.
    %shim_prod = aie.tile(2, 0)   // shim tile: producer of link_in
    %memtile   = aie.tile(2, 1)   // MemTile: relay
    %comp_a    = aie.tile(2, 2)   // compute tile A: consumer of link_out_a
    %comp_b    = aie.tile(2, 3)   // compute tile B: consumer of link_out_b

    aie.objectfifo @link_in  (%shim_prod, {%memtile}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @link_out_a (%memtile, {%comp_a},  2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @link_out_b (%memtile, {%comp_b},  2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@link_in] -> [@link_out_a, @link_out_b] ([][0, 16])

    // Join: two compute tiles → memtile → shim consumer (row 0).
    %comp_c    = aie.tile(3, 2)   // compute tile C: producer of join_src_a
    %comp_d    = aie.tile(3, 3)   // compute tile D: producer of join_src_b
    %memtile2  = aie.tile(3, 1)   // MemTile: relay
    %shim_cons = aie.tile(3, 0)   // shim tile: consumer of join_out

    aie.objectfifo @join_src_a (%comp_c, {%memtile2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @join_src_b (%comp_d, {%memtile2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @join_out   (%memtile2, {%shim_cons}, 2 : i32) : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo.link [@join_src_a, @join_src_b] -> [@join_out] ([0, 16][])
  }
}
