// RUN: aie-opt --objectfifo-to-conduit %s 2>&1 | FileCheck %s
//
// Task #112 — Pass A allocate-revert bug fix.
//
// Bug: Phase 4.5 of ObjectFifoToConduit renames @chan → @chan_shim_alloc via
// replaceAllSymbolUses to satisfy runtime_sequence symbol references for
// shim-consumer fifos.  All conduit.* op name attrs are then explicitly
// reverted back to @chan.  AIE::ObjectFifoAllocateOp was missing from the
// revert list — its objFifo_name attr stayed at @chan_shim_alloc.  Phase 4.6
// then looked up the conduit.create by name @chan_shim_alloc, found nothing
// (the conduit.create's sym_name is @chan), emitted
// "cannot find conduit.create for 'chan_shim_alloc'", and erased the
// allocate.  The user's MemTile delegate was silently lost.
//
// After fix: Phase 4.6 finds the conduit.create and lowers the allocate to a
// scatter{N=1} relay through the MemTile.
//
// Scenario: producer = compute tile(0,2), consumer = shim tile(0,0)
// (triggers Phase 4.5 rename), allocate delegates to MemTile tile(0,1)
// (triggers Phase 4.6 scatter relay).

// CHECK-LABEL: module @shim_consumer_with_allocate
//
// Source channel kept under original name (sym revert succeeded).
// CHECK: conduit.create @chan
//
// Relay channel emitted by Phase 4.6 — proves the allocate was found, not
// dropped on the floor.
// CHECK: conduit.create @chan_relay
//
// scatter{N=1} relay through the MemTile.
// CHECK: conduit.scatter
// CHECK-SAME: src = @chan
// CHECK-SAME: dsts = [@chan_relay]
// CHECK-SAME: memtile = "tile(0,1)"
//
// Phase 4.5 still emits the shim DMA allocation under the renamed symbol.
// CHECK: aie.shim_dma_allocation @chan_shim_alloc
//
// No leftover aie.objectfifo or aie.objectfifo.allocate ops.
// CHECK-NOT: aie.objectfifo
// CHECK-NOT: aie.objectfifo.allocate
//
// Regression guard: the bug-era warning must NOT appear.
// CHECK-NOT: cannot find conduit.create

module @shim_consumer_with_allocate {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan (%tile_0_2, {%tile_0_0}, 2 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @chan (%tile_0_1)

    aie.core(%tile_0_2) {
      %sub = aie.objectfifo.acquire @chan (Produce, 1)
          : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %sub[0]
          : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      aie.objectfifo.release @chan (Produce, 1)
      aie.end
    }
  }
}
