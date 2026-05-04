// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-canonicalize-channel-puts --conduit-depth-promote --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-canonicalize-channel-puts --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s
//
// Pass A used to drop the MemTile relay aie.tile when its only SSA users
// were the aie.objectfifo ops being lowered — Pass C then crashed with
// "relay tile 'tile(0,1)' not found".  Sprint N+1 HIGH, surfaced
// 2026-04-29 by post-Sprint-N Llama compile-only verify (Task #22).  4
// instances in Llama: op5 StridedCopy, op6 Repeat, op10 Transpose ×2.
// Captured reproducer: tests/captured_reproducers/path_c_relay_tile_lookup/.
//
// Fix (F1b, landed in this commit): switch conduit.scatter / conduit.gather
// / conduit.transpose's `memtile` from a string attribute to an
// SSA-Index-typed Value operand whose defining op is an aie.tile.  Pass A
// emits the relay tile via AIE::TileOp::getOrCreate(builder, device, c, r),
// which both materializes the aie.tile (if absent) AND establishes an SSA
// use-chain from the lifted conduit.scatter to the aie.tile.  DCE can no
// longer drop the tile because it now has a real user.
//
// Fixture geometry (smallest IR that triggered the bug):
//   * 1 device (npu1_1col) — multi-device not required by the bug.
//   * shim(0,0) → memtile(0,1) → compute(0,2)
//   * 1 producer objectfifo @in_fifo  (shim → memtile)
//   * 1 consumer objectfifo @out_fifo (memtile → compute)
//   * 1 aie.objectfifo.link [@in_fifo] -> [@out_fifo] — becomes
//     conduit.scatter with memtile = %mem_tile_0_1 (SSA) after Pass A.
//   * MemTile (0,1) had no other SSA users (no aie.buffer, no aie.lock,
//     no aie.dma) — only the two objectfifos.  Adding any other op that
//     consumed %memtile would have masked the bug.
//   * One aie.core on the consumer to keep @out_fifo from being dropped.
//
// CHECK after fix: Pass C linkPhase finds the tile via SSA edge and
// lowers normally, producing an aie.memtile_dma chain on the relay.

// CHECK: aie.device
// CHECK: %[[MT:.*]] = aie.tile(0, 1)
// CHECK: aie.memtile_dma(%[[MT]])

module @path_c_relay_tile_dropped_by_pass_a {
  aie.device(npu1_1col) {
    %shim    = aie.tile(0, 0)
    %memtile = aie.tile(0, 1)
    %compute = aie.tile(0, 2)

    aie.objectfifo @in_fifo  (%shim,    {%memtile}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_fifo (%memtile, {%compute}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // Link op becomes conduit.scatter referencing %memtile via SSA after
    // Pass A — F1b refactor ensures the aie.tile is preserved.
    aie.objectfifo.link [@in_fifo] -> [@out_fifo] ([] [])

    %core_compute = aie.core(%compute) {
      %sv = aie.objectfifo.acquire @out_fifo (Consume, 1)
          : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @out_fifo (Consume, 1)
      aie.end
    }
  }
}
