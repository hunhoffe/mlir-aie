// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test: conduit.link with a CoreTile relay (row >= 2) must lower
// without error and produce a BD-chain + two flows (commit 2455feb).
//
// Before the fix, Pass C hard-rejected conduit.link where the relay tile is
// a compute tile (not a MemTile), causing 4 positive corpus tests to fail.
// The fix adds a CoreTile relay path in linkPhase() that:
//   1. Creates S2MM + MM2S DMA chains on the relay tile's aie.mem block.
//   2. Emits aie.flow from relay tile MM2S → consumer tile S2MM.
//
// Topology: shim(0,0) → relay(0,2) [CoreTile] → dst(0,4)
//   shim sends via MM2S; relay receives on S2MM and re-sends via MM2S;
//   dst receives on S2MM.
//
// Expected output structure:
//   aie.flow(shim → relay)   [shim MM2S → relay S2MM]
//   aie.mem(relay) with S2MM and MM2S DMA chains
//   aie.flow(relay → dst)    [relay MM2S → dst S2MM]
//   aie.mem(dst)  with S2MM chain

// CHECK-LABEL: module @conduit_core_tile_relay

// --- Flow from shim to relay (CoreTile) ---
// CHECK: aie.flow(%shim_noc_tile_0_0, DMA : 0, %tile_0_2, DMA : 0)

// --- Relay tile gets both S2MM and MM2S DMA chains ---
// CHECK: aie.mem(%tile_0_2)
// CHECK:   aie.dma_start(S2MM, 0
// CHECK:   aie.dma_bd(%in_fifo_cons_buff_0
// CHECK:   aie.dma_start(MM2S, 0
// CHECK:   aie.dma_bd(%in_fifo_cons_buff_0

// --- Flow from relay to destination ---
// CHECK: aie.flow(%tile_0_2, DMA : 0, %tile_0_4, DMA : 0)

// --- Destination tile gets S2MM DMA chain ---
// CHECK: aie.mem(%tile_0_4)
// CHECK:   aie.dma_start(S2MM, 0

// --- No residual conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.link

module @conduit_core_tile_relay {
  aie.device(npu1_1col) {
    %shim  = aie.tile(0, 0)   // Shim source
    %relay = aie.tile(0, 2)   // CoreTile relay (row=2, not a MemTile)
    %dst   = aie.tile(0, 4)   // Final consumer

    aie.objectfifo @in_fifo  (%shim,  {%relay}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_fifo (%relay, {%dst},   2 : i32) : !aie.objectfifo<memref<16xi32>>

    // Link: relay tile acts as a transparent forwarder (CoreTile, not MemTile).
    // This pattern used to fail before commit 2455feb; now lowers correctly.
    aie.objectfifo.link [@in_fifo] -> [@out_fifo] ([] [])

    %core_dst = aie.core(%dst) {
      %sv = aie.objectfifo.acquire @out_fifo (Consume, 1)
          : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @out_fifo (Consume, 1)
      aie.end
    }
  }
}
