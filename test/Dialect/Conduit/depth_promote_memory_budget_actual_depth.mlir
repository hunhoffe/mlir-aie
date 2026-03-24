// RUN: aie-opt --conduit-depth-promote %s | FileCheck %s
//
// Regression test for: depth-promote memory budget pre-population used
// hardcoded newDepth=2 instead of the conduit's actual depth.
//
// Bug: ConduitDepthPromotion.cpp pre-populated tileMemUsed with
//   perSlotBytes * 2
// for every existing conduit regardless of its actual depth. A depth-4
// conduit was counted as depth-2, underestimating memory by 2x and allowing
// a subsequent depth-1 conduit to be promoted when the tile was already full.
//
// Fix: use `depth` (the actual conduit depth) instead of the hardcoded 2.
//
// Test setup:
//   Tile (0,2) has a 32KB memory budget (AIE2 compute tile).
//   "big_conduit" has depth=4, element_type=memref<4096xi32> (4 slots × 16KB = 64KB).
//   With the bug: pre-population counts 2 slots × 16KB = 32KB → budget appears full.
//     But since the candidate "small_fifo" needs 2×32B=64B and the pre-population
//     leaves 0KB free — it should be rejected.
//   With the actual bug (newDepth=2): pre-population adds 2×16KB=32KB for big_conduit,
//     which exactly fills the 32KB budget, correctly blocking promotion of small_fifo.
//     But if big_conduit had depth=8 (64×16KB=128KB total), newDepth=2 would add
//     only 32KB, leaving 0KB "free" and still blocking small_fifo. The real failure
//     mode is when the actual depth is LARGER than 2: the budget appears to have
//     MORE free space than reality, allowing invalid promotions.
//
// This test uses a scenario that clearly distinguishes old vs new behavior:
//   Tile (0,2): existing "heavy_conduit" depth=4, memref<2048xi32> per slot.
//               4 slots × 8KB = 32KB — exactly fills a 32KB tile.
//   Candidate: "light_fifo" depth=1, memref<32xi32> per slot, on tile (0,2).
//   Expected with fix: light_fifo NOT promoted (tile full at 32KB).
//   Expected with bug: light_fifo MIGHT be promoted (tile counted at 2×8KB=16KB,
//                      leaving 16KB "free", erroneously allowing promotion).
//
// With the fix, "light_fifo" stays at depth=1, capacity=128 (32 i32 × 4 bytes = 128B).
// CHECK-DAG: conduit.create {capacity = 128 : i64, {{.*}} depth = 1 : i64, {{.*}} name = "light_fifo"
//
// "heavy_conduit" always stays at depth=4 (depth>1 conduits are never candidates).
// CHECK-DAG: conduit.create {capacity = 65536 : i64, {{.*}} depth = 4 : i64, {{.*}} name = "heavy_conduit"

// expected-remark @+1 {{conduit-depth-promote: promoted 0 conduit(s)}}
module {

// A depth-4 conduit on tile (0,2) occupying the full 32KB budget:
// 4 slots × memref<2048xi32> = 4 × 8192 bytes = 32768 bytes = 32KB.
// capacity = 4 * 2048 * 4 = 32768.
func.func @heavy_existing() {
  conduit.create {name = "heavy_conduit",
                  capacity = 65536 : i64,
                  producer_tile = array<i64: 0, 0>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<2048xi32>,
                  depth = 4 : i64}
  return
}

// A depth-1 candidate on the same tile (0,2).
// With the fix, the pre-population correctly charges 4×8KB=32KB for heavy_conduit,
// leaving 0 bytes free → light_fifo must NOT be promoted.
// With the bug (newDepth=2), only 2×8KB=16KB is charged → 16KB appears free →
// light_fifo would be incorrectly promoted to depth=2.
func.func @light_candidate(%result: memref<32xi32>) {
  // expected-remark @+1 {{conduit-depth-promote: skipping 'light_fifo' -- memory budget}}
  conduit.create {name = "light_fifo",
                  capacity = 128 : i64,
                  producer_tile = array<i64: 0, 0>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<32xi32>,
                  depth = 1 : i64}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %win = conduit.acquire {name = "light_fifo", count = 1 : i64,
                            port = #conduit.port<Consume>}
               : !conduit.window<memref<32xi32>>
    %elem = conduit.subview_access %win {index = 0 : i64}
               : !conduit.window<memref<32xi32>> -> memref<32xi32>
    memref.copy %elem, %result : memref<32xi32> to memref<32xi32>
    conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<32xi32>>
  }
  return
}

} // module
