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
//   "heavy_conduit" has depth=4, element_type=memref<2048xi32> (4 slots x 16KB = 64KB).
//   Actually: per-slot = numElements * sizeof(i32) = 2048*4 = 8192 bytes = 8KB.
//   4 slots x 8KB = 32KB — exactly fills the budget.
//
//   Candidate: "light_fifo" depth=1, memref<32xi32> per slot, on tile (0,2).
//   Expected with fix: light_fifo NOT promoted (tile full at 32KB).
//   Expected with bug: light_fifo MIGHT be promoted (tile counted at 2x8KB=16KB,
//                      leaving 16KB "free", erroneously allowing promotion).
//
// With the fix, "light_fifo" stays at depth=1, slot_elems=128 (32 i32 x 4 bytes = 128B).
// CHECK-DAG: conduit.create @light_fifo {{{.*}}depth = 1 : i64, {{.*}}
//
// "heavy_conduit" always stays at depth=4 (depth>1 conduits are never candidates).
// CHECK-DAG: conduit.create @heavy_conduit {{{.*}}depth = 4 : i64, {{.*}}
// expected-remark @+1 {{conduit-depth-promote: promoted 0 conduit(s)}}
module {
aie.device(npu1) {

%t02 = aie.tile(0, 2)
%t03 = aie.tile(0, 3)
%t04 = aie.tile(0, 4)

// A depth-4 conduit on tile (0,2) occupying the full 32KB budget:
// 4 slots x memref<2048xi32> = 4 x 8192 bytes = 32768 bytes = 32KB.
// slot_elems = 4 * 2048 * 4 = 32768.  Wait, slot_elems = numElements * sizeof
// = 2048 * 4 = 8192?  Actually slot_elems = 2048 * 32 / 8 = ... let me just
// use the original value.
conduit.create @heavy_conduit {                element_type = memref<2048xi32>,
                depth = 4 : i64}

// A depth-1 candidate on the same tile (0,2).
// With the fix, the pre-population correctly charges 4x8KB=32KB for heavy_conduit,
// leaving 0 bytes free → light_fifo must NOT be promoted.
// expected-remark @+1 {{conduit-depth-promote: skipping 'light_fifo' -- memory budget}}
conduit.create @light_fifo {                element_type = memref<32xi32>,
                depth = 1 : i64}

// Structural tile info: tile(0,2) consumes both conduits.
%core02 = aie.core(%t02) {
  %w1 = conduit.acquire {name = @heavy_conduit, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<2048xi32>>
  conduit.release %w1 {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<2048xi32>>
  %w2 = conduit.acquire {name = @light_fifo, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<32xi32>>
  conduit.release %w2 {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<32xi32>>
  aie.end
}
%core03 = aie.core(%t03) {
  %w = conduit.acquire {name = @heavy_conduit, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<2048xi32>>
  conduit.release %w {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<2048xi32>>
  aie.end
}
%core04 = aie.core(%t04) {
  %w = conduit.acquire {name = @light_fifo, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<32xi32>>
  conduit.release %w {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<32xi32>>
  aie.end
}

func.func @heavy_existing() {
  return
}

func.func @light_candidate(%result: memref<32xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %win = conduit.acquire {name = @light_fifo, count = 1 : i64,
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

} // aie.device
} // module
