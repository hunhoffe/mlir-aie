// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Foundation Phase 2 (Task #19), gap #10 — explicit Pattern B test:
// asymmetric loop nesting on producer vs consumer with EQUAL total
// acquire counts.
//
// Geometry (matching the IRON ChanneledUnary / RoPE shape):
//   * Producer core: scf.for(0,8) { scf.for(0,4) { acquire } } →
//     8 * 4 = 32 acquires.
//   * Consumer core: scf.for(0,32) { acquire }  →  32 acquires.
//   * Single channel; compute-to-compute (no shim BD), so emissions=1
//     and acquires_per_BD=1.
//
// Pass A's `productOfEnclosingLoops` walks ALL enclosing loops above
// the first acquire on each side, so:
//   * producer side trip = 8 * 4 = 32
//   * consumer side trip = 32
// → trips agree → dma_repeat = 32 / 1 / 1 = 32.
//
// This pins the contract that Pass A computes "product of all enclosing
// loops at the first acquire" rather than the naive shortcut "outermost
// trip count only".  Without the product behavior, producer side would
// be misread as 8 (outer only), mismatching consumer 32, and the
// inference would be skipped with a remark.
//
// Replaces the weak proxy in `infer_iter_count_per_channel.mlir`
// (which used symmetric 2-level nests on both sides).

// CHECK-LABEL: module @pattern_b_per_channel_mixed_depth
// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 32

module @pattern_b_per_channel_mixed_depth {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @chan(%tile_0_2, {%tile_0_3}, 2 : i32)
        : !aie.objectfifo<memref<8xbf16>>

    func.func private @produce(memref<8xbf16>)
    func.func private @consume(memref<8xbf16>)

    // Producer: 2-level nest, total = 8 * 4 = 32 acquires.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %w = aie.objectfifo.acquire @chan(Produce, 1)
              : !aie.objectfifosubview<memref<8xbf16>>
          %buf = aie.objectfifo.subview.access %w[0]
              : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
          func.call @produce(%buf) : (memref<8xbf16>) -> ()
          aie.objectfifo.release @chan(Produce, 1)
        }
      }
      aie.end
    }

    // Consumer: single flattened loop, 32 acquires.
    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      scf.for %i = %c0 to %c32 step %c1 {
        %w = aie.objectfifo.acquire @chan(Consume, 1)
            : !aie.objectfifosubview<memref<8xbf16>>
        %buf = aie.objectfifo.subview.access %w[0]
            : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
        func.call @consume(%buf) : (memref<8xbf16>) -> ()
        aie.objectfifo.release @chan(Consume, 1)
      }
      aie.end
    }
  }
}
