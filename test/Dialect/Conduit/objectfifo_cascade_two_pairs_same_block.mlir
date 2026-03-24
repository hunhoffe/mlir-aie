// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Regression test for: cascade Produce release handler picks last acquire
// in block instead of the acquire immediately preceding the release.
//
// Bug: the forward scan for the paired acquire did not stop at the release op,
// so for two acquire/release pairs in the same block the second release was
// incorrectly paired with the last acquire in the block (always the second one),
// while the first release had no acquire left to pair with.
//
// Fix: break the scan when &scan == releaseOp so only acquires before this
// specific release are considered.
//
// This test has two back-to-back cascade acquire/release pairs for the same
// fifo in the same producer core body. Each release must be paired with its
// own preceding acquire — not the last acquire in the block.
//
// Expected: two put_cascade ops emitted, one per pair.
// Before fix: the first release would find no matching acquire (wrong),
// and the second release would claim the last acquire in the whole block.

// CHECK-LABEL: module

// Producer core must have exactly two put_cascade ops (one per pair).
// CHECK:      aie.core
// CHECK:        conduit.put_cascade "cas_fifo"
// CHECK:        conduit.put_cascade "cas_fifo"
// CHECK-NOT:    conduit.acquire
// CHECK-NOT:    conduit.release
// CHECK-NOT:    aie.objectfifo.acquire

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @cas_fifo(%tile03, {%tile13}, 1 : i32) {via_cascade = true}
        : !aie.objectfifo<memref<1xvector<16xi32>>>

    // Producer core: two acquire/release pairs for the same cascade fifo
    // in the same basic block.  Before the fix, the first release would scan
    // past itself and pick the second acquire (wrong pairing).
    aie.core(%tile03) {
      // --- Pair 1 ---
      %subview0 = aie.objectfifo.acquire @cas_fifo(Produce, 1)
          : !aie.objectfifosubview<memref<1xvector<16xi32>>>
      %elem0 = aie.objectfifo.subview.access %subview0[0]
          : !aie.objectfifosubview<memref<1xvector<16xi32>>> -> memref<1xvector<16xi32>>
      %c0 = arith.constant 0 : index
      %v0 = arith.constant dense<1> : vector<16xi32>
      memref.store %v0, %elem0[%c0] : memref<1xvector<16xi32>>
      aie.objectfifo.release @cas_fifo(Produce, 1)  // must pair with subview0

      // --- Pair 2 ---
      %subview1 = aie.objectfifo.acquire @cas_fifo(Produce, 1)
          : !aie.objectfifosubview<memref<1xvector<16xi32>>>
      %elem1 = aie.objectfifo.subview.access %subview1[0]
          : !aie.objectfifosubview<memref<1xvector<16xi32>>> -> memref<1xvector<16xi32>>
      %v1 = arith.constant dense<2> : vector<16xi32>
      memref.store %v1, %elem1[%c0] : memref<1xvector<16xi32>>
      aie.objectfifo.release @cas_fifo(Produce, 1)  // must pair with subview1

      aie.end
    }

    aie.core(%tile13) {
      %subview = aie.objectfifo.acquire @cas_fifo(Consume, 1)
          : !aie.objectfifosubview<memref<1xvector<16xi32>>>
      %elem = aie.objectfifo.subview.access %subview[0]
          : !aie.objectfifosubview<memref<1xvector<16xi32>>> -> memref<1xvector<16xi32>>
      %c0 = arith.constant 0 : index
      %r = memref.load %elem[%c0] : memref<1xvector<16xi32>>
      vector.print %r : vector<16xi32>
      aie.objectfifo.release @cas_fifo(Consume, 1)
      aie.end
    }
  }
}
