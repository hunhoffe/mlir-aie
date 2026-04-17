// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test for non-uniform acquire count across blocks.
//
// Pattern (consumer core):
//   acquire(2)                                     // preamble: hold 2 elements
//   scf.for { acquire(3), access [0,1,2], release(1) }  // loop: sliding window
//   release(2)                                     // tail: release remaining
//
// Before the fix: findWindowInDominatingBlock returned the preamble's
// acquire(2) window for the loop body (it dominates and has the same fifo
// name).  This caused subview_access index 2 to be bound-checked against
// count=2, producing "index 2 out of bounds for acquire count 2".
//
// After the fix: the loop body emits its own conduit.acquire{count=3}
// because the preamble's count=2 is less than the requested count=3.
// The acquire carries a prior_count=2 annotation so Pass C computes the
// lock delta (3-2=1) for AcquireGreaterEqual, matching the oracle.

// CHECK-LABEL: module @nonuniform_acquire_count
// The pipeline must complete without errors (exit 0).
// Verify the consumer core has a use_lock in the preamble AND inside the loop.
// CHECK:       aie.core
// CHECK:         aie.use_lock
// CHECK:         scf.for
// CHECK:           aie.use_lock
// CHECK:         }

module @nonuniform_acquire_count {
    aie.device(xcve2302) {
        %tile02 = aie.tile(0, 2)
        %tile12 = aie.tile(1, 2)

        aie.objectfifo @fifo (%tile02, {%tile12}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

        // Producer: writes 9 elements, one per iteration.
        %core02 = aie.core(%tile02) {
            %v = arith.constant 42 : i32
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c9 = arith.constant 9 : index

            scf.for %i = %c0 to %c9 step %c1 {
                %sv = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                memref.store %v, %e[%c0] : memref<16xi32>
                aie.objectfifo.release @fifo (Produce, 1)
            }
            aie.end
        }

        // Consumer: preamble acquire(2) + loop acquire(3)/release(1) + tail release(2).
        // This is the ResNet bottleneck sliding-window pattern.
        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c6 = arith.constant 6 : index

            // Preamble: acquire 2 elements for initial window.
            %sv0 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %e0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %e1 = aie.objectfifo.subview.access %sv0[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %v0 = memref.load %e0[%c0] : memref<16xi32>
            %v1 = memref.load %e1[%c0] : memref<16xi32>

            // Main loop: sliding window — acquire(3) gets one more element,
            // release(1) slides the window by 1.
            scf.for %i = %c0 to %c6 step %c1 {
                %sv1 = aie.objectfifo.acquire @fifo (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
                %e2 = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %e3 = aie.objectfifo.subview.access %sv1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %e4 = aie.objectfifo.subview.access %sv1[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %v2 = memref.load %e2[%c0] : memref<16xi32>
                %v3 = memref.load %e3[%c0] : memref<16xi32>
                %v4 = memref.load %e4[%c0] : memref<16xi32>
                aie.objectfifo.release @fifo (Consume, 1)
            }

            // Tail: release remaining 2 elements.
            aie.objectfifo.release @fifo (Consume, 2)

            aie.end
        }
    }
}
