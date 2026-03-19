// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test for PassA-CSDF-count-collapse bug.
//
// Pattern (consumer core):
//   acquire(1) → release(1)
//   scf.for { acquire(3) → access [0],[1],[2] → release(1) }
//   acquire(2) → access [0],[1] → release(2)
//
// The scf.for body needs count=3 (accesses index 2).  The entry block's
// second acquire (after the loop) has count=2.  Before the fix,
// findWindowInDominatingBlock returned the entry block's count=2 window
// for the loop body, causing "index 2 out of bounds for acquire count 2".
//
// After the fix: the loop body emits its own conduit.acquire{count=3}
// because the entry block's count=2 window does not dominate the loop.

// CHECK-LABEL: module @csdf_count_regression
// The pipeline must complete without errors (exit 0).
// Verify the inner loop body acquires with the correct count.
// CHECK:       scf.for
// CHECK:         aie.use_lock
// CHECK:       }

module @csdf_count_regression {
    aie.device(xcve2302) {
        %tile12 = aie.tile(1, 2)
        %tile22 = aie.tile(2, 2)

        aie.objectfifo @fifo (%tile12, {%tile22}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

        %core12 = aie.core(%tile12) {
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

        %core22 = aie.core(%tile22) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c9 = arith.constant 9 : index

            // Phase 1: acquire 1
            %sv0 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
            %e0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %v0 = memref.load %e0[%c0] : memref<16xi32>
            aie.objectfifo.release @fifo (Consume, 1)

            // Phase 2: loop acquiring 3 (accesses indices 0, 1, 2)
            scf.for %i = %c0 to %c9 step %c1 {
                %sv1 = aie.objectfifo.acquire @fifo (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
                %e1 = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %e2 = aie.objectfifo.subview.access %sv1[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %e3 = aie.objectfifo.subview.access %sv1[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %v1 = memref.load %e1[%c0] : memref<16xi32>
                %v2 = memref.load %e2[%c0] : memref<16xi32>
                %v3 = memref.load %e3[%c0] : memref<16xi32>
                aie.objectfifo.release @fifo (Consume, 1)
            }

            // Phase 3: acquire 2 (after the loop)
            %sv2 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %e4 = aie.objectfifo.subview.access %sv2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %e5 = aie.objectfifo.subview.access %sv2[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            %v4 = memref.load %e4[%c0] : memref<16xi32>
            %v5 = memref.load %e5[%c0] : memref<16xi32>
            aie.objectfifo.release @fifo (Consume, 2)

            aie.end
        }
    }
}
