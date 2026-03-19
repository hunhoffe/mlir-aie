// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test for PassA-SSA-dominance bug.
//
// Pattern: acquire → release → scf.for { acquire → release } → acquire → release
//
// The outer block has two separate acquire/release groups separated by a
// release.  The second acquire comes AFTER the inner scf.for in the same
// block.  findWindowInDominatingBlock must NOT return the second acquire's
// window for the inner block, because it does not dominate the inner block.
//
// Before the fix: the inner block reused the outer block's second acquire
// window, producing "operand #0 does not dominate this use" because the
// conduit.acquire was placed after the scf.for that uses its result.
//
// After the fix: the inner block emits its own conduit.acquire, maintaining
// SSA dominance.

// CHECK-LABEL: module @ssa_dominance_regression
// The pipeline must complete without errors (exit 0).
// Verify that the inner scf.for body has its OWN use_lock acquire,
// not a reference to a value defined after the scf.for.
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK:           aie.use_lock
// CHECK:         }
// CHECK:       }

module @ssa_dominance_regression {
    aie.device(xcvc1902) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        aie.objectfifo @fifo (%tile12, {%tile13}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

        func.func @work(%buf: memref<16xi32>) -> () {
            return
        }

        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c4 = arith.constant 4 : index
            %cmax = arith.constant 0xFFFFFFFF : index

            // Outer infinite loop
            scf.for %arg0 = %c0 to %cmax step %c1 {
                // Group 1: acquire → use → release
                %sv0 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @work(%elem0) : (memref<16xi32>) -> ()
                aie.objectfifo.release @fifo (Produce, 1)

                // Inner loop: independent acquire/release per iteration
                scf.for %idx = %c0 to %c4 step %c1 {
                    %sv1 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                    %elem1 = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                    func.call @work(%elem1) : (memref<16xi32>) -> ()
                    aie.objectfifo.release @fifo (Produce, 1)
                }

                // Group 2: acquire → use → release (AFTER the inner loop)
                %sv2 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem2 = aie.objectfifo.subview.access %sv2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @work(%elem2) : (memref<16xi32>) -> ()
                aie.objectfifo.release @fifo (Produce, 1)
            }
            aie.end
        }
    }
}
