// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Cross-block acquire subsumption (AIE2_dynamic_locks pattern).
//
// When a producer acquires once before a loop and the loop body acquires
// again with the same count, the in-loop acquire is subsumed by the
// pre-loop one — no additional use_lock(AcquireGreaterEqual) should be
// emitted inside the loop body.
//
// Input pattern:
//   acquire @fifo (Produce, 1)       <- pre-loop; hold = 1
//   scf.for ... {
//     acquire @fifo (Produce, 1)     <- in-loop; hold already 1, delta = 0
//     store ...
//     release @fifo (Produce, 1)     <- release; hold = 0 after iteration
//   }
//
// Oracle (stateful transform) emits exactly 6 use_lock ops:
//   core: 1 AcquireGreaterEqual(prod_lock) + 1 Release(cons_lock) per iter
//   mem:  2 AcquireGreaterEqual + 2 Release (MM2S + S2MM BD chains)
// Conduit must match: no second AcquireGreaterEqual inside the loop.
//
// Regression for the double-acquire bug: Conduit previously emitted
// AcquireGreaterEqual(fifo_prod_lock_0) twice in the core — once before
// the loop and once inside — deadlocking after the first iteration when
// the DMA held the lock and the core tried to re-acquire it.

// CHECK-LABEL: module @aie2_dynamic_locks

// Exactly one AcquireGreaterEqual on fifo_prod_lock_0 (before the loop).
// CHECK:       aie.use_lock(%[[prod_lock:.*]], AcquireGreaterEqual, 1)
// The scf.for body must NOT have another AcquireGreaterEqual on prod_lock.
// CHECK:       scf.for
// CHECK-NOT:     aie.use_lock(%[[prod_lock]], AcquireGreaterEqual
// CHECK:         memref.store
// CHECK:         aie.use_lock(%[[cons_lock:.*]], Release, 1)
// CHECK:       }
// CHECK:       aie.end

module @aie2_dynamic_locks {
    aie.device(xcve2302) {
        %tile22 = aie.tile(2, 2)  // producer tile
        %tile43 = aie.tile(4, 3)  // consumer tile
        aie.objectfifo @fifo (%tile22, {%tile43}, 1 : i32) : !aie.objectfifo<memref<i64>>

        // Consumer core: provides structural tile info for consumer endpoint.
        %core43 = aie.core(%tile43) {
            %sv = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i64>>
            %e = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<i64>> -> memref<i64>
            aie.objectfifo.release @fifo (Consume, 1)
            aie.end
        }

        // Producer core: acquire before loop, acquire again inside loop
        // (same count — should not generate a second lock acquire),
        // store, release inside loop.
        %core22 = aie.core(%tile22) {
            %i_c0 = arith.constant 0 : index
            %i_c1 = arith.constant 1 : index
            %i_c3 = arith.constant 3 : index
            %c1 = arith.constant 1 : i64

            // Pre-loop acquire: hold = 1.
            %subview0 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i64>>

            scf.for %idx = %i_c0 to %i_c3 step %i_c1 {
                // In-loop acquire with same count: already held, delta = 0.
                // Must NOT emit a second AcquireGreaterEqual.
                %subview = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i64>>
                %elem = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<i64>> -> memref<i64>
                memref.store %c1, %elem[] : memref<i64>
                aie.objectfifo.release @fifo (Produce, 1)
            }

            aie.end
        }
    }
}
