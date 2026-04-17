// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s 2>&1 | FileCheck %s --check-prefix=CONDUIT
// RUN: aie-opt --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s --check-prefix=ORACLE
//
// DEFERRED-6: Cross-block acquire lit test — nested conditional acquire.
//
// Validates findWindowInDominatingBlock cross-block subsumption when a core
// acquires before an scf.if and both arms also acquire the same fifo.
//
// Variant 1 (consumer on tile(4,3), fifo1 depth=3):
//   acquire(1) before scf.if, acquire(1) in each arm, release(1) in each arm.
//   The arm acquires are subsumed (1 <= 1) — no additional lock acquisition.
//
// Variant 2 (consumer on tile(3,3), fifo2 depth=3):
//   acquire(2) before scf.if, acquire(1) in each arm, release(1) in each arm,
//   release(1) after scf.if.  The arm acquires are subsumed (1 <= 2).
//
// Both variants must produce the same lock structure as the stateful transform
// oracle: one AcquireGreaterEqual before the if, no lock acquire in the arms,
// Release in each arm.  No C1 phantom warning should be emitted.
//
// Resource counts (buffers, BD chain depth) differ between Conduit and Oracle
// — this is expected and documented (CLAUDE.md: "Resource count parity with
// oracle is NOT required").

// ---- Module anchor ----
// CONDUIT-LABEL: module @cross_block_cond_acquire
// ORACLE-LABEL:  module @cross_block_cond_acquire

// ---- Variant 1: acquire(1) before if + acquire(1) in each arm ----
//
// Consumer core for fifo1: one AcquireGreaterEqual(1) before scf.if,
// no AcquireGreaterEqual inside either arm, Release(1) in each arm.
//
// CONDUIT:       aie.use_lock(%fifo1_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CONDUIT:       scf.if
// CONDUIT-NOT:   AcquireGreaterEqual
// CONDUIT:       aie.use_lock(%fifo1_cons_prod_lock_0, Release, 1)
// CONDUIT:       } else {
// CONDUIT-NOT:   AcquireGreaterEqual
// CONDUIT:       aie.use_lock(%fifo1_cons_prod_lock_0, Release, 1)
// CONDUIT:       aie.end

// ORACLE:        aie.use_lock(%fifo1_cons_cons_lock_0, AcquireGreaterEqual, 1)
// ORACLE:        scf.if
// ORACLE-NOT:    AcquireGreaterEqual
// ORACLE:        aie.use_lock(%fifo1_cons_prod_lock_0, Release, 1)
// ORACLE:        } else {
// ORACLE-NOT:    AcquireGreaterEqual
// ORACLE:        aie.use_lock(%fifo1_cons_prod_lock_0, Release, 1)
// ORACLE:        aie.end

// ---- Variant 2: acquire(2) before if + acquire(1) in each arm ----
//
// Consumer core for fifo2: one AcquireGreaterEqual(2) before scf.if,
// no AcquireGreaterEqual inside either arm, Release(1) in each arm,
// Release(1) after the if.
//
// CONDUIT:       aie.use_lock(%fifo2_cons_cons_lock_0, AcquireGreaterEqual, 2)
// CONDUIT:       scf.if
// CONDUIT-NOT:   AcquireGreaterEqual
// CONDUIT:       aie.use_lock(%fifo2_cons_prod_lock_0, Release, 1)
// CONDUIT:       } else {
// CONDUIT-NOT:   AcquireGreaterEqual
// CONDUIT:       aie.use_lock(%fifo2_cons_prod_lock_0, Release, 1)
// CONDUIT:       }
// CONDUIT:       aie.use_lock(%fifo2_cons_prod_lock_0, Release, 1)
// CONDUIT:       aie.end

// ORACLE:        aie.use_lock(%fifo2_cons_cons_lock_0, AcquireGreaterEqual, 2)
// ORACLE:        scf.if
// ORACLE-NOT:    AcquireGreaterEqual
// ORACLE:        aie.use_lock(%fifo2_cons_prod_lock_0, Release, 1)
// ORACLE:        } else {
// ORACLE-NOT:    AcquireGreaterEqual
// ORACLE:        aie.use_lock(%fifo2_cons_prod_lock_0, Release, 1)
// ORACLE:        }
// ORACLE:        aie.use_lock(%fifo2_cons_prod_lock_0, Release, 1)
// ORACLE:        aie.end

// ---- No C1 phantom warnings (scanned from last match to end of input) ----
// CONDUIT-NOT:   phantom
// ORACLE-NOT:    phantom

module @cross_block_cond_acquire {
    aie.device(xcve2302) {
        // Tiles: non-adjacent pairs to ensure DMA path.
        %tile12 = aie.tile(1, 2)  // producer for fifo1
        %tile43 = aie.tile(4, 3)  // consumer for fifo1 (variant 1)
        %tile22 = aie.tile(2, 2)  // producer for fifo2
        %tile33 = aie.tile(3, 3)  // consumer for fifo2 (variant 2)

        aie.objectfifo @fifo1 (%tile12, {%tile43}, 3 : i32) : !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @fifo2 (%tile22, {%tile33}, 3 : i32) : !aie.objectfifo<memref<16xi32>>

        func.func @work(%buf: memref<16xi32>) -> () {
            return
        }

        // ---------- Producers ----------

        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c3 = arith.constant 3 : index
            scf.for %i = %c0 to %c3 step %c1 {
                %sv = aie.objectfifo.acquire @fifo1 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @work(%elem) : (memref<16xi32>) -> ()
                aie.objectfifo.release @fifo1 (Produce, 1)
            }
            aie.end
        }

        %core22 = aie.core(%tile22) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c3 = arith.constant 3 : index
            scf.for %i = %c0 to %c3 step %c1 {
                %sv = aie.objectfifo.acquire @fifo2 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @work(%elem) : (memref<16xi32>) -> ()
                aie.objectfifo.release @fifo2 (Produce, 1)
            }
            aie.end
        }

        // ---------- Consumer 1 (Variant 1): acquire(1) + acquire(1) in arms ----------
        //
        // Outer acquire(1) before scf.if, acquire(1) in both arms, release(1)
        // in each arm.  The arm acquires see hold=1, 1<=1, delta=0 — subsumed.

        %core43 = aie.core(%tile43) {
            %cond = arith.constant true

            %sv0 = aie.objectfifo.acquire @fifo1 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>

            scf.if %cond {
                %sv1 = aie.objectfifo.acquire @fifo1 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @work(%elem) : (memref<16xi32>) -> ()
                aie.objectfifo.release @fifo1 (Consume, 1)
            } else {
                %sv2 = aie.objectfifo.acquire @fifo1 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem = aie.objectfifo.subview.access %sv2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @work(%elem) : (memref<16xi32>) -> ()
                aie.objectfifo.release @fifo1 (Consume, 1)
            }
            aie.end
        }

        // ---------- Consumer 2 (Variant 2): acquire(2) + acquire(1) in arms ----------
        //
        // Outer acquire(2) before scf.if, use one element, then acquire(1) in
        // both arms (subsumed: 1<=2), release(1) in each arm, release(1) after.

        %core33 = aie.core(%tile33) {
            %cond = arith.constant true

            %sv0 = aie.objectfifo.acquire @fifo2 (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
            %elem0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
            func.call @work(%elem0) : (memref<16xi32>) -> ()

            scf.if %cond {
                %sv1 = aie.objectfifo.acquire @fifo2 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @work(%elem) : (memref<16xi32>) -> ()
                aie.objectfifo.release @fifo2 (Consume, 1)
            } else {
                %sv2 = aie.objectfifo.acquire @fifo2 (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %elem = aie.objectfifo.subview.access %sv2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                func.call @work(%elem) : (memref<16xi32>) -> ()
                aie.objectfifo.release @fifo2 (Consume, 1)
            }

            aie.objectfifo.release @fifo2 (Consume, 1)

            aie.end
        }
    }
}
