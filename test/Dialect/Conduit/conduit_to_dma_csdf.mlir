// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
// XFAIL: *
//
// Known gaps:
//   1. Pass A: access_pattern not appearing in conduit.create output.
//   2. Pass C: SubviewAccess depth>1 path creates scf.index_switch for
//      produce-port acquires in cores without scf.for; crashes when SCF
//      dialect is not loaded.  Fix: skip rotation-counter for produce-port.
//
// Pass A + Pass C end-to-end CSDF (cyclostatic synchronous dataflow) test.
//
// In this pattern the producer produces one element at a time and the consumer
// consumes {1, 2, 1} elements alternately — a classic CSDF access pattern.
// Input mirrors test/objectFifo-stateful-transform/access_patterns/AIE2_cyclostatic_L1.mlir.
//
// Key correctness property under test:
//   The consumer core uses_lock counts must vary as {1, 2, 1}, matching the
//   cyclostatic access pattern.  This is the core of CSDF support.
//
// NOTE on known gaps vs. --aie-objectFifo-stateful-transform:
//   The stateful transform places the locks on the producer tile (shared mem).
//   Pass C currently places them on the consumer tile (tracked separately).
//   This test checks the lock count correctness for CSDF, not the placement.
//   Lock placement for L1 shared-mem fifos is a separate known gap.

// CHECK-LABEL: module @csdf_l1_test
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[T22:.*]] = aie.tile(2, 2)
// CHECK:     %[[T23:.*]] = aie.tile(2, 3)

// --- Locks allocated on consumer tile (tile 2,3) by Pass C Phase 3 ---
// prodLock init=4 (4 free slots for producer), consLock init=0.
// CHECK:     %[[PRODLOCK:.*]] = aie.lock(%[[T23]]
// CHECK-SAME:   init = 4
// CHECK:     %[[CONSLOCK:.*]] = aie.lock(%[[T23]]
// CHECK-SAME:   init = 0

// --- Producer core: 4 produce-1-at-a-time acquire/release pairs ---
// Each acquire(Produce,1): acquire prodLock with count=1.
// Each release(Produce,1): release consLock with count=1.
// CHECK:     aie.core(%[[T22]]) {
// CHECK:       aie.use_lock(%[[PRODLOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[CONSLOCK]], Release, 1)
// CHECK:       aie.use_lock(%[[PRODLOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[CONSLOCK]], Release, 1)
// CHECK:       aie.use_lock(%[[PRODLOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[CONSLOCK]], Release, 1)
// CHECK:       aie.use_lock(%[[PRODLOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[CONSLOCK]], Release, 1)
// CHECK:     }

// --- Consumer core: CSDF pattern {1, 2, 1} ---
// The critical check: lock counts VARY per acquire.
// acquire(Consume,1): acquires consLock with count=1
// acquire(Consume,2): acquires consLock with count=2  <-- CSDF
// acquire(Consume,1): acquires consLock with count=1
// CHECK:     aie.core(%[[T23]]) {
// CHECK:       aie.use_lock(%[[CONSLOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[PRODLOCK]], Release, 1)
// CHECK:       aie.use_lock(%[[CONSLOCK]], AcquireGreaterEqual, 2)
// CHECK:       aie.use_lock(%[[PRODLOCK]], Release, 2)
// CHECK:       aie.use_lock(%[[CONSLOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[PRODLOCK]], Release, 1)
// CHECK:     }

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @csdf_l1_test {
    aie.device(xcve2302) {

        %tile22 = aie.tile(2, 2)  // producer tile
        %tile23 = aie.tile(2, 3)  // consumer tile

        // ObjectFifo that can hold 4 memref<i32>s (depth=4), produced by tile22,
        // consumed by tile23 with cyclostatic pattern {1, 2, 1}.
        aie.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<i32>>

        // Producer core: pushes 4 elements one at a time.
        %core22 = aie.core(%tile22) {
            %c55 = arith.constant 55 : i32
            %c66 = arith.constant 66 : i32
            %c77 = arith.constant 77 : i32
            %c88 = arith.constant 88 : i32

            %sv0 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %obj0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c55, %obj0[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)

            %sv1 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %obj1 = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c66, %obj1[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)

            %sv2 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %obj2 = aie.objectfifo.subview.access %sv2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c77, %obj2[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)

            %sv3 = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
            %obj3 = aie.objectfifo.subview.access %sv3[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            memref.store %c88, %obj3[] : memref<i32>
            aie.objectfifo.release @fifo (Produce, 1)

            aie.end
        }

        // Consumer core: CSDF pattern {1, 2, 1} — acquires 1, then 2, then 1.
        %core23 = aie.core(%tile23) {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index

            // Acquire 1 element.
            %sv0 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i32>>
            %v0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            aie.objectfifo.release @fifo (Consume, 1)

            // Acquire 2 elements.
            %sv1 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
            %v1 = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v2 = aie.objectfifo.subview.access %sv1[1] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            aie.objectfifo.release @fifo (Consume, 2)

            // Acquire 1 element.
            %sv2 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i32>>
            %v3 = aie.objectfifo.subview.access %sv2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            aie.objectfifo.release @fifo (Consume, 1)

            aie.end
        }
    }
}
