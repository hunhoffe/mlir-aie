// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// P2-D: sequential acquire pattern (AIE2_delayed_release).
//
// ObjectFIFO acquire(N) semantics: N is the *total* number of elements
// currently needed, not an increment.  When a core calls:
//
//   acquire(2)  → hold 2 elements
//   acquire(1)  → still hold 2 (1 < 2, no additional lock acquisition)
//   acquire(3)  → now hold 3 (need 1 more: delta = 3-2 = 1)
//   acquire(1)  → still hold 3 (1 < 3, no additional)
//   release(3)  → release all 3 held
//
// The stateful transform emits: AcquireGreaterEqual(2), [nothing],
// AcquireGreaterEqual(1), [nothing], Release(3).
//
// Pass A must collapse sequential acquires in the same release-group into
// a single conduit.acquire with count = max_in_group.  This ensures M8's
// invariant (release_count ≤ acquired_count) is satisfied.
//
// Expected Conduit IR for the consumer core:
//   - ONE conduit.acquire{count=3} (max of {2,1,3,1})
//   - FOUR conduit.subview_access ops, all using the same window
//   - ONE conduit.release{count=3}
//   - No spurious phantom-acquire C1 warnings
//
// The producer uses a simple acquire(1)/release(1) loop — the sequential
// acquire logic does not affect single-acquire groups.

// CHECK-LABEL: module @seq_acq_test
// CHECK:   aie.device(xcve2302) {

// CHECK:     conduit.create @fifo
// CHECK-SAME:   depth = 4 : i64

// --- Producer core: uniform 1/1 pattern, unchanged ---
// CHECK:     aie.core(%{{.*}}) {
// CHECK:       conduit.acquire
// CHECK-SAME:     count = 1
// CHECK-SAME:     name = @fifo
// CHECK-SAME:     port = #conduit.port<Produce>
// CHECK:       conduit.subview_access
// CHECK:       conduit.release

// --- Consumer core: sequential acquire {2,1,3,1} → single acquire(3) ---
// CHECK:     aie.core(%{{.*}}) {
// One conduit.acquire with count=3 (the max of the group).
// CHECK:       conduit.acquire
// CHECK-SAME:     count = 3
// CHECK-SAME:     name = @fifo
// CHECK-SAME:     port = #conduit.port<Consume>
// All four subview_access ops reuse the same window (no second acquire).
// CHECK-COUNT-4:  conduit.subview_access
// No second conduit.acquire in the consumer.
// CHECK-NOT:   conduit.acquire
// One conduit.release with the original release count.
// CHECK:       conduit.release
// CHECK-NOT:   conduit.release

module @seq_acq_test {
    aie.device(xcve2302) {
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)
        %buf23 = aie.buffer(%tile23) {sym_name = "buf23"} : memref<4xi32>

        aie.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<i32>>

        // Producer: uniform 1/1 pattern (acquire 1, store, release 1).
        %core22 = aie.core(%tile22) {
            %c99 = arith.constant 99 : i32
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i4 = arith.constant 4 : index
            scf.for %it = %i0 to %i4 step %i1 {
                %sv = aie.objectfifo.acquire @fifo (Produce, 1) : !aie.objectfifosubview<memref<i32>>
                %obj = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
                memref.store %c99, %obj[] : memref<i32>
                aie.objectfifo.release @fifo (Produce, 1)
            }
            aie.end
        }

        // Consumer: sequential acquire pattern {2, 1, 3, 1} with release(3).
        // acquire(2) → want 2 total;  hold = 2
        // acquire(1) → want 1 total;  hold ≥ 1, no new lock needed
        // acquire(3) → want 3 total;  delta = 1
        // acquire(1) → want 1 total;  hold ≥ 1, no new lock needed
        // release(3) → release max held = 3
        %core23 = aie.core(%tile23) {
            %i0 = arith.constant 0 : index
            %i1 = arith.constant 1 : index
            %i2 = arith.constant 2 : index
            %i3 = arith.constant 3 : index

            %sv0 = aie.objectfifo.acquire @fifo (Consume, 2) : !aie.objectfifosubview<memref<i32>>
            %obj0 = aie.objectfifo.subview.access %sv0[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v0 = memref.load %obj0[] : memref<i32>
            memref.store %v0, %buf23[%i0] : memref<4xi32>

            %sv1 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i32>>
            %obj1 = aie.objectfifo.subview.access %sv1[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v1 = memref.load %obj1[] : memref<i32>
            memref.store %v1, %buf23[%i1] : memref<4xi32>

            %sv2 = aie.objectfifo.acquire @fifo (Consume, 3) : !aie.objectfifosubview<memref<i32>>
            %obj2 = aie.objectfifo.subview.access %sv2[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v2 = memref.load %obj2[] : memref<i32>
            memref.store %v2, %buf23[%i2] : memref<4xi32>

            %sv3 = aie.objectfifo.acquire @fifo (Consume, 1) : !aie.objectfifosubview<memref<i32>>
            %obj3 = aie.objectfifo.subview.access %sv3[0] : !aie.objectfifosubview<memref<i32>> -> memref<i32>
            %v3 = memref.load %obj3[] : memref<i32>
            memref.store %v3, %buf23[%i3] : memref<4xi32>

            // Release the maximum held = 3.
            aie.objectfifo.release @fifo (Consume, 3)

            aie.end
        }
    }
}
