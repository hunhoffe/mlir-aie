// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Pass A test: cyclostatic access pattern detection.
// Pass A correctly detects the CSDF access pattern {1, 2, 1} and emits
// access_pattern = array<i64: 1, 2, 1> on conduit.create.
// Input has a single objectfifo whose consumer acquires {1, 2, 1} elements
// across three acquire ops — a classic CSDF pattern.
//
// Pass A should:
//   1. Emit conduit.create with access_pattern = array<i64: 1, 2, 1>
//   2. Emit three conduit.acquire ops with counts 1, 2, 1 (preserving the
//      original per-op counts from the aie.objectfifo.acquire ops).
//   3. Erase all aie.objectfifo* ops.

// CHECK-LABEL: module @csdf_pass_a_test
// CHECK:   aie.device(xcve2302) {

// --- conduit.create must carry access_pattern=[1,2,1] ---
// CHECK:     conduit.create
// CHECK-SAME:   access_pattern = array<i64: 1, 2, 1>
// CHECK-SAME:   capacity = 4 : i64
// CHECK-SAME:   depth = 4 : i64
// CHECK-SAME:   name = "fifo"

// --- Producer core: three acquire(Produce,1) / release(Produce,1) pairs ---
// CHECK:     aie.core(%{{.*}}) {
// CHECK:       conduit.acquire
// CHECK-SAME:     count = 1
// CHECK-SAME:     name = "fifo"
// CHECK-SAME:     port = "Produce"
// CHECK:       conduit.release
// CHECK:       conduit.acquire
// CHECK-SAME:     count = 1
// CHECK-SAME:     port = "Produce"
// CHECK:       conduit.release

// --- Consumer core: CSDF sequence {1, 2, 1} ---
// CHECK:     aie.core(%{{.*}}) {
// CHECK:       conduit.acquire
// CHECK-SAME:     count = 1
// CHECK-SAME:     name = "fifo"
// CHECK-SAME:     port = "Consume"
// CHECK:       conduit.release
// CHECK:       conduit.acquire
// CHECK-SAME:     count = 2
// CHECK-SAME:     port = "Consume"
// CHECK:       conduit.release
// CHECK:       conduit.acquire
// CHECK-SAME:     count = 1
// CHECK-SAME:     port = "Consume"
// CHECK:       conduit.release

// --- No residual ObjectFIFO ops ---
// CHECK-NOT: aie.objectfifo
// CHECK-NOT: aie.objectfifo.acquire
// CHECK-NOT: aie.objectfifo.release
// CHECK-NOT: aie.objectfifo.subview

module @csdf_pass_a_test {
    aie.device(xcve2302) {

        %tile22 = aie.tile(2, 2)  // producer tile
        %tile23 = aie.tile(2, 3)  // consumer tile

        aie.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<i32>>

        // Producer: pushes 4 elements one at a time.
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

        // Consumer: CSDF pattern {1, 2, 1}.
        %core23 = aie.core(%tile23) {
            %i0 = arith.constant 0 : index

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
