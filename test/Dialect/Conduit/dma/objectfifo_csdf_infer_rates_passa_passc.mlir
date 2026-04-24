// RUN: aie-opt --objectfifo-to-conduit=infer-rates=true --conduit-depth-promote --conduit-to-dma %s | FileCheck %s
//
// Pass A + depth-promote + Pass C end-to-end test for CSDF inferred-rates.
//
// Coverage gap (audit GAP A): no end-to-end lit had previously exercised the
// `--objectfifo-to-conduit=infer-rates=true` path through depth-promote and
// all the way down to DMA BDs.  Existing tests cover Pass A in isolation
// (objectfifo_csdf_infer_rates.mlir) and Pass C with hand-written rates
// (dma/conduit_to_dma_csdf.mlir).  This test composes all three so that a
// future regression in rate propagation between the passes is caught at lit.
//
// Topology mirrors dma/conduit_to_dma_csdf.mlir: shared-memory adjacency
// between producer tile(2,2) and consumer tile(2,3); consumer uses a CSDF
// {1, 2, 1} acquire pattern; producer releases 1 element at a time.  With
// infer-rates=true Pass A attaches consumer_rates = [1, 2, 1] and
// producer_rates = [1, 1, 1, 1] on the conduit.create.  Pass C consumes
// these rates (no producer_rates/consumer_rates remain in the final IR)
// and emits a CSDF-aware lock-use sequence on the consumer.

// CHECK-LABEL: module @csdf_infer_rates_passa_passc
// CHECK:   aie.device(xcve2302) {

// Producer-side conduit.create input shape (single CHECK on input).
// The objectfifo has depth=4 and a single consumer; Pass A's infer-rates
// hook attaches CSDF rates that depth-promote then consumes.

// --- Locks on the producer tile (shared-memory path) ---
// prodLock init = depth (4 free slots), consLock init = 0.
// CHECK:     %[[PRODLOCK:.*]] = aie.lock(%{{.*}})
// CHECK-SAME:   init = 4
// CHECK:     %[[CONSLOCK:.*]] = aie.lock(%{{.*}})
// CHECK-SAME:   init = 0

// --- Consumer core: CSDF use_lock pattern {1, 2, 1} survives end-to-end ---
// CHECK:     aie.core(%{{.*}}) {
// CHECK:       aie.use_lock(%[[CONSLOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[PRODLOCK]], Release, 1)
// CHECK:       aie.use_lock(%[[CONSLOCK]], AcquireGreaterEqual, 2)
// CHECK:       aie.use_lock(%[[PRODLOCK]], Release, 2)
// CHECK:       aie.use_lock(%[[CONSLOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[PRODLOCK]], Release, 1)

// --- Pass C consumes rates; nothing carries them forward ---
// CHECK-NOT: producer_rates
// CHECK-NOT: consumer_rates
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @csdf_infer_rates_passa_passc {
    aie.device(xcve2302) {

        %tile22 = aie.tile(2, 2)  // producer tile
        %tile23 = aie.tile(2, 3)  // consumer tile (adjacent to tile22)

        // depth=4 single-consumer fifo with CSDF access pattern.
        aie.objectfifo @fifo (%tile22, {%tile23}, 4 : i32) : !aie.objectfifo<memref<i32>>

        // Producer: pushes 4 elements one at a time → producer_rates = [1,1,1,1].
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

        // Consumer: CSDF pattern {1, 2, 1} → consumer_rates = [1, 2, 1].
        %core23 = aie.core(%tile23) {
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
