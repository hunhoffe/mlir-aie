// RUN: aie-opt --objectfifo-to-conduit=infer-rates=true %s 2>&1 | FileCheck %s --check-prefix=CHECK-RATES
// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s --check-prefix=CHECK-NO-RATES
//
// Pass A infer-rates option tests.
//
// Test 1: single-consumer uniform acquire (count=2 always) — rates annotated.
//   producer: release(2) per iteration → inferred producer_rates = [2]
//   consumer: acquire(2) per iteration → inferred consumer_rates = [2]
//   With infer-rates=true: producer_rates = [2], consumer_rates = [2] on conduit.create.
//   With infer-rates=false (default): no rate annotations.
//
// Test 2: multi-consumer fifo — rates NOT annotated (remark emitted).
//   Two consumer cores: core A acquires 1, core B acquires 2.
//   The old bug merged these into [1,2] — a spurious CSDF pattern.
//   With infer-rates=true: remark emitted, no producer_rates/consumer_rates attached.
//
// Test 3: sliding-window fifo — rates NOT annotated (remark emitted, MVE-2).
//   Consumer: acquire(3)/release(1) — sliding window, keeps 3 slots, releases 1.
//   max(acquireCount=3) > min(releaseCount=1) → sliding-window detected.
//   With infer-rates=true: remark emitted, no producer_rates/consumer_rates attached.
//   Without guard: M6 would false-reject as CSDF-unbalanced (consumer_rate=3 ≠ producer_rate=1).
//
// These tests verify:
//   (a) infer-rates=true attaches rates for single-consumer fifos.
//   (b) multi-consumer fifos are skipped with a remark.
//   (c) infer-rates=false (default) produces NO rate annotations.
//   (d) sliding-window fifos are skipped with a remark (MVE-2 guard).

// ---------------------------------------------------------------------------
// Test 2: remark is emitted for multi_consumer BEFORE the module output (stderr+stdout merged).
// CHECK-RATES: remark: conduit-objectfifo: skipping CSDF rate annotation for multi-consumer fifo 'multi_consumer'

// Test 3: remark is emitted for sliding_window fifo (MVE-2).
// CHECK-RATES: remark: conduit-objectfifo: skipping CSDF rate annotation for sliding-window fifo 'sliding_window'

// Test 1: single_consumer conduit.create gets rates.
// CHECK-RATES: conduit.create @single_consumer
// CHECK-RATES-SAME: consumer_rates = array<i64: 2>
// CHECK-RATES-SAME: producer_rates = array<i64: 2>

// Test 2: multi_consumer conduit.create has no rates (only slot_elems/consumer_tiles/depth).
// CHECK-RATES: conduit.create @multi_consumer {consumer_tiles = array<i64: 2, 2, 4, 2>, depth = 4 : i64, element_type = memref<16xi32>, producer_tile = array<i64: 3, 2>, slot_elems = 64 : i64}

// Test 3: sliding_window conduit.create has no rates (MVE-2 guard prevents false M6 rejection).
// CHECK-RATES-NOT: conduit.create @sliding_window {depth = 0 : i64, {.*}}consumer_rates
// CHECK-RATES-NOT: conduit.create @sliding_window {depth = 0 : i64, {.*}}producer_rates

// ---------------------------------------------------------------------------
// Default (infer-rates=false): no rate annotations on any conduit.create.
// CHECK-NO-RATES-NOT: producer_rates
// CHECK-NO-RATES-NOT: consumer_rates

module @infer_rates_test {
    aie.device(xcve2302) {

        %tile12 = aie.tile(1, 2)  // producer tile (single_consumer)
        %tile22 = aie.tile(2, 2)  // consumer tile A (single_consumer + multi_consumer)
        %tile32 = aie.tile(3, 2)  // producer tile (multi_consumer)
        %tile42 = aie.tile(4, 2)  // consumer tile B (multi_consumer)
        %tile52 = aie.tile(5, 2)  // producer tile (sliding_window)
        %tile62 = aie.tile(6, 2)  // consumer tile (sliding_window)

        // Single-consumer fifo: producer → one consumer.
        // Producer releases 2 per round; consumer acquires 2 per round.
        aie.objectfifo @single_consumer (%tile12, {%tile22}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

        // Multi-consumer fifo: one producer → two consumers.
        // Core at tile22 acquires 1; core at tile42 acquires 2.
        // The old bug would merge these into [1,2] — a spurious CSDF pattern.
        aie.objectfifo @multi_consumer (%tile32, {%tile22, %tile42}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

        // Sliding-window fifo: producer releases 1 per round; consumer acquires
        // 3 but releases only 1 (keeps a window of 3 overlapping elements).
        // max(acquireCount=3) > min(releaseCount=1) → MVE-2 guard fires.
        // Without guard: M6 would see consumer_rate=3, producer_rate=1 →
        // CSDF-unbalanced false rejection.
        aie.objectfifo @sliding_window (%tile52, {%tile62}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

        // Producer for single_consumer: releases 2 per round.
        %core12 = aie.core(%tile12) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c4 = arith.constant 4 : index
            scf.for %i = %c0 to %c4 step %c1 {
                %sv = aie.objectfifo.acquire @single_consumer (Produce, 2) : !aie.objectfifosubview<memref<16xi32>>
                %e0 = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %e1 = aie.objectfifo.subview.access %sv[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                aie.objectfifo.release @single_consumer (Produce, 2)
            }
            aie.end
        }

        // Consumer for single_consumer (tile22): acquires 2 per round.
        %core22a = aie.core(%tile22) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c4 = arith.constant 4 : index
            scf.for %i = %c0 to %c4 step %c1 {
                %sv = aie.objectfifo.acquire @single_consumer (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
                %e0 = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %e1 = aie.objectfifo.subview.access %sv[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                aie.objectfifo.release @single_consumer (Consume, 2)
            }
            aie.end
        }

        // Producer for multi_consumer: releases 2 per round.
        %core32 = aie.core(%tile32) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c4 = arith.constant 4 : index
            scf.for %i = %c0 to %c4 step %c1 {
                %sv = aie.objectfifo.acquire @multi_consumer (Produce, 2) : !aie.objectfifosubview<memref<16xi32>>
                %e0 = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %e1 = aie.objectfifo.subview.access %sv[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                aie.objectfifo.release @multi_consumer (Produce, 2)
            }
            aie.end
        }

        // Consumer A for multi_consumer (tile22): acquires 1 per round.
        %core22b = aie.core(%tile22) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c4 = arith.constant 4 : index
            scf.for %i = %c0 to %c4 step %c1 {
                %sv = aie.objectfifo.acquire @multi_consumer (Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
                %e0 = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                aie.objectfifo.release @multi_consumer (Consume, 1)
            }
            aie.end
        }

        // Consumer B for multi_consumer (tile42): acquires 2 per round.
        %core42 = aie.core(%tile42) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c4 = arith.constant 4 : index
            scf.for %i = %c0 to %c4 step %c1 {
                %sv = aie.objectfifo.acquire @multi_consumer (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
                %e0 = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %e1 = aie.objectfifo.subview.access %sv[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                aie.objectfifo.release @multi_consumer (Consume, 2)
            }
            aie.end
        }

        // Producer for sliding_window (tile52): releases 1 per round.
        %core52 = aie.core(%tile52) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            scf.for %i = %c0 to %c64 step %c1 {
                %sv = aie.objectfifo.acquire @sliding_window (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
                %e0 = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                aie.objectfifo.release @sliding_window (Produce, 1)
            }
            aie.end
        }

        // Consumer for sliding_window (tile62): acquire(3)/release(1) sliding window.
        // Keeps 3 elements in the window; advances by 1 per step.
        // max(acquireCount=3) > min(releaseCount=1) → MVE-2 guard skips rates.
        %core62 = aie.core(%tile62) {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %c62 = arith.constant 62 : index
            scf.for %i = %c0 to %c62 step %c1 {
                %sv = aie.objectfifo.acquire @sliding_window (Consume, 3) : !aie.objectfifosubview<memref<16xi32>>
                %e0 = aie.objectfifo.subview.access %sv[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %e1 = aie.objectfifo.subview.access %sv[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                %e2 = aie.objectfifo.subview.access %sv[2] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
                aie.objectfifo.release @sliding_window (Consume, 1)
            }
            aie.end
        }
    }
}
