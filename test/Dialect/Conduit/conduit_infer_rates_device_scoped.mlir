// RUN: aie-opt --conduit-infer-rates --split-input-file --verify-diagnostics %s | FileCheck %s
//
// --conduit-infer-rates: device-scoped walk + correction cases.
//
// Three separate modules (split by the split-input-file separator) so
// FileCheck can use CHECK-LABEL anchors independently for each case.
//
// Case 1 — SPSC channel inside aie.device:
//   Rates ARE inferred. Verifies device-scoped walk works correctly.
//
// Case 2 — Sliding-window channel (correction case a):
//   acquire count (3) > release count (1). Rates NOT inferred. Pass emits a
//   remark explaining the skip; CHECK-NOT verifies no rates in the output.
//
// Case 3 — dma_repeat channel (correction case b):
//   dma_repeat set → skip ALL rate attachment. Pass emits a remark.
//   No producer_rates or consumer_rates in the output.
//
// Attribute printing order is alphabetical; use CHECK-DAG for unordered
// attribute matching within a block.

// -----

// Case 1: SPSC channel inside aie.device — rates ARE inferred.
//
// Expected:
//   - conduit-infer-rates walks the aie.device body (device-scoped).
//   - Infers producer_rates = [32], consumer_rates = [32] from put/get num_elems.
//   - M6 balance: sum(P)*len(C) = 32*1 == sum(C)*len(P) = 32*1 → passes.
//   - Remark emitted on conduit.create.

// CHECK-LABEL: module @case1_spsc
// CHECK:       conduit.create @simple_chan
// CHECK-DAG:   consumer_rates = array<i64: 32>
// CHECK-DAG:   producer_rates = array<i64: 32>

module @case1_spsc {
  aie.device(npu2) {
    // expected-remark@+1 {{conduit-infer-rates: attached producer_rates=[32] consumer_rates=[32] to conduit 'simple_chan'}}
    conduit.create @simple_chan {slot_elems = 32 : i64,
                    depth = 1 : i64,
                    element_type = memref<32xi32>}

    aie.runtime_sequence(%buf : memref<32xi32>) {
      %tok = conduit.put_memref_async {name = @simple_chan, num_elems = 32 : i64,
                                       offsets = array<i64: 0>,
                                       sizes   = array<i64: 32>,
                                       strides = array<i64: 1>}
                                      : !conduit.dma.token
      conduit.wait_all %tok : !conduit.dma.token
    }

    %core = aie.tile(0, 2)
    aie.core(%core) {
      %tok = conduit.get_memref_async {name = @simple_chan, num_elems = 32 : i64,
                                       offsets = array<i64: 0>,
                                       sizes   = array<i64: 32>,
                                       strides = array<i64: 1>}
                                      : !conduit.dma.token
      conduit.wait_all %tok : !conduit.dma.token
      aie.end
    }
  }
}

// -----

// Case 2: Sliding-window channel (correction case a) — rates NOT inferred.
//
// @sliding_win has:
//   conduit.acquire {count = 3}  — holds window of 3 slots
//   conduit.release {count = 1}  — releases 1 slot per step (slides by 1)
//
// max(acquire count) = 3 > min(release count) = 1 → sliding window detected.
// Emitting CSDF rates would cause M6 to report false imbalance.
// Correction case (a): skip rate attachment; emit remark.
//
// Expected:
//   - Remark emitted: "sliding-window channel (acquire count 3 > release count 1)"
//   - conduit.create @sliding_win has NO producer_rates or consumer_rates.
//
// The CHECK-NOT patterns are bounded: they apply from the @sliding_win match
// until the CHECK-LABEL for Case 3 terminates the scan.

// CHECK-LABEL: module @case2_sliding_window
// CHECK:       conduit.create @sliding_win
// CHECK-SAME:  slot_elems = 1
// CHECK-NOT:   producer_rates
// CHECK-NOT:   consumer_rates

module @case2_sliding_window {
  aie.device(npu2) {
    // expected-remark@+1 {{conduit-infer-rates: skipping 'sliding_win': sliding-window channel (acquire count 3 > release count 1)}}
    conduit.create @sliding_win {slot_elems = 1 : i64,
                    depth = 3 : i64,
                    element_type = memref<1xi32>}

    %prod_tile = aie.tile(0, 2)
    %cons_tile = aie.tile(1, 2)

    aie.core(%prod_tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %w = conduit.acquire {name = @sliding_win, count = 1 : i64,
                              port = #conduit.port<Produce>}
               : !conduit.window<memref<1xi32>>
        conduit.release %w {count = 1 : i64, port = #conduit.port<Produce>}
            : !conduit.window<memref<1xi32>>
      }
      aie.end
    }

    aie.core(%cons_tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c6 = arith.constant 6 : index
      // Preamble: acquire initial window of 3
      %wpre = conduit.acquire {name = @sliding_win, count = 3 : i64,
                               port = #conduit.port<Consume>}
               : !conduit.window<memref<1xi32>>
      conduit.release %wpre {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<1xi32>>
      // Loop body: acquire 3 (hold window), release 1 (slide by 1)
      scf.for %i = %c0 to %c6 step %c1 {
        %w = conduit.acquire {name = @sliding_win, count = 3 : i64,
                              port = #conduit.port<Consume>}
               : !conduit.window<memref<1xi32>>
        conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<1xi32>>
      }
      aie.end
    }
  }
}

// -----

// Case 3: dma_repeat channel (correction case b) — rates NOT inferred.
//
// @iterating has dma_repeat = 4 (maps to objectfifo iter_count).
// Pass C infers BD chain length from putCount directly and does not need
// producer_rates.  Emitting only producer_rates without consumer_rates would
// violate M6 on any subsequent verify pass.
// Correction case (b): skip ALL rate attachment; emit remark.
//
// Expected:
//   - Remark: "dma_repeat set; Pass C infers BD chain length..."
//   - conduit.create @iterating has NO producer_rates or consumer_rates.

// CHECK-LABEL: module @case3_dma_repeat
// CHECK:       conduit.create @iterating
// CHECK-SAME:  dma_repeat = 4
// CHECK-NOT:   producer_rates
// CHECK-NOT:   consumer_rates

module @case3_dma_repeat {
  aie.device(npu2) {
    // expected-remark@+1 {{conduit-infer-rates: skipping 'iterating': dma_repeat set; Pass C infers BD chain length from putCount independently}}
    conduit.create @iterating {slot_elems = 16 : i64,
                    depth = 1 : i64,
                    dma_repeat = 4 : i64,
                    element_type = memref<16xi32>}

    %core = aie.tile(0, 2)

    aie.runtime_sequence(%buf : memref<64xi32>) {
      // One put per invocation; dma_repeat=4 means DMA fires 4 times total.
      %tok = conduit.put_memref_async {name = @iterating, num_elems = 16 : i64,
                                       offsets = array<i64: 0>,
                                       sizes   = array<i64: 16>,
                                       strides = array<i64: 1>}
                                      : !conduit.dma.token
      conduit.wait_all %tok : !conduit.dma.token
    }

    aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = conduit.acquire {name = @iterating, count = 1 : i64,
                              port = #conduit.port<Consume>}
               : !conduit.window<memref<16xi32>>
        conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<16xi32>>
      }
      aie.end
    }
  }
}
