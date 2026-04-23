// RUN: aie-opt --aie-combine-device="same-tile=true" %s | FileCheck %s
//
// Lit-gap-audit finding 3.2 — in-place rewrite branch at
// `DeviceMergeUtils.cpp:319-321`.
//
// Source path under test:
//   When a host-side `aiex.configure @devB` has NO preceding sibling
//   `aiex.configure @devA` in its parent block,
//   `rewriteHostConfigureOnDeviceMerge` takes the in-place rewrite branch:
//     confB.setSymbolAttr(devARef);
//     if (runB && survSeqRef) runB.setRuntimeSequenceSymbolAttr(survSeqRef);
//
// This branch is partially covered by `lit_audit_2_1_devA_no_sequence.mlir`
// (devA has no sequence) and by `lit_audit_2_2_configure_order_reversed.mlir`
// (confB precedes confA so it cannot find a preceding sibling).  This test
// isolates the case where:
//   - BOTH devices have their own runtime_sequence (so the seqA-promotion
//     path is NOT exercised — the survSeqRef comes from the existing seqA's
//     name, not a promoted seqB).
//   - The host has ONLY `aiex.configure @devB` — there is no
//     `aiex.configure @devA` anywhere in the module's host orchestrator.
//
// Expected end-state:
//   - confB.symbol → @devA.
//   - confB's inner `aiex.run`'s runtime_sequence symbol → seqA's name
//     (i.e., devA's sequence symbol).  This is the audit-flagged subtlety:
//     the `aiex.run` was originally calling `@seqB`, but after the merge
//     seqB has been spliced into seqA, so the run must be retargeted to
//     `@seqA`.

// CHECK-LABEL: module @lit_audit_3_2_in_place_rewrite_no_sibling

// Single named device after merge:
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device(npu2) @devB

// The surviving sequence is devA's @seqA (seqB was cloned/appended into
// it — the simple-merge path with seqA pre-existing).
// CHECK:       aie.runtime_sequence @seqA

// In-place rewrite: confB.symbol → @devA, run sym → @seqA (NOT @seqB).
// CHECK:       aiex.configure @devA
// CHECK-NOT:   aiex.configure @devB
// CHECK:       aiex.run @seqA
// CHECK-NOT:   aiex.run @seqB

module @lit_audit_3_2_in_place_rewrite_no_sibling {
  aie.device(npu2) @devA {
    %tile = aie.tile(0, 2)

    conduit.create @chanA {element_type = memref<128xbf16>, depth = 2 : i64,
                           fusion_group = "g0"}

    func.func private @kernelA(memref<128xbf16>)

    aie.core(%tile) {
      %win = conduit.acquire {name = @chanA, count = 1 : i64,
                              port = #conduit.port<Produce>}
                 : !conduit.window<memref<128xbf16>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<128xbf16>> -> memref<128xbf16>
      func.call @kernelA(%buf) : (memref<128xbf16>) -> ()
      conduit.release %win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<128xbf16>>
      aie.end
    }

    aie.runtime_sequence @seqA(%a0: memref<128xbf16>) {
    }
  }

  aie.device(npu2) @devB {
    %tile = aie.tile(0, 2)

    conduit.create @chanB {element_type = memref<128xbf16>, depth = 2 : i64,
                           fusion_group = "g0"}

    func.func private @kernelB(memref<128xbf16>)

    aie.core(%tile) {
      %win = conduit.acquire {name = @chanB, count = 1 : i64,
                              port = #conduit.port<Consume>}
                 : !conduit.window<memref<128xbf16>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<128xbf16>> -> memref<128xbf16>
      func.call @kernelB(%buf) : (memref<128xbf16>) -> ()
      conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xbf16>>
      aie.end
    }

    aie.runtime_sequence @seqB(%b0: memref<128xbf16>) {
    }
  }

  // Host orchestrator: ONLY aiex.configure @devB — no @devA anywhere.
  // This forces the in-place rewrite branch unconditionally.
  aie.device(npu2) {
    aie.runtime_sequence(%h0: memref<128xbf16>) {
      aiex.configure @devB {
        aiex.run @seqB(%h0) : (memref<128xbf16>)
      }
    }
  }
}
