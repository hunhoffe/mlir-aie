// RUN: aie-opt --aie-combine-device="same-tile=true" %s | FileCheck %s
//
// Lit-gap-audit finding 2.1 — `survivingSeqName` fallback when devA has NO
// `aie.runtime_sequence` and devB does.
//
// Source path under test:
//   `mergeRuntimeSequencesSimple` (DeviceMergeUtils.cpp:116-120) takes the
//   "promote" branch: `seqA = seqB` — devB's runtime_sequence op is moved
//   into devA wholesale (NOT cloned), preserving its sym_name verbatim.
//
//   Then `rewriteHostConfigureOnDeviceMerge` (DeviceMergeUtils.cpp:210-218)
//   reads the surviving sym from `seqA` (which is now the moved seqB) and
//   uses it to rewrite host-side `aiex.run` symbol attrs.
//
// The audit flag: the in-place setSymbolAttr branch at lines 319-321 relies
// on this implicit name preservation but no test asserts it. If
// `mergeRuntimeSequencesSimple` ever switches to clone-then-rename (instead
// of move), or if `survSeqRef` is computed wrong when seqA was promoted, the
// host's `aiex.run @<seqB-name>` would silently target a stale symbol.
//
// What this test exercises:
//   - devA: NO runtime_sequence; only conduit channel + core consuming it.
//   - devB: HAS runtime_sequence with a uniquely-named symbol
//     `@unique_seqB_name` whose persistence we want to assert.
//   - Host orchestrator: a single `aiex.configure @devB { aiex.run
//     @unique_seqB_name(...) }` block. After the merge:
//       * confB has no sibling `aiex.configure @devA` to fold into → goes
//         through the in-place rewrite branch.
//       * The runtime_sequence symbol in the merged device must STILL be
//         `@unique_seqB_name` (not renamed by the merge).
//       * The host's `aiex.run @unique_seqB_name` must continue to resolve.

// CHECK-LABEL: module @lit_audit_2_1_devA_no_sequence

// Single surviving named device (devB folded into devA):
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device(npu2) @devB

// CRITICAL — the runtime_sequence symbol from devB must be preserved
// verbatim after the move-into-devA promotion.  If the move-vs-clone
// distinction in `mergeRuntimeSequencesSimple` is ever broken, the symbol
// will diverge and the host run below will reference a stale symbol.
// CHECK:       aie.runtime_sequence @unique_seqB_name

// In-place rewrite branch: confB.symbol → @devA, run sym retained.
// CHECK:       aiex.configure @devA
// CHECK-NOT:   aiex.configure @devB
// CHECK:       aiex.run @unique_seqB_name

module @lit_audit_2_1_devA_no_sequence {
  aie.device(npu2) @devA {
    %tile_a = aie.tile(0, 2)

    conduit.create @chanA {element_type = memref<128xbf16>, depth = 2 : i64,
                           fusion_group = "g0"}

    func.func private @kernelA(memref<128xbf16>)

    aie.core(%tile_a) {
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
    aie.end
  }

  aie.device(npu2) @devB {
    %tile_b = aie.tile(0, 2)

    conduit.create @chanB {element_type = memref<128xbf16>, depth = 2 : i64,
                           fusion_group = "g0"}

    func.func private @kernelB(memref<128xbf16>)

    aie.core(%tile_b) {
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

    // Uniquely-named runtime_sequence so we can assert its symbol is
    // preserved verbatim through the move-into-devA promotion.
    aie.runtime_sequence @unique_seqB_name(%b0: memref<128xbf16>) {
    }
  }

  // Host orchestrator: a single aiex.configure @devB.  No sibling
  // aiex.configure @devA in this parent block, so the rewrite goes through
  // the in-place branch (DeviceMergeUtils.cpp:319-321).
  aie.device(npu2) {
    aie.runtime_sequence(%h0: memref<128xbf16>) {
      aiex.configure @devB {
        aiex.run @unique_seqB_name(%h0) : (memref<128xbf16>)
      }
    }
  }
}
