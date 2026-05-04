// RUN: aie-opt --aie-combine-device="same-tile=true" %s | FileCheck %s
//
// Lit-gap-audit findings 2.2 + 3.3 — bidirectional sibling scan in
// `findSiblingConfigure*` (DeviceMergeUtils.cpp).
//
// Source path under test:
//   `findSiblingConfigureBefore` scans the parent block for a sibling
//   `aiex.configure @<devA>` BEFORE confB; if none is found,
//   `findSiblingConfigureAfter` scans for the FIRST sibling AFTER confB.
//   Together they make the fold-into-confA collapse fire regardless of
//   confA/confB textual order in the host orchestrator's parent block.
//
// Before this fix, only the BEFORE-conf scan existed.  When the host
// orchestrator emitted `aiex.configure @devB` BEFORE `aiex.configure
// @devA` in the same parent block (perfectly valid IR — see Llama
// `llama_npu.py` host loop), the scan returned null and confB took the
// in-place rewrite branch.  The result was TWO surviving
// `aiex.configure @devA` blocks in the same parent runtime_sequence — one
// from the original confA, one from the in-place-rewritten confB —
// verifier-clean but TWO LoadPDI cycles per launch instead of one.  The
// runtime semantics was "device A configured/launched twice in sequence",
// which is functionally correct but defeats the intent of fusion.
//
// POSITIVE TEST — asserts the fold fires for the reversed configure order.
// After the fix, the rewrite produces a SINGLE `aiex.configure @devA`
// block: confB's non-RunOp body ops are folded into bodyA at the FRONT
// (preserving the original B-setup-then-A-setup ordering), and confA's
// RunOp is rewritten to take the concatenation of A's + B's args
// (`runA.getArgs() ++ runB.getArgs()` — order matches the merged
// `@seqA`'s block-arg layout, which Phase 2 builds as A's args followed by
// B's args regardless of textual configure order).

// CHECK-LABEL: module @lit_audit_2_2_configure_order_reversed

// Single named device after merge:
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device(npu2) @devB

// Single surviving runtime_sequence (devA's @seqA — seqB was spliced into
// it via the simple-merge path).
// CHECK:       aie.runtime_sequence @seqA

// Single surviving aiex.configure block targeting @devA — confB folded in.
// The merged run takes the concatenation of A's + B's args (2 total),
// in (runA.args, runB.args) order, calling the surviving @seqA.
// CHECK:       aiex.configure @devA
// CHECK:       aiex.run @seqA(%{{[^,]+}}, %{{[^,]+}})
// CHECK-SAME:    : (memref<128xbf16>, memref<128xbf16>)

// No second `aiex.configure` block of any kind survives — confB has been
// erased as part of the fold.
// CHECK-NOT:   aiex.configure

module @lit_audit_2_2_configure_order_reversed {
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

    aie.runtime_sequence @seqA(%a0: memref<128xbf16>) {
    }
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

    aie.runtime_sequence @seqB(%b0: memref<128xbf16>) {
    }
  }

  // Host orchestrator: REVERSED ordering — confB BEFORE confA in the same
  // parent block.  `findSiblingConfigureBefore(confB)` returns null (no
  // sibling @devA precedes it), then `findSiblingConfigureAfter(confB)`
  // finds confA.  The fold collapses into a single `aiex.configure @devA`
  // at confA's textual position.
  aie.device(npu2) {
    aie.runtime_sequence(%h0: memref<128xbf16>, %h1: memref<128xbf16>) {
      aiex.configure @devB {
        aiex.run @seqB(%h1) : (memref<128xbf16>)
      }
      aiex.configure @devA {
        aiex.run @seqA(%h0) : (memref<128xbf16>)
      }
    }
  }
}
