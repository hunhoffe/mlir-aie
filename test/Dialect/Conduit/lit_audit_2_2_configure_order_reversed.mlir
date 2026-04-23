// RUN: aie-opt --aie-combine-device="same-tile=true" %s | FileCheck %s
//
// Lit-gap-audit findings 2.2 + 3.3 — `findSiblingConfigureBefore` only
// scans BEFORE confB for a sibling `aiex.configure @<devA>`.  If the host
// orchestrator emits `aiex.configure @devB` BEFORE `aiex.configure @devA`
// in the same parent block, the fold-into-confA collapse silently does NOT
// happen — confB takes the in-place rewrite branch, and confA stays.
//
// Source path under test:
//   `findSiblingConfigureBefore` (DeviceMergeUtils.cpp:175-190): scans the
//   parent block from the start, breaks at `&op == conf.getOperation()`.
//   Forward siblings are never considered.
//
// NEGATIVE TEST — documents current limitation.
// Today (no source fix): the rewrite produces TWO separate
// `aiex.configure @devA` blocks in the same parent runtime_sequence — one
// from the original confA, one from the in-place rewrite of confB.  This
// preserves the verifier (no dangling @devB ref) but does NOT collapse the
// two LoadPDI cycles into one.  The runtime semantics is "device A
// configured/launched twice in sequence", which is functionally correct
// but defeats the intent of fusion.
//
// TODO(post-fix): when `findSiblingConfigureBefore` is generalised to scan
// the entire parent block (or split into before+after passes), update the
// CHECK lines below to assert a SINGLE `aiex.configure @devA` survives and
// the run takes the concatenation of both arg lists.

// CHECK-LABEL: module @lit_audit_2_2_configure_order_reversed

// Single named device after merge:
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device(npu2) @devB

// CURRENT BEHAVIOR — both configures point at @devA but no fold happened.
// We assert there are TWO `aiex.configure @devA` blocks (one from the
// in-place-rewritten confB, one from the original confA).  The order is
// preserved: the rewritten confB appears FIRST (at its original textual
// position), then the original confA.
// CHECK:       aiex.configure @devA
// CHECK:       aiex.run
// CHECK:       aiex.configure @devA
// CHECK:       aiex.run

// No surviving @devB reference.
// CHECK-NOT:   aiex.configure @devB

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
  // sibling @devA precedes it), so confB takes the in-place rewrite branch
  // (confB.symbol → @devA).  confA is untouched.  Result: two
  // `aiex.configure @devA` blocks survive — fold did NOT happen.
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
