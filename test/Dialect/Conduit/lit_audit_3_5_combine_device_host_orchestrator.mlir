// RUN: aie-opt --aie-combine-device="same-tile=true" %s | FileCheck %s
//
// Lit-gap-audit finding 3.5 — `--aie-combine-device` host-orchestrator
// path.  This pass shares `rewriteHostConfigureOnDeviceMerge` with
// `--conduit-fuse-operators` and `--conduit-fuse-core-bodies`, but only
// `--conduit-fuse-operators` had host-orchestrator lit coverage prior to
// this test (`fuse_operators_host_orchestrator{,_with_subviews}.mlir`).
//
// Source path under test:
//   `ConduitCombineDevicePass.cpp:163-185`:
//     1. Phase 1 sequence merge (`mergeRuntimeSequencesSimple`).
//     2. `rewriteHostConfigureOnDeviceMerge` retargets confB to devA and
//        folds into a sibling confA if one exists (which it does here).
//     3. devB->erase().
//
//   Like `--conduit-fuse-core-bodies`, combine-device does NOT trim dead
//   block args after the merge — so `reconcileHostRunArgsAfterTrim` is
//   intentionally NOT invoked.  The folded `aiex.run`'s arg vector stays
//   at the naive `runA.getArgs() ++ runB.getArgs()` concat, and that
//   matches the merged callee's arity (no trim).
//
// What this test exercises:
//   - 2 devices @devA / @devB, same tile (same-tile mode), connected by a
//     fusion_group channel pair.  combine-device does NOT require
//     fusion_group to fire (it merges all sibling devices), but the
//     fusion_group is included so the test resembles realistic IR.
//   - Each device has its own runtime_sequence with a 1-arg signature.
//   - Host orchestrator: confA (single run) precedes confB (single run).
//   - After the pass: single device, single `aiex.configure @devA`, run
//     takes the 2-arg concat `(%h0, %h1)`.

// CHECK-LABEL: module @lit_audit_3_5_combine_device_host_orchestrator

// Single surviving named device:
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device(npu2) @devB

// Folded host orchestrator: confB collapsed into confA.
// CHECK:       aiex.configure @devA
// CHECK-NOT:   aiex.configure @devB

// The merged run takes the concatenation of A's + B's args (2 total).
// combine-device does not trim, so run-arity equals concat-arity.
// CHECK:       aiex.run @{{.+}}(%{{[^,]+}}, %{{[^,]+}})
// CHECK-SAME:    : (memref<128xbf16>, memref<128xbf16>)

module @lit_audit_3_5_combine_device_host_orchestrator {
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

  // Host orchestrator: confA precedes confB, so confB folds into confA.
  aie.device(npu2) {
    aie.runtime_sequence(%h0: memref<128xbf16>, %h1: memref<128xbf16>) {
      aiex.configure @devA {
        aiex.run @seqA(%h0) : (memref<128xbf16>)
      }
      aiex.configure @devB {
        aiex.run @seqB(%h1) : (memref<128xbf16>)
      }
    }
  }
}
