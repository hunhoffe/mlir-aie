// RUN: aie-opt --conduit-fuse-core-bodies %s | FileCheck %s
//
// Lit-gap-audit finding 2.3 / 3.5 — `--conduit-fuse-core-bodies`
// host-orchestrator path.  `mergeAndUnifyDevices`
// (ConduitFuseCoreBodyPass.cpp:243-246) shares
// `rewriteHostConfigureOnDeviceMerge` with `--conduit-fuse-operators` and
// `--aie-combine-device`, but ONLY `--conduit-fuse-operators` has lit
// coverage of the host-orchestrator merge today.
//
// Source path under test:
//   - `mergeDevicesForFusion` discovers a fusion_group connection across
//     devA and devB.
//   - `mergeAndUnifyDevices` runs Phase-1 + simple Phase-2 sequence merge,
//     then calls `rewriteHostConfigureOnDeviceMerge`.
//   - Unlike `--conduit-fuse-operators`, fuse-core-bodies does NOT trim
//     dead block args after the merge, so `reconcileHostRunArgsAfterTrim`
//     is intentionally NOT invoked here.  The folded `aiex.run`'s arg
//     vector therefore stays at the naive `runA.getArgs() ++ runB.getArgs()`
//     concat — and the merged callee's arity also matches that concat
//     (no trim).
//
// What this test exercises:
//   - 2 devices, same tile (fuse-core-body uses same-tile mode), connected
//     by a fusion_group channel pair.
//   - Each device has its own runtime_sequence with a 1-arg signature.
//   - Host orchestrator: `aiex.configure @devA { aiex.run @seqA(%h0) }` then
//     `aiex.configure @devB { aiex.run @seqB(%h1) }`.
//   - After the pass: single device, single `aiex.configure @devA`, run
//     takes the 2-arg concat `(%h0, %h1)`.

// CHECK-LABEL: module @lit_audit_2_3_fuse_core_body_host_orchestrator

// Single surviving named device:
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device(npu2) @devB

// Folded host orchestrator: confB collapsed into confA (same parent block,
// confA precedes confB).
// CHECK:       aiex.configure @devA
// CHECK-NOT:   aiex.configure @devB

// The merged run takes the concatenation of A's + B's args (2 total).
// fuse-core-bodies does not trim, so the run-arity equals concat-arity.
// CHECK:       aiex.run @{{.+}}(%{{[^,]+}}, %{{[^,]+}})
// CHECK-SAME:    : (memref<128xbf16>, memref<128xbf16>)

module @lit_audit_2_3_fuse_core_body_host_orchestrator {
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
