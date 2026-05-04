// RUN: aie-opt --aie-combine-device="same-tile=true" %s | FileCheck %s
//
// Task #82 lit pin — runtime_sequence arg_index reprojection under
// `--aie-combine-device`.
//
// Bug shape (pre-fix): `mergeRuntimeSequencesSimple` (DeviceMergeUtils.cpp:84)
// appends devB's runtime_sequence block args to devA via `addArgument()` and
// clones devB's body via `IRMapping`.  IRMapping correctly remaps SSA
// operand uses of the block args, but `clone()` deep-copies attributes
// verbatim — leaving the integer `arg_index` attribute on cloned
// `conduit.put_memref{,_async}` / `get_memref{,_async}` ops referring to
// devB's LOCAL (pre-merge) arg space.  In the merged sequence, devA owns
// the first |devA.args| slots, so devB's local arg 0 collides with devA's
// arg 0 — the cloned op silently binds to the wrong host buffer.
//
// Surfaces in the swiglu hybrid pipeline (Task #82): two distinct
// producer-side host inputs (@ext_in_gate + @ext_in_up) collapse to
// `arg_index = 0` in the merged @gate_seq, even though they should bind to
// distinct host slots.  Hidden by symmetric input data (gate=up) in the
// existing fixture; would silently corrupt any non-symmetric K=2
// convergent fusion.
//
// Test setup: two single-device runtime_sequences on the same tile(0,2),
// each with TWO args carrying conduit.put_memref / get_memref ops with
// arg_index = 0, 1 (devB-local).  After --aie-combine-device, the merged
// sequence has FOUR args, and devB's clones must have arg_index reprojected
// to 2, 3 (= 2 + 0, 2 + 1).  Pre-fix: arg_index would still be 0, 1 —
// aliasing devA's slots.
//
// Why this is the right pin location:
//   * Pure dialect-level: exercises only --aie-combine-device, isolating
//     the bug from downstream fuse-operators / fuse-core-bodies / Pass C.
//   * Catches the bug for ANY caller of mergeRuntimeSequencesSimple
//     (--aie-combine-device standalone, --conduit-fuse-core-bodies, etc.).
//   * No HW dependency — runs in the standard dialect lit suite.
//
// Sibling fixtures:
//   * combine_device.mlir — basic combine test without runtime_seq
//     (predates this bug class; doesn't exercise arg_index).
//   * lit_audit_2_4_arg_index_post_merge.mlir — flagged this exact bug as
//     a TODO in 2026-Q1 ("confirm whether `--conduit-fuse-operators`
//     Step 8c re-projects arg_index; if not, file as a source bug").
//     That fixture passes by coincidence (trim compacts indices back).
//     This fixture surfaces the bug DIRECTLY without trim coincidence.

// CHECK-LABEL: module @combine_runtime_seq_arg_index
//
// Single merged device:
// CHECK:       aie.device(npu2)
// CHECK-NOT:   aie.device(npu2)
//
// Merged runtime_sequence has FOUR args (devA's two + devB's two):
// CHECK:       aie.runtime_sequence
// CHECK-SAME:    %{{[^,)]+}}: memref<128xbf16>
// CHECK-SAME:    %{{[^,)]+}}: memref<128xbf16>
// CHECK-SAME:    %{{[^,)]+}}: memref<128xbf16>
// CHECK-SAME:    %{{[^,)]+}}: memref<128xbf16>
//
// devA's put_memref for @chA stays at arg_index = 0:
// CHECK:       conduit.put_memref
// CHECK-SAME:    arg_index = 0
// CHECK-SAME:    name = @chA
//
// devA's get_memref for @chB stays at arg_index = 1:
// CHECK:       conduit.get_memref
// CHECK-SAME:    arg_index = 1
// CHECK-SAME:    name = @chB
//
// devB's put_memref for @chC MUST be reprojected to arg_index = 2
// (devB-local 0 + devA's pre-merge arg count 2).
// Pre-fix this would still say `arg_index = 0` and alias chA's slot.
// CHECK:       conduit.put_memref
// CHECK-SAME:    arg_index = 2
// CHECK-SAME:    name = @chC
//
// devB's get_memref for @chD MUST be reprojected to arg_index = 3
// (devB-local 1 + devA's pre-merge arg count 2).
// Pre-fix this would still say `arg_index = 1` and alias chB's slot.
// CHECK:       conduit.get_memref
// CHECK-SAME:    arg_index = 3
// CHECK-SAME:    name = @chD

module @combine_runtime_seq_arg_index {
  aie.device(npu2) @devA {
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @chA {element_type = memref<128xbf16>, depth = 2 : i64,
                         fusion_group = "fg_combine"}
    conduit.create @chB {element_type = memref<128xbf16>, depth = 2 : i64}

    aie.runtime_sequence(%a0: memref<128xbf16>, %a1: memref<128xbf16>) {
      conduit.put_memref {name = @chA, num_elems = 128 : i64,
                          offsets = array<i64: 0>,
                          sizes = array<i64: 128>,
                          strides = array<i64: 1>,
                          arg_index = 0 : i64}
      conduit.get_memref {name = @chB, num_elems = 128 : i64,
                          offsets = array<i64: 0>,
                          sizes = array<i64: 128>,
                          strides = array<i64: 1>,
                          arg_index = 1 : i64}
    }
  }

  aie.device(npu2) @devB {
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @chC {element_type = memref<128xbf16>, depth = 2 : i64,
                         fusion_group = "fg_combine"}
    conduit.create @chD {element_type = memref<128xbf16>, depth = 2 : i64}

    aie.runtime_sequence(%b0: memref<128xbf16>, %b1: memref<128xbf16>) {
      conduit.put_memref {name = @chC, num_elems = 128 : i64,
                          offsets = array<i64: 0>,
                          sizes = array<i64: 128>,
                          strides = array<i64: 1>,
                          arg_index = 0 : i64}
      conduit.get_memref {name = @chD, num_elems = 128 : i64,
                          offsets = array<i64: 0>,
                          sizes = array<i64: 128>,
                          strides = array<i64: 1>,
                          arg_index = 1 : i64}
    }
  }
}
