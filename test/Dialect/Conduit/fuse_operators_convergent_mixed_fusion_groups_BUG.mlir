//===- fuse_operators_convergent_mixed_fusion_groups_BUG.mlir -*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Track 3 — diagnostic regression for the "mixed convergent + 1:1
// fusion_groups on the same consumer" case.
//
// Setup: a single consumer device participates in BOTH a convergent
// fusion_group (two producers fan-in via "swiglu_fg") AND, on a different
// channel, an unrelated 1:1 fusion_group ("plain_fg"). Per Q2 of the
// Track 3 design (CLAUDE.md USER-LOCKED 2026-04-26), the initial landing
// of convergent merge must REJECT this composition with a clear diagnostic
// rather than silently picking one shape — composing convergent and 1:1
// merges on the same consumer is out of scope until the design has decided
// the placement / depth-promote interaction.
//
// Phase 2 (Sprint N+2) flips this fixture from XFAIL → PASS: the
// pre-scan in `--conduit-fuse-operators` emits the `expected-error`
// diagnostic on `@devConsumer` and signals pass failure before any
// per-pair iteration runs, so the composition is rejected up-front
// rather than silently picking one shape.
//
// Sibling pattern reference: fuse_operators_routing_mode_conflict_error.mlir
// (uses --verify-diagnostics + expected-error in the same way).
//===----------------------------------------------------------------------===//

// RUN: aie-opt --conduit-fuse-operators --verify-diagnostics %s

module @mixed_convergent_and_one_to_one {
  // Convergent producer 0 (fans into consumer's convergent inputs).
  aie.device(npu2) @devGate {
    %shim = aie.tile(0, 0)
    %t    = aie.tile(0, 2)
    conduit.create @ext_in_gate {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @inter_gate {element_type = memref<128xbf16>, depth = 1 : i64,
                                fusion_group = "swiglu_fg",
                                fusion_index = 0 : i32}
  }

  // Convergent producer 1.
  aie.device(npu2) @devUp {
    %shim = aie.tile(0, 0)
    %t    = aie.tile(0, 2)
    conduit.create @ext_in_up {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @inter_up {element_type = memref<128xbf16>, depth = 1 : i64,
                              fusion_group = "swiglu_fg",
                              fusion_index = 1 : i32}
  }

  // Unrelated 1:1 producer that also targets the same consumer (different
  // fusion_group entirely):
  aie.device(npu2) @devSide {
    %shim = aie.tile(0, 0)
    %t    = aie.tile(0, 2)
    conduit.create @ext_in_side {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @inter_side {element_type = memref<128xbf16>, depth = 1 : i64,
                                fusion_group = "plain_fg"}
  }

  // Consumer device participates in BOTH groups: this is the rejected
  // composition. The diagnostic must clearly name the consumer and both
  // fusion_group tags so the operator author can disambiguate.
  // expected-error@below {{conduit-fuse-operators: consumer device participates in both convergent fusion_group "swiglu_fg" and 1:1 fusion_group "plain_fg"; mixed convergent + 1:1 fusion is out of scope}}
  aie.device(npu2) @devConsumer {
    %shim = aie.tile(0, 0)
    %t    = aie.tile(0, 2)

    // Convergent inputs (matching swiglu_fg).
    conduit.create @consume_gate {element_type = memref<128xbf16>, depth = 1 : i64,
                                  fusion_group = "swiglu_fg",
                                  fusion_index = 0 : i32}
    conduit.create @consume_up {element_type = memref<128xbf16>, depth = 1 : i64,
                                fusion_group = "swiglu_fg",
                                fusion_index = 1 : i32}

    // 1:1 input from the unrelated side producer.
    conduit.create @consume_side {element_type = memref<128xbf16>, depth = 1 : i64,
                                  fusion_group = "plain_fg"}

    conduit.create @ext_out {element_type = memref<128xbf16>, depth = 2 : i64}
  }
}
