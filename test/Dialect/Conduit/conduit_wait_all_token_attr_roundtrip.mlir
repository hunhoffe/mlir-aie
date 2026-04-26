//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --split-input-file %s | FileCheck %s

// Basic Path C (2026-04-25, Task #18) parse/print roundtrip for the
// `token` attribute on conduit.wait_all.  The attribute defaults to
// `true` (DefaultValuedOptionalAttr<BoolAttr, "true">), so the absent /
// explicit-true / explicit-false cases must all roundtrip identically.
// Explicit-true is elided by the printer (default-value elision); the
// explicit-false case must preserve the attribute on print so the
// release-marker semantic survives into Pass C lowering (where it
// selects aiex.dma_free_task vs aiex.dma_await_task).

aie.device(npu1) {
  conduit.create @c {depth = 0 : i64, element_type = memref<128xbf16>}

  // CHECK-LABEL: func.func @wait_all_token_default_absent
  // CHECK:         conduit.wait_all %{{.+}} : !conduit.dma.token
  // CHECK-NOT:     {token
  func.func @wait_all_token_default_absent() {
    %tok = conduit.put_memref_async {name = @c, num_elems = 128 : i64,
                                     offsets = array<i64: 0>,
                                     sizes = array<i64: 128>,
                                     strides = array<i64: 1>}
        : !conduit.dma.token
    conduit.wait_all %tok : !conduit.dma.token
    return
  }
}

// -----

aie.device(npu1) {
  conduit.create @c {depth = 0 : i64, element_type = memref<128xbf16>}

  // CHECK-LABEL: func.func @wait_all_token_explicit_true
  // Explicit token = true is the DEFAULT — printer elides the attr, so
  // the printed form looks identical to the absent-attr case above.
  // CHECK:         conduit.wait_all %{{.+}} : !conduit.dma.token
  // CHECK-NOT:     {token
  func.func @wait_all_token_explicit_true() {
    %tok = conduit.put_memref_async {name = @c, num_elems = 128 : i64,
                                     offsets = array<i64: 0>,
                                     sizes = array<i64: 128>,
                                     strides = array<i64: 1>}
        : !conduit.dma.token
    conduit.wait_all %tok {token = true} : !conduit.dma.token
    return
  }
}

// -----

aie.device(npu1) {
  conduit.create @c {depth = 0 : i64, element_type = memref<128xbf16>}

  // CHECK-LABEL: func.func @wait_all_token_explicit_false
  // CHECK:         conduit.wait_all %{{.+}} {token = false} : !conduit.dma.token
  func.func @wait_all_token_explicit_false() {
    %tok = conduit.put_memref_async {name = @c, num_elems = 128 : i64,
                                     offsets = array<i64: 0>,
                                     sizes = array<i64: 128>,
                                     strides = array<i64: 1>}
        : !conduit.dma.token
    conduit.wait_all %tok {token = false} : !conduit.dma.token
    return
  }
}
