//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-channels %s | FileCheck %s
// Metafix Candidate 1 (Path C async): also smoke through downstream
// shim-allocation-substitution + BD-ID assignment so legalization traps
// are caught at lit time, not first NPU contact.
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-channels --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// Path C async fuse-pass interaction pin (Task #33, design from
// path-c-test-matrix.md §1.1).
//
// Goal: when --dma-task-to-conduit emits async ops + conduit.wait_all{token}
// release boundaries, --conduit-fuse-channels MUST preserve the WaitAll{token}
// op shape, position, and SSA def-use AND must not promote token=true to
// token=false (or vice versa).  This is the per-channel coloring pin under
// async tokens (CLAUDE.md "Locked design decisions": wait_all{token=false}
// legal only when ALL operands are !conduit.dma.token).
//
// Input shape:
//   - Two MM2S channels (@ext_in_a, @ext_in_b) on the same shim tile.
//   - rt-seq has, per channel: configure + start + dma_free_task.
//   - --dma-task-to-conduit lowers each free to wait_all{token = false}
//     consuming the put_memref_async's !conduit.dma.token.
//
// Expected output (after --conduit-fuse-channels):
//   - Both put_memref_async ops survive.
//   - Both wait_all{token = false} ops survive in source-relative position.
//   - Token attr is NOT promoted (false stays false; this is the
//     release-marker semantic — flipping to true would mean "await", which
//     would block the rt-seq instead of releasing the BD).

// CHECK-LABEL: module @path_c_async_fuse_channels_token_preserve

// Both async puts survive fuse-channels (it annotates rather than rewrites
// the SSA chain; the put_memref_async ops must NOT be merged or dropped),
// and each WaitAll release marker must survive with token = false in
// source-relative position immediately after its put.  Default elision
// means the printer would drop token=true; the explicit `token = false`
// must round-trip print.
// CHECK:       conduit.put_memref_async
// CHECK-SAME:  name = @ext_in_a
// CHECK:       conduit.wait_all
// CHECK-SAME:  token = false
// CHECK:       conduit.put_memref_async
// CHECK-SAME:  name = @ext_in_b
// CHECK:       conduit.wait_all
// CHECK-SAME:  token = false

// Critical: no spurious wait_all{token=true} (await) appears as a side effect
// of the fuse pass.  Pass A only emits await for IRON dma_await_task; with
// only dma_free_task in source, every WaitAll must be a release marker.
// CHECK-NOT:   token = true

module @path_c_async_fuse_channels_token_preserve {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // depth=1 so --conduit-fuse-channels actually triggers fusion
    // (depth>1 emits "skipping S2MM fusion" and the wait_all-token
    // preservation invariant this test pins isn't reachable).
    aie.objectfifo @ext_in_a(%shim, {%tile}, 1 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @ext_in_b(%shim, {%tile}, 1 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @kernel(memref<128xbf16>)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %ina = aie.objectfifo.acquire @ext_in_a(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %ina_buf = aie.objectfifo.subview.access %ina[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @kernel(%ina_buf) : (memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_in_a(Consume, 1)

        %inb = aie.objectfifo.acquire @ext_in_b(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %inb_buf = aie.objectfifo.subview.access %inb[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @kernel(%inb_buf) : (memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_in_b(Consume, 1)
      }
      aie.end
    } {link_with = "kernel.a"}

    aie.runtime_sequence(%arg0: memref<256xbf16>) {
      // Channel A: configure + start + free (becomes WaitAll{token=false}).
      %ta = aiex.dma_configure_task_for @ext_in_a {
        aie.dma_bd(%arg0 : memref<256xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta)
      aiex.dma_free_task(%ta)

      // Channel B: configure + start + free (becomes WaitAll{token=false}).
      %tb = aiex.dma_configure_task_for @ext_in_b {
        aie.dma_bd(%arg0 : memref<256xbf16>, 128, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tb)
      aiex.dma_free_task(%tb)
    }
  }
}
