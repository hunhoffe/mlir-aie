//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.

// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-core-bodies %s | FileCheck %s
// Metafix Candidate 1: also smoke through full Pass C + downstream
// legalization so any per-channel BD-pool exhaustion or dialect-verifier
// failure surfaces here, not on first NPU contact.
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-core-bodies --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s

// Path C async fuse-pass interaction pin (Task #33, design from
// path-c-test-matrix.md §1.3).
//
// Goal: --conduit-fuse-core-bodies operates INSIDE cores (it merges per-tile
// loop bodies).  conduit.wait_all{token} ops live in the runtime_sequence
// (host orchestrator scope), which is OUTSIDE cores.  Per the Path C design
// §5, the expected interaction between core-body fusion and rt-seq WaitAlls
// is "none" — but the absence is itself the invariant we pin.
//
// This is a regression-pin against accidental rt-seq walking by the
// core-body fusion pass.  The release-marker semantic (`token = false`) is a
// release boundary for the per-channel BD pool; if core-body fusion
// reordered, dropped, or tag-stripped the WaitAll, the per-channel coloring
// would silently break.
//
// Input shape:
//   - Two aie.core ops on the same tile (0,2) connected by an intermediate
//     conduit.create channel (canonical core-body fusion shape, mirrors
//     fuse_core_bodies.mlir).
//   - External I/O via aie.objectfifo so --objectfifo-to-conduit injects
//     producer_tile / consumer_tiles metadata that Pass C's allocator needs
//     (the missing-tile-metadata gap is what made the prior raw-conduit.create
//     formulation of this fixture trip "SubviewAccess could not be resolved
//     to an allocated aie.buffer" on the second RUN line).
//   - rt-seq has aiex.dma_configure_task_for + dma_start_task + dma_free_task
//     for @input MM2S (becomes wait_all{token=false} release) AND
//     dma_configure_task_for + dma_start_task + dma_await_task for @output
//     S2MM (becomes wait_all{token=true elided} await).
//
// Expected output (after --conduit-fuse-core-bodies):
//   - Two cores → one fused core (the actual job of the pass), with
//     link_files merged.
//   - @intermediate erased (replaced by L1 memref.alloca).
//   - rt-seq put_memref_async / get_memref_async / wait_all ops survive
//     UNCHANGED in count, position, and token attr.

// CHECK-LABEL: module @path_c_async_fuse_corebody_passthrough

// External I/O survives; intermediate erased.
// CHECK-NOT:   conduit.create @intermediate
// CHECK-DAG:   conduit.create @input
// CHECK-DAG:   conduit.create @output

// Only one aie.core remains on the fused tile, with merged link_files.
// CHECK:       aie.core
// CHECK:       link_files = ["producer.o", "consumer.o"]
// CHECK-NOT:   aie.core

// rt-seq put_memref_async + WaitAll{token=false} (release marker for @input
// MM2S) survives untouched.  Default elision means token=true would print
// bare; the explicit `token = false` must round-trip print.
// CHECK:       conduit.put_memref_async
// CHECK-SAME:  name = @input
// CHECK:       conduit.wait_all
// CHECK-SAME:  token = false

// rt-seq get_memref_async + WaitAll (await on @output S2MM, token=true elided
// to default) survives untouched.  Trailing CHECK-NOT pins that no spurious
// token=false sneaks onto the await.
// CHECK:       conduit.get_memref_async
// CHECK-SAME:  name = @output
// CHECK:       conduit.wait_all
// CHECK-NOT:   token = false

module @path_c_async_fuse_corebody_passthrough {
  aie.device(npu2) {
    %shim = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Intermediate stays raw conduit.create (no tile metadata) — the canonical
    // shape fuse_core_bodies.mlir uses for an on-tile L1-allocable channel
    // that core-body fusion will erase.
    conduit.create @intermediate {element_type = memref<128xbf16>, depth = 2 : i64}

    // External I/O via aie.objectfifo so --objectfifo-to-conduit injects
    // producer_tile / consumer_tiles metadata that Pass C needs.
    aie.objectfifo @input(%shim, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @output(%tile_0_2, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @produce_kernel(memref<128xbf16>, memref<128xbf16>)
    func.func private @consume_kernel(memref<128xbf16>, memref<128xbf16>)

    // Core A on tile (0,2): producer — reads @input, writes @intermediate.
    aie.core(%tile_0_2) {
      %in_win = aie.objectfifo.acquire @input(Consume, 1)
                    : !aie.objectfifosubview<memref<128xbf16>>
      %in_buf = aie.objectfifo.subview.access %in_win[0]
                    : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
      %inter_win = conduit.acquire {name = @intermediate, count = 1 : i64,
                                    port = #conduit.port<Produce>}
                       : !conduit.window<memref<128xbf16>>
      %inter_buf = conduit.subview_access %inter_win {index = 0 : i64}
                       : !conduit.window<memref<128xbf16>> -> memref<128xbf16>
      func.call @produce_kernel(%in_buf, %inter_buf)
          : (memref<128xbf16>, memref<128xbf16>) -> ()
      conduit.release %inter_win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<128xbf16>>
      aie.objectfifo.release @input(Consume, 1)
      aie.end
    } {link_with = "producer.o"}

    // Core B on tile (0,2): consumer — reads @intermediate, writes @output.
    aie.core(%tile_0_2) {
      %inter_win = conduit.acquire {name = @intermediate, count = 1 : i64,
                                    port = #conduit.port<Consume>}
                       : !conduit.window<memref<128xbf16>>
      %inter_buf = conduit.subview_access %inter_win {index = 0 : i64}
                       : !conduit.window<memref<128xbf16>> -> memref<128xbf16>
      %out_win = aie.objectfifo.acquire @output(Produce, 1)
                     : !aie.objectfifosubview<memref<128xbf16>>
      %out_buf = aie.objectfifo.subview.access %out_win[0]
                     : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
      func.call @consume_kernel(%inter_buf, %out_buf)
          : (memref<128xbf16>, memref<128xbf16>) -> ()
      aie.objectfifo.release @output(Produce, 1)
      conduit.release %inter_win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xbf16>>
      aie.end
    } {link_with = "consumer.o"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
      // MM2S: configure + start + free (becomes wait_all{token=false} release).
      %tin = aiex.dma_configure_task_for @input {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tin)
      aiex.dma_free_task(%tin)

      // S2MM: configure + start + await (becomes wait_all token=true default).
      %tout = aiex.dma_configure_task_for @output {
        aie.dma_bd(%arg1 : memref<128xbf16>, 0, 128) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%tout)
      aiex.dma_await_task(%tout)
    }
  }
}
