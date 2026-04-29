// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-canonicalize-channel-puts --conduit-depth-promote --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-canonicalize-channel-puts --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// Pass C compute-tile loop-unroll-collapse pin (formerly the
// `..._overlong_bd_chain_BUG.mlir` regression pin; flipped 2026-04-28 to
// the post-fix CHECK-line form).
//
// The original bug:
//   IRON's `for batch in range(N)` host-side Python-unroll emits N
//   structurally-identical `aiex.dma_configure_task_for` ops on a single
//   channel.  --dma-task-to-conduit round-trips them 1:1 into N
//   structurally-identical `conduit.put_memref_async` ops paired with
//   wait_all{token=true} await + wait_all{token=false} free chains.  Pass C
//   Phase 1 then tallied putCount = N and Phase 5.5 Case A's
//   `(info.putCount > 1 && info.dmaRepeat == 0)` linear-chain branch fired
//   on the compute-tile S2MM consumer, allocating N sequential aie.dma_bd
//   blocks chained linearly via aie.next_bd to aie.end.  At N >
//   targetModel.getNumBDs(...) (=16 on AIE2 compute), the upstream
//   HasValidBDs verifier rejected the result with `'aie.mem' op has more
//   than 16 blocks`.
//
// The cure (--conduit-canonicalize-channel-puts):
//   Detects the IRON-pattern N structurally-identical puts + matching
//   wait_all chains and collapses to 1 put + channel-level dma_repeat=N
//   on the conduit.create.  Pass C then sees putCount=1, dma_repeat=N and
//   emits the standard depth-many-BD circular ring on the compute tile,
//   matching upstream stateful's compute-tile rotation behavior.
//
// Geometry (small enough that N is well within compute-tile BD cap):
//   * shim(0,0) producer → compute(0,2) consumer via @chan
//   * @chan: depth=2, memref<8xi32>
//   * Consumer core: scf.for trip=N=8 acquire/release of @chan
//   * Runtime sequence: 8 IRON-pattern `aiex.dma_configure_task_for` ops
//     all using the SAME bd offset (0) and SAME shape (memref<8xi32>) —
//     the IRON `rt.fill` per-batch kernel emits structurally-identical
//     ops because the batch differentiation is encoded INSIDE the
//     `aie.dma_bd` TAP, not on the configure's offset/sizes/strides.
//   * Each configure is paired with dma_await_task + dma_free_task,
//     matching IRON's `task_group` / `finish_task_group` lowering.

// CHECK-LABEL: aie.device(npu2)

// Canon collapses the 8 identical puts → 1 put + dma_repeat=8 on the
// channel BEFORE Pass C runs.  The N=8 IRON-pattern dispatches that the
// host-side Python-unroll emitted are now folded into the channel's
// dispatch counter.

// Compute-tile S2MM ring on tile_0_2 has exactly the FIFO depth (=2)
// many aie.dma_bd blocks — NOT 8.  The chain is circular: the last BD
// next_bds back to the first.  Compute-tile cycles forever.
//
// CHECK:       aie.mem(%{{.*}}tile_0_2)
// CHECK:         aie.dma_start(S2MM,
// First BD
// CHECK:         aie.use_lock
// CHECK:         aie.dma_bd
// CHECK:         aie.use_lock
// CHECK:         aie.next_bd
// Second BD (= depth)
// CHECK:         aie.use_lock
// CHECK:         aie.dma_bd
// CHECK:         aie.use_lock
// CHECK:         aie.next_bd
// CHECK-NOT:     aie.dma_bd
// CHECK:         aie.end

// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.put_memref_async

module @path_c_compute_tile_loop_unroll_collapse {
  aie.device(npu2) {
    %shim_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%shim_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      // 8 acquires/releases — one per host dispatch.
      scf.for %i = %c0 to %c8 step %c1 {
        %0 = aie.objectfifo.acquire @chan(Consume, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        aie.objectfifo.release @chan(Consume, 1)
      }
      aie.end
    }

    // 8 IRON-pattern same-channel dispatches, each PAIRED with await + free.
    // bd offset = 0 across all 8; shape memref<8xi32>; identical attrs.
    // After --dma-task-to-conduit, this round-trips into 8 structurally-
    // identical `conduit.put_memref_async` ops on @chan with matching
    // wait_all{token=true} (await) + wait_all{token=false} (free) chains.
    // --conduit-canonicalize-channel-puts then collapses to:
    //   1 conduit.put_memref_async + 1 await + 1 free on @chan
    //   conduit.create @chan { ..., dma_repeat = 8 }
    aie.runtime_sequence(%arg0: memref<8xi32>) {
      %t0 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%arg0 : memref<8xi32>, 0, 8) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_free_task(%t0)
      %t1 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%arg0 : memref<8xi32>, 0, 8) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t1)
      %t2 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%arg0 : memref<8xi32>, 0, 8) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      aiex.dma_await_task(%t2)
      aiex.dma_free_task(%t2)
      %t3 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%arg0 : memref<8xi32>, 0, 8) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3)
      aiex.dma_await_task(%t3)
      aiex.dma_free_task(%t3)
      %t4 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%arg0 : memref<8xi32>, 0, 8) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t4)
      aiex.dma_await_task(%t4)
      aiex.dma_free_task(%t4)
      %t5 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%arg0 : memref<8xi32>, 0, 8) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t5)
      aiex.dma_await_task(%t5)
      aiex.dma_free_task(%t5)
      %t6 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%arg0 : memref<8xi32>, 0, 8) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t6)
      aiex.dma_await_task(%t6)
      aiex.dma_free_task(%t6)
      %t7 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%arg0 : memref<8xi32>, 0, 8) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t7)
      aiex.dma_await_task(%t7)
      aiex.dma_free_task(%t7)
    }
  }
}
