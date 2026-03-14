// RUN: aie-opt --conduit-fuse-channels --conduit-fuse-channels %s -split-input-file | FileCheck %s
// Running the pass twice verifies idempotency: every case that produces an
// annotation (a, c, f, g, h, j, l, m) must produce the same annotation on
// the second run; every case that produces no annotation (b, d, e, i, k, n)
// must remain unannotated.  The FileCheck patterns serve as the oracle.
//
// --conduit-fuse-channels: DMA channel fusion annotation tests.
//
// Theory:
//   Two conduits on the same producer tile can share one hardware DMA channel
//   when their buffer-window ops are sequentially non-overlapping in a common
//   basic block.  The pass annotates eligible conduit.create ops with:
//       fuse_mode = "static"  fused_dma_channel_group = "groupN"
//   where N is the hardware channel group.
//
//   Interval non-overlap condition (single block):
//     last_op(A) < first_op(B)  OR  last_op(B) < first_op(A)
//
//   This is the "register allocation across non-overlapping live ranges"
//   analogy applied to DMA channel assignment.
//
// Test plan:
//   (a) Two conduits on same tile, A's ops precede B's → annotated same group
//   (b) Two conduits on same tile, ops interleaved → NOT annotated (overlap)
//   (c) Three conduits on same tile, A before B before C → all annotated
//   (d) Conduits on different tiles → NOT annotated (different tile groups)
//   (e) Shim-tile conduit (row=0) is excluded regardless of ops
//   (f) Idempotency: annotation survives re-run
//   (g) Async acquire path: wait_window + release (SSA tracing through WaitWindow)
//   (h) Tier 3 put_memref ops as activity markers → fused when non-overlapping
//   (i) No ops in block: conduit.create with no acquire/release → not annotated
//   (j) DMA exhaustion use case: 4 conduits, 2 non-overlapping pairs → 2 groups
//   (k) Single conduit on tile → not annotated (no fusion benefit)
//   (l) Two interleaved pairs: a+c group0, b+d group1
//   (m) Three-way partial clique: a+c fused, b singleton → not annotated
//   (n) Three-way full clique: all pairwise overlapping → none annotated
//   (o) fuse_mode = "runtime" via scf.if (Gap A2)
//   (p) get_memref as activity marker (Gap A3)
//   (q) release_async as activity marker (Gap A5)
//   (r) MemTile producer (row=1) not excluded (Gap A6)
//   (s) Cross-block same-name: first-block-wins (Gap A8)
//
//===----------------------------------------------------------------------===//
// (a) Two conduits on same tile — A's ops precede B's — same group annotated.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_sequential
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "chan_a"
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "chan_b"

func.func @fuse_sequential() {
  // Both conduits on tile [0, 2].
  conduit.create {name = "chan_a", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  conduit.create {name = "chan_b", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}

  // chan_a ops execute, then chan_b ops — non-overlapping.
  %wa = conduit.acquire {name = "chan_a", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wa {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  %wb = conduit.acquire {name = "chan_b", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wb {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  return
}

// -----

//===----------------------------------------------------------------------===//
// (b) Two conduits on same tile — ops interleaved — NOT fused.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_fuse_interleaved
// CHECK-NOT:   fused_dma_channel_group
// CHECK-NOT:   fuse_mode

func.func @no_fuse_interleaved() {
  conduit.create {name = "chan_a", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  conduit.create {name = "chan_b", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}

  // Interleaved: acquire A, acquire B, release A, release B.
  // live(A) = [acquire_A, release_A]  live(B) = [acquire_B, release_B]
  // Intervals overlap → no fusion.
  %wa = conduit.acquire {name = "chan_a", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  %wb = conduit.acquire {name = "chan_b", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wa {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  conduit.release %wb {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  return
}

// -----

//===----------------------------------------------------------------------===//
// (c) Three conduits on same tile — c1 before c2 before c3, all sequential.
//
//     Intervals (creates at 0-2 are not counted; ops start at idx 3):
//       c1: [3,4]   c2: [5,6]   c3: [7,8]
//
//     Linear-scan (sorted by lo: c1, c2, c3 — already in order):
//       c1 → group 0  (groupEnd[0] = 4)
//       c2 → 4 < 5? YES → group 0  (groupEnd[0] = 6)
//       c3 → 6 < 7? YES → group 0  (groupEnd[0] = 8)
//
//     All three share group 0: one hardware channel suffices.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_three_sequential
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group{{.*}}name = "c1"
// CHECK-NEXT:  conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group{{.*}}name = "c2"
// CHECK-NEXT:  conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group{{.*}}name = "c3"

func.func @fuse_three_sequential() {
  conduit.create {name = "c1", capacity = 8 : i64,
                  producer_tile = array<i64: 1, 2>,
                  consumer_tiles = array<i64: 1, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "c2", capacity = 8 : i64,
                  producer_tile = array<i64: 1, 2>,
                  consumer_tiles = array<i64: 1, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "c3", capacity = 8 : i64,
                  producer_tile = array<i64: 1, 2>,
                  consumer_tiles = array<i64: 1, 5>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  %w1 = conduit.acquire {name = "c1", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %w1 {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  %w2 = conduit.acquire {name = "c2", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %w2 {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  %w3 = conduit.acquire {name = "c3", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %w3 {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  return
}

// -----

//===----------------------------------------------------------------------===//
// (d) Two conduits on DIFFERENT tiles — NOT fused (different tile groups).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_fuse_different_tiles
// CHECK-NOT:   fused_dma_channel_group
// CHECK-NOT:   fuse_mode

func.func @no_fuse_different_tiles() {
  // tile [0,2] and tile [1,2] are different — no grouping.
  conduit.create {name = "tile0_chan", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "tile1_chan", capacity = 8 : i64,
                  producer_tile = array<i64: 1, 2>,
                  consumer_tiles = array<i64: 1, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  %wa = conduit.acquire {name = "tile0_chan", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wa {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  %wb = conduit.acquire {name = "tile1_chan", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wb {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  return
}

// -----

//===----------------------------------------------------------------------===//
// (e) Shim-tile conduit (row == 0) — excluded from fusion regardless.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_fuse_shim
// CHECK-NOT:   fused_dma_channel_group
// CHECK-NOT:   fuse_mode

func.func @no_fuse_shim() {
  // Both conduits on shim tile [0,0] (row=0 → excluded).
  conduit.create {name = "shim_a", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 0>,
                  shim_consumer_tiles = array<i64: 0, 0>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "shim_b", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 0>,
                  shim_consumer_tiles = array<i64: 0, 0>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  return
}

// -----

//===----------------------------------------------------------------------===//
// (f) Idempotency: the global RUN line runs the pass twice.  This case
//     verifies that the second run produces the same annotation as the first.
//     The fused_dma_channel_group attribute is set via setAttr which silently
//     overwrites; a second run re-derives the same group from the same
//     intervals and writes the same value.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @idempotent
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group{{[0-9]+}}"{{.*}}name = "id_a"
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group{{[0-9]+}}"{{.*}}name = "id_b"

func.func @idempotent() {
  conduit.create {name = "id_a", capacity = 8 : i64,
                  producer_tile = array<i64: 2, 2>,
                  consumer_tiles = array<i64: 2, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "id_b", capacity = 8 : i64,
                  producer_tile = array<i64: 2, 2>,
                  consumer_tiles = array<i64: 2, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  %wa = conduit.acquire {name = "id_a", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wa {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  %wb = conduit.acquire {name = "id_b", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wb {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  return
}

// -----

//===----------------------------------------------------------------------===//
// (g) Async acquire path: acquire_async → wait_window → release.
//
//     conduit.release takes the window SSA value from conduit.wait_window.
//     getConduitOpName must trace through wait_window to recover the channel
//     name.  This tests the getWindowChannelName(WaitWindow) code path.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_async_path
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group{{.*}}name = "async_a"
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group{{.*}}name = "async_b"

func.func @fuse_async_path() {
  conduit.create {name = "async_a", capacity = 8 : i64,
                  producer_tile = array<i64: 3, 2>,
                  consumer_tiles = array<i64: 3, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "async_b", capacity = 8 : i64,
                  producer_tile = array<i64: 3, 2>,
                  consumer_tiles = array<i64: 3, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  // Async acquire path for async_a — non-overlapping with async_b below.
  %tok_a = conduit.acquire_async {name = "async_a", count = 1 : i64}
               : !conduit.window.token
  %win_a = conduit.wait_window %tok_a for "async_a"
               : !conduit.window.token -> !conduit.window<memref<8xi32>>
  // conduit.release traces win_a → wait_window → "async_a"
  conduit.release %win_a {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  // Sync acquire path for async_b — begins after async_a is fully released.
  %win_b = conduit.acquire {name = "async_b", count = 1 : i64, port = "Consume"}
               : !conduit.window<memref<8xi32>>
  conduit.release %win_b {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  return
}

// -----

//===----------------------------------------------------------------------===//
// (h) Tier 3 put_memref ops as activity markers.
//
//     conduit.put_memref carries an explicit 'name' attribute; it should
//     extend the live interval for its channel just as acquire/release does.
//     Two conduits with non-overlapping put_memref sequences are fuseable.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_tier3_put_memref
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group{{.*}}name = "dma_a"
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group{{.*}}name = "dma_b"

func.func @fuse_tier3_put_memref() {
  conduit.create {name = "dma_a", capacity = 8 : i64,
                  producer_tile = array<i64: 4, 2>,
                  consumer_tiles = array<i64: 4, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "dma_b", capacity = 8 : i64,
                  producer_tile = array<i64: 4, 2>,
                  consumer_tiles = array<i64: 4, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  // dma_a DMA transfer completes before dma_b starts — non-overlapping.
  conduit.put_memref {name = "dma_a", num_elems = 8 : i64,
                      offsets = array<i64: 0>, sizes = array<i64: 8>,
                      strides = array<i64: 1>}
  conduit.put_memref {name = "dma_b", num_elems = 8 : i64,
                      offsets = array<i64: 0>, sizes = array<i64: 8>,
                      strides = array<i64: 1>}

  return
}

// -----

//===----------------------------------------------------------------------===//
// (i) Conduit with no ops in any block — single conduit with no
//     acquire/release.  The pass finds no intervals and does not annotate.
//     (A conduit with activity and one without cannot be fused: the lone
//     create has no interval so it does not appear in blockConduits at all.)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_annotate_no_ops
// CHECK-NOT:   fused_dma_channel_group
// CHECK-NOT:   fuse_mode

func.func @no_annotate_no_ops() {
  // Two conduits on the same tile, but neither has any acquire/release ops.
  // No intervals are found → no blockConduits entries → no annotation.
  conduit.create {name = "unused_a", capacity = 8 : i64,
                  producer_tile = array<i64: 5, 2>,
                  consumer_tiles = array<i64: 5, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "unused_b", capacity = 8 : i64,
                  producer_tile = array<i64: 5, 2>,
                  consumer_tiles = array<i64: 5, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  return
}

// -----

//===----------------------------------------------------------------------===//
// (j) Four conduits on one tile, all strictly sequential.
//
//     Intervals (creates at 0-3 not counted; ops start at idx 4):
//       p: [4,5]   q: [6,7]   r: [8,9]   s: [10,11]
//
//     Linear-scan (sorted by lo: p, q, r, s — already in order):
//       p → group 0  (groupEnd[0] = 5)
//       q → 5 < 6? YES → group 0  (groupEnd[0] = 7)
//       r → 7 < 8? YES → group 0  (groupEnd[0] = 9)
//       s → 9 < 10? YES → group 0  (groupEnd[0] = 11)
//
//     All four share group 0: a single hardware channel serves all four,
//     reprogrammed between each use.  With 2 MM2S channels on AIE2 this tile
//     would otherwise fail; one channel now covers all four transfers.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_four_sequential
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "p"
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "q"
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "r"
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "s"

func.func @fuse_four_sequential() {
  conduit.create {name = "p", capacity = 8 : i64,
                  producer_tile = array<i64: 6, 2>,
                  consumer_tiles = array<i64: 6, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "q", capacity = 8 : i64,
                  producer_tile = array<i64: 6, 2>,
                  consumer_tiles = array<i64: 6, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "r", capacity = 8 : i64,
                  producer_tile = array<i64: 6, 2>,
                  consumer_tiles = array<i64: 6, 5>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "s", capacity = 8 : i64,
                  producer_tile = array<i64: 6, 2>,
                  consumer_tiles = array<i64: 6, 6>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  %wp = conduit.acquire {name = "p", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wp {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  %wq = conduit.acquire {name = "q", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wq {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  %wr = conduit.acquire {name = "r", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wr {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  %ws = conduit.acquire {name = "s", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %ws {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  return
}

// -----

//===----------------------------------------------------------------------===//
// (k) Single conduit on its tile — no fusion possible, no annotation emitted.
//     A tile group with only one member is skipped entirely (size < 2 guard).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_annotate_singleton
// CHECK-NOT:   fused_dma_channel_group
// CHECK-NOT:   fuse_mode

func.func @no_annotate_singleton() {
  // Only one conduit on tile [7, 2] — no peer to fuse with.
  conduit.create {name = "solo", capacity = 8 : i64,
                  producer_tile = array<i64: 7, 2>,
                  consumer_tiles = array<i64: 7, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  %w = conduit.acquire {name = "solo", count = 1 : i64, port = "Consume"}
          : !conduit.window<memref<8xi32>>
  conduit.release %w {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  return
}

// -----

//===----------------------------------------------------------------------===//
// (l) Two interleaved pairs on the same tile.
//
//     This is the key DMA-exhaustion scenario: 4 conduits, 2 MM2S channels.
//     Within each pair the windows overlap, so the pair cannot share one
//     channel.  But the two pairs execute sequentially, so each pair member
//     can reuse a channel released by the matching member of the first pair.
//
//     Program order (creates not counted by interval computation):
//       idx 4: acquire a   idx 5: acquire b
//       idx 6: release a   idx 7: release b   ← pair 1 done
//       idx 8: acquire c   idx 9: acquire d
//       idx 10: release c  idx 11: release d  ← pair 2 done
//
//     Intervals:   a=[4,6]  b=[5,7]  c=[8,10]  d=[9,11]
//
//     Linear-scan (sorted by lo: a, b, c, d):
//       a → group 0  (groupEnd[0] = 6)
//       b → 6 < 5? NO  → group 1  (groupEnd[1] = 7)
//       c → 6 < 8? YES → group 0  (groupEnd[0] = 10)
//       d → 10 < 9? NO. 7 < 9? YES → group 1  (groupEnd[1] = 11)
//
//     Result: a+c share group 0; b+d share group 1.
//     4 conduits → 2 hardware channels.  DMA exhaustion avoided.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @two_interleaved_pairs
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "a"
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group1"{{.*}}name = "b"
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "c"
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group1"{{.*}}name = "d"

func.func @two_interleaved_pairs() {
  conduit.create {name = "a", capacity = 8 : i64,
                  producer_tile = array<i64: 8, 2>,
                  consumer_tiles = array<i64: 8, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "b", capacity = 8 : i64,
                  producer_tile = array<i64: 8, 2>,
                  consumer_tiles = array<i64: 8, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "c", capacity = 8 : i64,
                  producer_tile = array<i64: 8, 2>,
                  consumer_tiles = array<i64: 8, 5>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "d", capacity = 8 : i64,
                  producer_tile = array<i64: 8, 2>,
                  consumer_tiles = array<i64: 8, 6>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  // Pair 1: a and b windows overlap — cannot share a channel with each other.
  %wa = conduit.acquire {name = "a", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  %wb = conduit.acquire {name = "b", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wa {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  conduit.release %wb {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  // Pair 2: c and d windows overlap — cannot share a channel with each other.
  // But pair 2 starts after pair 1 ends, so a+c and b+d are non-overlapping.
  %wc = conduit.acquire {name = "c", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  %wd = conduit.acquire {name = "d", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wc {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  conduit.release %wd {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  return
}

// -----

//===----------------------------------------------------------------------===//
// (m) Three-way partial clique: A∩B overlap, B∩C overlap, A∩C disjoint.
//
//     Program order (creates not counted; indices after 3 creates):
//       idx 3: acquire a    idx 4: acquire b
//       idx 5: release a    idx 6: acquire c
//       idx 7: release b    idx 8: release c
//
//     Intervals: a=[3,5]  b=[4,7]  c=[6,8]
//
//     Linear-scan (sorted by lo: a, b, c):
//       a → group 0  (groupEnd[0] = 5)
//       b → 5 < 4? NO  → group 1  (groupEnd[1] = 7)
//       c → 5 < 6? YES → group 0  (groupEnd[0] = 8)
//
//     groupCount: {0:2, 1:1}
//     a+c annotated with group 0; b is a singleton → NOT annotated.
//
//     This tests mixed output: on the same tile, some conduits are annotated
//     and some are not.  B has no compatible partner; the pass correctly
//     suppresses its annotation rather than creating a singleton group.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @partial_clique
// a and c fused; b has no partner so its group is a singleton → not annotated.
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "a"
// The b create must not have fused_dma_channel_group.  MLIR prints attrs
// alphabetically: a group annotation would appear between element_type and
// name.  The literal transition (no wildcard between them) fails if the
// attribute is present.  FileCheck substring-matches, so no leading {{.*}}.
// CHECK:       element_type = memref<8xi32>, name = "b"
// CHECK:       conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "c"

func.func @partial_clique() {
  conduit.create {name = "a", capacity = 8 : i64,
                  producer_tile = array<i64: 9, 2>,
                  consumer_tiles = array<i64: 9, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "b", capacity = 8 : i64,
                  producer_tile = array<i64: 9, 2>,
                  consumer_tiles = array<i64: 9, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "c", capacity = 8 : i64,
                  producer_tile = array<i64: 9, 2>,
                  consumer_tiles = array<i64: 9, 5>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  // a∩b overlap (a not yet released when b is acquired).
  %wa = conduit.acquire {name = "a", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  %wb = conduit.acquire {name = "b", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wa {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  // b∩c overlap (b not yet released when c is acquired).
  %wc = conduit.acquire {name = "c", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wb {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  // a and c are disjoint: a ended at release_a, c started at acquire_c later.
  conduit.release %wc {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  return
}

// -----

//===----------------------------------------------------------------------===//
// (n) Three-way full clique: all three pairwise overlapping.
//
//     Program order:
//       idx 3: acquire a   idx 4: acquire b   idx 5: acquire c
//       idx 6: release a   idx 7: release b   idx 8: release c
//
//     Intervals: a=[3,6]  b=[4,7]  c=[5,8]
//
//     Linear-scan (sorted by lo: a, b, c):
//       a → group 0  (groupEnd[0] = 6)
//       b → 6 < 4? NO  → group 1  (groupEnd[1] = 7)
//       c → 6 < 5? NO. 7 < 5? NO  → group 2  (groupEnd[2] = 8)
//
//     groupCount: {0:1, 1:1, 2:1} — all singletons.
//     No conduits are annotated.
//
//     This is the "pass correctly does nothing" case: 3 mutually overlapping
//     conduits cannot share any DMA channel.  The interval graph is K₃
//     (complete graph on 3 nodes), chromatic number = 3 = clique number.
//     All groups are size 1 → no annotation emitted.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @full_clique
// CHECK-NOT:   fused_dma_channel_group
// CHECK-NOT:   fuse_mode

func.func @full_clique() {
  conduit.create {name = "a", capacity = 8 : i64,
                  producer_tile = array<i64: 10, 2>,
                  consumer_tiles = array<i64: 10, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "b", capacity = 8 : i64,
                  producer_tile = array<i64: 10, 2>,
                  consumer_tiles = array<i64: 10, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "c", capacity = 8 : i64,
                  producer_tile = array<i64: 10, 2>,
                  consumer_tiles = array<i64: 10, 5>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  // All three acquired before any released — all intervals mutually overlap.
  %wa = conduit.acquire {name = "a", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  %wb = conduit.acquire {name = "b", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  %wc = conduit.acquire {name = "c", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wa {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  conduit.release %wb {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  conduit.release %wc {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  return
}

// -----

//===----------------------------------------------------------------------===//
// (o) fuse_mode = "runtime" via scf.if (Gap A2).
//
//     Two conduits on tile [11, 2] with sequential acquire/release inside an
//     scf.if block (no else).  Because the ops are inside a conditional block,
//     the pass cannot statically guarantee non-overlap in all executions.
//     Both conduits receive fuse_mode = "runtime" to signal that the runtime
//     must arbitrate channel sharing.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_runtime_mode
// CHECK: conduit.create {{{.*}}fuse_mode = "runtime"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "if_a"
// CHECK: conduit.create {{{.*}}fuse_mode = "runtime"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "if_b"

func.func @fuse_runtime_mode(%cond: i1) {
  conduit.create {name = "if_a", capacity = 8 : i64,
                  producer_tile = array<i64: 11, 2>,
                  consumer_tiles = array<i64: 11, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "if_b", capacity = 8 : i64,
                  producer_tile = array<i64: 11, 2>,
                  consumer_tiles = array<i64: 11, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  scf.if %cond {
    %wa = conduit.acquire {name = "if_a", count = 1 : i64, port = "Consume"}
             : !conduit.window<memref<8xi32>>
    conduit.release %wa {count = 1 : i64, port = "Consume"}
        : !conduit.window<memref<8xi32>>
    %wb = conduit.acquire {name = "if_b", count = 1 : i64, port = "Consume"}
             : !conduit.window<memref<8xi32>>
    conduit.release %wb {count = 1 : i64, port = "Consume"}
        : !conduit.window<memref<8xi32>>
  }
  return
}

// -----

//===----------------------------------------------------------------------===//
// (p) get_memref as activity marker (Gap A3).
//
//     Two conduits on tile [12, 2] with sequential conduit.get_memref ops.
//     get_memref carries an explicit 'name' attribute and should extend the
//     live interval for its channel just as acquire/release does.
//     Both conduits are non-overlapping → annotated fuse_mode = "static".
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_get_memref
// CHECK: conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "get_a"
// CHECK: conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "get_b"

func.func @fuse_get_memref() {
  conduit.create {name = "get_a", capacity = 8 : i64,
                  producer_tile = array<i64: 12, 2>,
                  consumer_tiles = array<i64: 12, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "get_b", capacity = 8 : i64,
                  producer_tile = array<i64: 12, 2>,
                  consumer_tiles = array<i64: 12, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  conduit.get_memref {name = "get_a", num_elems = 8 : i64,
                      offsets = array<i64: 0>, sizes = array<i64: 8>,
                      strides = array<i64: 1>}
  conduit.get_memref {name = "get_b", num_elems = 8 : i64,
                      offsets = array<i64: 0>, sizes = array<i64: 8>,
                      strides = array<i64: 1>}
  return
}

// -----

//===----------------------------------------------------------------------===//
// (q) release_async as activity marker (Gap A5).
//
//     Two conduits on tile [13, 2].  Conduit rel_a uses release_async as its
//     sole activity op.  Conduit rel_b uses a plain acquire + release afterward.
//     Both are in the same block, non-overlapping.
//     Both are annotated fuse_mode = "static", same group.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_release_async
// CHECK: conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "rel_a"
// CHECK: conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "rel_b"

func.func @fuse_release_async() {
  conduit.create {name = "rel_a", capacity = 8 : i64,
                  producer_tile = array<i64: 13, 2>,
                  consumer_tiles = array<i64: 13, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "rel_b", capacity = 8 : i64,
                  producer_tile = array<i64: 13, 2>,
                  consumer_tiles = array<i64: 13, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  // release_async marks the end of rel_a's interval.
  %tok_a = conduit.release_async {name = "rel_a", count = 1 : i64, port = "Produce"}
               : !conduit.window.token

  // rel_b starts after rel_a's release_async — non-overlapping.
  %wb = conduit.acquire {name = "rel_b", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wb {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  return
}

// -----

//===----------------------------------------------------------------------===//
// (r) MemTile producer (row = 1) not excluded (Gap A6).
//
//     Two conduits on MemTile [0, 1] (row=1).  Only row=0 (shim) is excluded
//     from fusion analysis.  MemTile tiles (row=1) are eligible.
//     Sequential ops → both annotated fuse_mode = "static", same group.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_memtile_producer
// CHECK: conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "mt_a"
// CHECK: conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"{{.*}}name = "mt_b"

func.func @fuse_memtile_producer() {
  // MemTile tiles (row=1) are NOT excluded from fusion analysis.
  conduit.create {name = "mt_a", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "mt_b", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  %wa = conduit.acquire {name = "mt_a", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wa {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  %wb = conduit.acquire {name = "mt_b", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wb {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  return
}

// -----

//===----------------------------------------------------------------------===//
// (s) Cross-block same-name: first-block-wins (Gap A8).
//
//     Two conduits on tile [14, 2].  The func body has cross_a then cross_b
//     (non-overlapping).  An scf.for body also has both conduits in the same
//     order — both blocks have the same non-overlapping sequential pattern,
//     so fusion outcome is the same regardless of which block the pass
//     processes first.  Verifies that multi-block presence does not corrupt
//     the fusion outcome.
//
//     Both conduits are annotated fuse_mode = "static", same group.
//     fuse_mode = "runtime" must NOT appear.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cross_block_stable
// CHECK: conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group{{.*}}name = "cross_a"
// CHECK: conduit.create {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group{{.*}}name = "cross_b"
// Both conduits in same group regardless of which block wins.
// CHECK-NOT: fuse_mode = "runtime"

func.func @cross_block_stable() {
  conduit.create {name = "cross_a", capacity = 8 : i64,
                  producer_tile = array<i64: 14, 2>,
                  consumer_tiles = array<i64: 14, 3>,
                  element_type = memref<8xi32>, depth = 1 : i64}
  conduit.create {name = "cross_b", capacity = 8 : i64,
                  producer_tile = array<i64: 14, 2>,
                  consumer_tiles = array<i64: 14, 4>,
                  element_type = memref<8xi32>, depth = 1 : i64}

  // Outer block: cross_a then cross_b (non-overlapping).
  %wa = conduit.acquire {name = "cross_a", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wa {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  %wb = conduit.acquire {name = "cross_b", count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<8xi32>>
  conduit.release %wb {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>

  // scf.for body: same order. Both blocks have non-overlapping sequential pattern,
  // so fusion outcome is the same regardless of which block the pass processes first.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %wa2 = conduit.acquire {name = "cross_a", count = 1 : i64, port = "Consume"}
              : !conduit.window<memref<8xi32>>
    conduit.release %wa2 {count = 1 : i64, port = "Consume"}
        : !conduit.window<memref<8xi32>>
    %wb2 = conduit.acquire {name = "cross_b", count = 1 : i64, port = "Consume"}
              : !conduit.window<memref<8xi32>>
    conduit.release %wb2 {count = 1 : i64, port = "Consume"}
        : !conduit.window<memref<8xi32>>
  }
  return
}
