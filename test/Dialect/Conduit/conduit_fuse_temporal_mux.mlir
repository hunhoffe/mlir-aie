// RUN: aie-opt --conduit-fuse-channels %s -split-input-file 2>/dev/null | FileCheck %s
//
// --conduit-fuse-channels: MM2S live-interval fusion tests.
//
// Background:
// This file tests the MM2S live-interval fusion analysis in --conduit-fuse-channels.
// Sections (a), (b), (i), (j), (k), (l) that tested S2MM Temporal Multiplexing
// annotation (time_multiplex_count / conduit.create erasure) have been removed;
// that analysis was moved to Pass C (--conduit-to-dma) which infers from putCount
// automatically.
//
// Remaining sections (c)-(h) test MM2S live-interval fusion via greedy interval
// coloring, which annotates conduit.create ops with fused_dma_channel_group and
// fuse_mode when their live intervals are non-overlapping in a basic block.
//
// Test plan (MM2S live-interval fusion):
//   (c) No token dependency: two puts at same endpoint with no dep chain ->
//       both get fused_dma_channel_group (sequential in block) but no
//       time_multiplex_count.
//   (d) Partial dep: t1->t2 chained, t3 independent -> all three get fuse
//       attrs; no time_multiplex_count on any of them.
//   (e) Different endpoints: t1->t2 but different consumer_tile -> both get
//       fused_dma_channel_group (same producer tile, sequential intervals) but
//       no time_multiplex_count.
//   (f) Packet-mode excluded: routing_mode="packet" -> fused_dma_channel_group
//       annotated (same producer tile) but no time_multiplex_count.
//   (g) Depth > 1 excluded: capacity>1 / depth>1 -> NOT annotated with
//       fused_dma_channel_group (Pass C remark emitted; skipped).
//   (h) Link src excluded from temporal mux: conduit in scatter src ->
//       fused_dma_channel_group annotated but no time_multiplex_count.
//
//===----------------------------------------------------------------------===//

// -----

//===----------------------------------------------------------------------===//
// (c) No token dependency -- two puts at same endpoint, no dep chain.
//
//     Without a token edge, the pass cannot prove non-overlap for TM purposes.
//     The MM2S live-interval analysis sees sequential ops and annotates
//     fused_dma_channel_group + fuse_mode, but does NOT add time_multiplex_count.
//     Expected: neither create gains time_multiplex_count.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @no_dep_no_merge
// CHECK:       conduit.create @nd1 {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"
// CHECK-NOT:   time_multiplex_count
// CHECK:       conduit.create @nd2 {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"
// CHECK-NOT:   time_multiplex_count
func.func @no_dep_no_merge() {
  conduit.create @nd1 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 1, 1>,
                  consumer_tiles = array<i64: 1, 2>,
                  element_type = memref<4096xbf16>,
                  depth = 1 : i64}
  conduit.create @nd2 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 1, 1>,
                  consumer_tiles = array<i64: 1, 2>,
                  element_type = memref<4096xbf16>,
                  depth = 1 : i64}

  // No dep list on nd2 -- not ordered relative to nd1.
  %ta = conduit.put_memref_async {name = @nd1, num_elems = 4096 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 4096>,
            strides = array<i64: 1>} : !conduit.dma.token
  %tb = conduit.put_memref_async {name = @nd2, num_elems = 4096 : i64,
            offsets = array<i64: 4096>, sizes = array<i64: 4096>,
            strides = array<i64: 1>} : !conduit.dma.token
  conduit.wait_all %ta : !conduit.dma.token
  conduit.wait_all %tb : !conduit.dma.token
  return
}

// -----

//===----------------------------------------------------------------------===//
// (d) Partial dep: t1->t2 chained, t3 independent at the same endpoint.
//
//     t1 and t2 form a 2-chain.  t3 has no dep on either.
//     The MM2S analysis annotates all three with fused_dma_channel_group since
//     they all live on the same producer tile with sequential intervals.
//     TM (time_multiplex_count) is no longer set by this pass -- it moved to
//     Pass C.  All three conduit.create ops survive (no erasure here).
//     Expected: pd1, pd2, pd3 all get fuse attrs; no time_multiplex_count.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @partial_dep
// CHECK:       conduit.create @pd1 {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"
// CHECK-NOT:   time_multiplex_count
// pd2 is NOT erased by this pass (TM erasure moved to Pass C).
// CHECK:       conduit.create @pd2 {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"
// CHECK-NOT:   time_multiplex_count
// pd3 gets fuse attrs but no time_multiplex_count.
// CHECK:       conduit.create @pd3 {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"
// CHECK-NOT:   time_multiplex_count
func.func @partial_dep() {
  conduit.create @pd1 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 2, 1>,
                  consumer_tiles = array<i64: 2, 2>,
                  element_type = memref<4096xbf16>,
                  depth = 1 : i64}
  conduit.create @pd2 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 2, 1>,
                  consumer_tiles = array<i64: 2, 2>,
                  element_type = memref<4096xbf16>,
                  depth = 1 : i64}
  conduit.create @pd3 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 2, 1>,
                  consumer_tiles = array<i64: 2, 2>,
                  element_type = memref<4096xbf16>,
                  depth = 1 : i64}

  %ta = conduit.put_memref_async {name = @pd1, num_elems = 4096 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 4096>,
            strides = array<i64: 1>} : !conduit.dma.token
  // pd2 depends on pd1 -- they form a 2-chain.
  %tb = conduit.put_memref_async [%ta : !conduit.dma.token]
            {name = @pd2, num_elems = 4096 : i64,
             offsets = array<i64: 4096>, sizes = array<i64: 4096>,
             strides = array<i64: 1>} : !conduit.dma.token
  // pd3 is independent -- no dep on ta or tb.
  %tc = conduit.put_memref_async {name = @pd3, num_elems = 4096 : i64,
            offsets = array<i64: 8192>, sizes = array<i64: 4096>,
            strides = array<i64: 1>} : !conduit.dma.token
  conduit.wait_all %tb : !conduit.dma.token
  conduit.wait_all %tc : !conduit.dma.token
  return
}

// -----

//===----------------------------------------------------------------------===//
// (e) Different endpoints -- token-chained but different consumer_tile.
//
//     t1 and t2 are ordered by dep, but they go to different consumer tiles.
//     They cannot share one physical S2MM because the destination differs.
//     The MM2S analysis annotates them in a fused_dma_channel_group (same
//     producer tile, sequential intervals) but no time_multiplex_count.
//     Expected: neither gets time_multiplex_count.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @different_consumer_tile
// CHECK:       conduit.create @de1 {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"
// CHECK-NOT:   time_multiplex_count
// CHECK:       conduit.create @de2 {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"
// CHECK-NOT:   time_multiplex_count
func.func @different_consumer_tile() {
  conduit.create @de1 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 3, 1>,
                  consumer_tiles = array<i64: 3, 2>,
                  element_type = memref<4096xbf16>,
                  depth = 1 : i64}
  conduit.create @de2 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 3, 1>,
                  consumer_tiles = array<i64: 3, 3>,   // different consumer row
                  element_type = memref<4096xbf16>,
                  depth = 1 : i64}

  %ta = conduit.put_memref_async {name = @de1, num_elems = 4096 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 4096>,
            strides = array<i64: 1>} : !conduit.dma.token
  %tb = conduit.put_memref_async [%ta : !conduit.dma.token]
            {name = @de2, num_elems = 4096 : i64,
             offsets = array<i64: 4096>, sizes = array<i64: 4096>,
             strides = array<i64: 1>} : !conduit.dma.token
  conduit.wait_all %tb : !conduit.dma.token
  return
}

// -----

//===----------------------------------------------------------------------===//
// (f) Packet-mode routing excluded.
//
//     Two channels ordered by dep but routing_mode = "packet".
//     Packet channels use shared physical channels with flow IDs -- temporal
//     mux would clobber the packet ID assignment.
//     The MM2S analysis still annotates fused_dma_channel_group on them
//     (same producer tile, sequential intervals), but no time_multiplex_count.
//     Expected: neither gets time_multiplex_count.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @packet_mode_excluded
// CHECK:       conduit.create @pkt1 {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"
// CHECK-NOT:   time_multiplex_count
// CHECK:       conduit.create @pkt2 {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"
// CHECK-NOT:   time_multiplex_count
func.func @packet_mode_excluded() {
  conduit.create @pkt1 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 4, 1>,
                  consumer_tiles = array<i64: 4, 2>,
                  element_type = memref<4096xbf16>,
                  depth = 1 : i64,
                  routing_mode = #conduit.routing_mode<packet>}
  conduit.create @pkt2 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 4, 1>,
                  consumer_tiles = array<i64: 4, 2>,
                  element_type = memref<4096xbf16>,
                  depth = 1 : i64,
                  routing_mode = #conduit.routing_mode<packet>}

  %ta = conduit.put_memref_async {name = @pkt1, num_elems = 4096 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 4096>,
            strides = array<i64: 1>} : !conduit.dma.token
  %tb = conduit.put_memref_async [%ta : !conduit.dma.token]
            {name = @pkt2, num_elems = 4096 : i64,
             offsets = array<i64: 4096>, sizes = array<i64: 4096>,
             strides = array<i64: 1>} : !conduit.dma.token
  conduit.wait_all %tb : !conduit.dma.token
  return
}

// -----

//===----------------------------------------------------------------------===//
// (g) Depth > 1 excluded.
//
//     Two channels ordered by dep, but depth=2 (concurrent double-buffer).
//     A depth>1 channel has concurrent semantics (producer pre-fills multiple
//     slots) that are incompatible with one-shot temporal mux.
//     The MM2S analysis also skips depth>1 channels (emits a remark and skips).
//     Expected: neither gets time_multiplex_count or fused_dma_channel_group.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @depth2_excluded
// CHECK:       conduit.create @d2a {
// CHECK-NOT:   time_multiplex_count
// CHECK:       conduit.create @d2b {
// CHECK-NOT:   time_multiplex_count
func.func @depth2_excluded() {
  conduit.create @d2a {slot_elems = 2 : i64,
                  producer_tile = array<i64: 5, 1>,
                  consumer_tiles = array<i64: 5, 2>,
                  element_type = memref<4096xbf16>,
                  depth = 2 : i64}
  conduit.create @d2b {slot_elems = 2 : i64,
                  producer_tile = array<i64: 5, 1>,
                  consumer_tiles = array<i64: 5, 2>,
                  element_type = memref<4096xbf16>,
                  depth = 2 : i64}

  %ta = conduit.put_memref_async {name = @d2a, num_elems = 4096 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 4096>,
            strides = array<i64: 1>} : !conduit.dma.token
  %tb = conduit.put_memref_async [%ta : !conduit.dma.token]
            {name = @d2b, num_elems = 4096 : i64,
             offsets = array<i64: 4096>, sizes = array<i64: 4096>,
             strides = array<i64: 1>} : !conduit.dma.token
  conduit.wait_all %tb : !conduit.dma.token
  return
}

// -----

//===----------------------------------------------------------------------===//
// (h) Link src excluded from temporal mux.
//
//     lk1 appears in a conduit.scatter dsts list.  lk2 is ordered after
//     lk1 by a token dep at the same endpoint.  Because lk1 is a relay
//     source (lk channel resources are managed by linkPhase), it must not
//     be merged by the temporal mux analysis.
//     The MM2S analysis sees lk1/lk2 as sequential at the same producer tile
//     and annotates fused_dma_channel_group on them.
//     Expected: lk1 and lk2 get fuse attrs but no time_multiplex_count.
//              lk_dst is on a different producer tile, no fuse attrs.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @link_src_excluded
// CHECK:       conduit.create @lk1 {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"
// CHECK-NOT:   time_multiplex_count
// CHECK:       conduit.create @lk2 {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group = "group0"
// CHECK-NOT:   time_multiplex_count
// lk_dst is on a different producer tile (6,1) vs lk1/lk2 (6,2) -- no fuse attrs.
// CHECK:       conduit.create @lk_dst {
// CHECK-NOT:   fused_dma_channel_group
func.func @link_src_excluded() {
  conduit.create @lk1 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 6, 2>,
                  consumer_tiles = array<i64: 6, 3>,
                  element_type = memref<4096xbf16>,
                  depth = 1 : i64}
  conduit.create @lk2 {slot_elems = 1 : i64,
                  producer_tile = array<i64: 6, 2>,
                  consumer_tiles = array<i64: 6, 3>,
                  element_type = memref<4096xbf16>,
                  depth = 1 : i64}
  conduit.create @lk_dst {slot_elems = 1 : i64,
                  producer_tile = array<i64: 6, 1>,
                  consumer_tiles = array<i64: 6, 2>,
                  element_type = memref<4096xbf16>,
                  depth = 1 : i64}
  conduit.scatter{src = @lk1, dsts = [@lk_dst] {memtile = "tile(6,1)"}}

  %ta = conduit.put_memref_async {name = @lk1, num_elems = 4096 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 4096>,
            strides = array<i64: 1>} : !conduit.dma.token
  %tb = conduit.put_memref_async [%ta : !conduit.dma.token]
            {name = @lk2, num_elems = 4096 : i64,
             offsets = array<i64: 4096>, sizes = array<i64: 4096>,
             strides = array<i64: 1>} : !conduit.dma.token
  conduit.wait_all %tb : !conduit.dma.token
  return
}
