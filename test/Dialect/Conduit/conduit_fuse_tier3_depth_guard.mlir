// RUN: aie-opt --conduit-fuse-channels --verify-diagnostics %s
//
// --conduit-fuse-channels: Tier 3 depth>1 guard.
//
// Tier 3 channels (those with put_memref/get_memref ops) with depth>1 have a
// multi-block BD ring.  Fusing them into a channel group would create BD chain
// ordering conflicts, so the pass must skip them with a remark and leave them
// unannotated.
//
// Tier 3 channels at depth=1 can be fused normally (no remark emitted).
// This only arises in hand-authored Conduit IR; Pass A/B only emit Tier 3 at
// depth=1.
//
// Test plan:
//   (a) Tier 3 depth=1 + Tier 2 depth=1 on same tile: both fuseable, no remark.
//   (b) Tier 3 depth=2 on same tile as Tier 2 depth=1: deep_tier3 skipped with
//       remark; Tier 2 partner is now a singleton → not annotated either.
//   (c) Two Tier 2 channels on same tile (no Tier 3): fused normally, no remark.

// -----

//===----------------------------------------------------------------------===//
// (a) Tier 3 depth=1 fused normally — no remark.
//
//     shallow_dma (Tier 3, depth=1) and tier2_chan (Tier 2, depth=1) are on
//     the same producer tile [0, 2] with sequential non-overlapping ops.
//     Both should be annotated with the same fuse group; no remark emitted.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_tier3_depth1
// CHECK:       conduit.create @shallow_dma {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group// CHECK:       conduit.create @tier2_shallow {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group
func.func @fuse_tier3_depth1() {
  conduit.create @shallow_dma {capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  conduit.create @tier2_shallow {capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 4>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}

  // shallow_dma uses a put_memref (Tier 3), then tier2_shallow uses acquire/release.
  // Non-overlapping: shallow_dma interval ends before tier2_shallow begins.
  conduit.put_memref {name = "shallow_dma", num_elems = 8 : i64,
                      offsets = array<i64: 0>, sizes = array<i64: 8>,
                      strides = array<i64: 1>}

  %w = conduit.acquire {name = "tier2_shallow", count = 1 : i64,
                        port = #conduit.port<Consume>}
          : !conduit.window<memref<8xi32>>
  conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>

  return
}

// -----

//===----------------------------------------------------------------------===//
// (b) Tier 3 depth=2 skipped with remark — partner becomes singleton.
//
//     deep_tier3 (Tier 3, depth=2) and tier2_partner (Tier 2, depth=1) are on
//     the same producer tile [1, 2] with sequential non-overlapping ops.
//     deep_tier3 must be skipped (remark emitted); tier2_partner then has no
//     fusion partner → it is a singleton → also not annotated.
//===----------------------------------------------------------------------===//

// The remark is attached to the conduit.create op for deep_tier3.
// CHECK-LABEL: func.func @skip_tier3_depth2
// CHECK-NOT:   fused_dma_channel_group

func.func @skip_tier3_depth2() {
  // expected-remark @+1 {{conduit-fuse-channels: skipping 'deep_tier3' — Tier 3 channel with depth>1 not supported in fuse groups}}
  conduit.create @deep_tier3 {capacity = 16 : i64,
                  producer_tile = array<i64: 1, 2>,
                  consumer_tiles = array<i64: 1, 3>,
                  element_type = memref<8xi32>,
                  depth = 2 : i64}
  conduit.create @tier2_partner {capacity = 8 : i64,
                  producer_tile = array<i64: 1, 2>,
                  consumer_tiles = array<i64: 1, 4>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}

  // deep_tier3 uses put_memref (Tier 3); tier2_partner uses acquire/release.
  // Sequential non-overlapping — would be fuseable but for the depth>1 guard.
  conduit.put_memref {name = "deep_tier3", num_elems = 8 : i64,
                      offsets = array<i64: 0>, sizes = array<i64: 8>,
                      strides = array<i64: 1>}

  %w = conduit.acquire {name = "tier2_partner", count = 1 : i64,
                        port = #conduit.port<Consume>}
          : !conduit.window<memref<8xi32>>
  conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>

  return
}

// -----

//===----------------------------------------------------------------------===//
// (c) Two Tier 2 channels on same tile — no Tier 3 ops — fused normally.
//
//     Confirms the guard does not affect pure Tier 2 fusion.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fuse_tier2_only
// CHECK:       conduit.create @t2_a {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group// CHECK:       conduit.create @t2_b {{{.*}}fuse_mode = "static"{{.*}}fused_dma_channel_group
func.func @fuse_tier2_only() {
  conduit.create @t2_a {capacity = 8 : i64,
                  producer_tile = array<i64: 2, 2>,
                  consumer_tiles = array<i64: 2, 3>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  conduit.create @t2_b {capacity = 8 : i64,
                  producer_tile = array<i64: 2, 2>,
                  consumer_tiles = array<i64: 2, 4>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}

  %wa = conduit.acquire {name = "t2_a", count = 1 : i64,
                         port = #conduit.port<Consume>}
           : !conduit.window<memref<8xi32>>
  conduit.release %wa {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>

  %wb = conduit.acquire {name = "t2_b", count = 1 : i64,
                         port = #conduit.port<Consume>}
           : !conduit.window<memref<8xi32>>
  conduit.release %wb {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>

  return
}
