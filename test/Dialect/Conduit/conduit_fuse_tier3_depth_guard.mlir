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

aie.device(npu1) {

// (a) Tier 3 depth=1 fused normally — no remark.
conduit.create @shallow_dma {slot_elems = 8 : i64,
                element_type = memref<8xi32>,
                depth = 1 : i64}
conduit.create @tier2_shallow {slot_elems = 8 : i64,
                element_type = memref<8xi32>,
                depth = 1 : i64}

// (b) Tier 3 depth=2 skipped with remark.
// expected-remark @+1 {{conduit-fuse-channels: skipping 'deep_tier3' — Tier 3 channel with depth>1 not supported in fuse groups}}
conduit.create @deep_tier3 {slot_elems = 16 : i64,
                element_type = memref<8xi32>,
                depth = 2 : i64}
conduit.create @tier2_partner {slot_elems = 8 : i64,
                element_type = memref<8xi32>,
                depth = 1 : i64}

// (c) Two Tier 2 channels on same tile — fused normally.
conduit.create @t2_a {slot_elems = 8 : i64,
                element_type = memref<8xi32>,
                depth = 1 : i64}
conduit.create @t2_b {slot_elems = 8 : i64,
                element_type = memref<8xi32>,
                depth = 1 : i64}

func.func @fuse_tier3_depth1() {
  conduit.put_memref {name = @shallow_dma, num_elems = 8 : i64,
                      offsets = array<i64: 0>, sizes = array<i64: 8>,
                      strides = array<i64: 1>}

  %w = conduit.acquire {name = @tier2_shallow, count = 1 : i64,
                        port = #conduit.port<Consume>}
          : !conduit.window<memref<8xi32>>
  conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>

  return
}

func.func @skip_tier3_depth2() {
  conduit.put_memref {name = @deep_tier3, num_elems = 8 : i64,
                      offsets = array<i64: 0>, sizes = array<i64: 8>,
                      strides = array<i64: 1>}

  %w = conduit.acquire {name = @tier2_partner, count = 1 : i64,
                        port = #conduit.port<Consume>}
          : !conduit.window<memref<8xi32>>
  conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>

  return
}

func.func @fuse_tier2_only() {
  %wa = conduit.acquire {name = @t2_a, count = 1 : i64,
                         port = #conduit.port<Consume>}
           : !conduit.window<memref<8xi32>>
  conduit.release %wa {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>

  %wb = conduit.acquire {name = @t2_b, count = 1 : i64,
                         port = #conduit.port<Consume>}
           : !conduit.window<memref<8xi32>>
  conduit.release %wb {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<8xi32>>

  return
}

} // aie.device
