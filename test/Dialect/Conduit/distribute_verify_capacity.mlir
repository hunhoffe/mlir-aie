// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// M7-dist composed-consume: Denolf Eq. 48 buffer capacity check for 1:N distribute.
//
// These tests verify that the cross-conduit composed consume check correctly
// detects when a source buffer is undersized due to a slow destination consumer
// gating buffer reuse in the distribute pattern.
//
// Key property: per-edge M7 checks on individual conduits pass, but the
// composed consume analysis reveals that the source capacity is insufficient.
// The slowest consumer prevents buffer containers from being freed, causing
// occupancy to exceed source capacity.
//
// Section 1: 1→2 distribute, slow consumer (period 3 vs 1) — source too small.
// Section 2: 1→3 distribute, one very slow consumer causes overflow.

// -----

// Section 1: FAIL — 1→2 distribute, D2 is a slow consumer.
//
// src: P=[3], C=[3], cap=3 → per-edge OK, M7 peak=3 = cap ✓
// dst1: P=[3], C=[3], cap=6 → per-edge OK ✓
// dst2: P=[1,1,1], C=[1,1,1], cap=6 → per-edge OK ✓ (sum=3, period=3: 3*3=9=3*3)
//
// Composed consume (Eq. 48): H=lcm(1,1,3)=3
//   t=0: cumProd=3, cumCons1=3, cumCons2=1, composed=min(3,1)=1, occ=3-1=2
//   t=1: cumProd=6, cumCons1=6, cumCons2=2, composed=min(6,2)=2, occ=6-2=4
//   t=2: cumProd=9, cumCons1=9, cumCons2=3, composed=min(9,3)=3, occ=9-3=6
//   Peak=6 > srcCap=3 → FAIL

func.func @distribute_slow_consumer_overflow() {
  conduit.create {name = "ov_src", capacity = 3 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<3xi32>,
                  depth = 3 : i64,
                  producer_rates = array<i64: 3>,
                  consumer_rates = array<i64: 3>}
  conduit.create {name = "ov_d1", capacity = 6 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<6xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 3>,
                  consumer_rates = array<i64: 3>}
  conduit.create {name = "ov_d2", capacity = 6 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 1, 2>,
                  element_type = memref<6xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 1, 1, 1>,
                  consumer_rates = array<i64: 1, 1, 1>}
  // expected-error@+1 {{'conduit.link' op M7-dist composed-consume (Denolf Eq. 48): source buffer capacity insufficient for multi-consumer distribute: peak occupancy=6 exceeds source capacity=3 (bottleneck consumer: 'ov_d2', hyper-period=3 steps; a container can only be freed after ALL 2 consumers have consumed it)}}
  conduit.link {srcs = ["ov_src"], dsts = ["ov_d1", "ov_d2"],
                mode = #conduit.link_mode<distribute>, memtile = "tile(0,1)"}
  return
}

// -----

// Section 2: FAIL — 1→3 distribute, D3 is very slow (period 4).
//
// src: P=[4], C=[4], cap=4 → per-edge OK, M7 peak=4 = cap ✓
// dst1: P=[4], C=[4], cap=8 → per-edge OK ✓
// dst2: P=[2,2], C=[2,2], cap=8 → per-edge OK ✓ (sum=4, period=2: 4*2=8=4*2)
// dst3: P=[1,1,1,1], C=[1,1,1,1], cap=8 → per-edge OK ✓ (sum=4, period=4: 4*4=16=4*4)
//
// Composed consume (Eq. 48): H=lcm(1,1,2,4)=4
//   t=0: cumProd=4, cumCons1=4, cumCons2=2, cumCons3=1, composed=1, occ=3
//   t=1: cumProd=8, cumCons1=8, cumCons2=4, cumCons3=2, composed=2, occ=6
//   t=2: cumProd=12, cumCons1=12, cumCons2=6, cumCons3=3, composed=3, occ=9
//   t=3: cumProd=16, cumCons1=16, cumCons2=8, cumCons3=4, composed=4, occ=12
//   Peak=12 > srcCap=4 → FAIL

func.func @distribute_three_consumer_overflow() {
  conduit.create {name = "t3_src", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<4xi32>,
                  depth = 4 : i64,
                  producer_rates = array<i64: 4>,
                  consumer_rates = array<i64: 4>}
  conduit.create {name = "t3_d1", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<8xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 4>,
                  consumer_rates = array<i64: 4>}
  conduit.create {name = "t3_d2", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 1, 2>,
                  element_type = memref<8xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 2, 2>,
                  consumer_rates = array<i64: 2, 2>}
  conduit.create {name = "t3_d3", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 2, 2>,
                  element_type = memref<8xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 1, 1, 1, 1>,
                  consumer_rates = array<i64: 1, 1, 1, 1>}
  // expected-error@+1 {{'conduit.link' op M7-dist composed-consume (Denolf Eq. 48): source buffer capacity insufficient for multi-consumer distribute: peak occupancy=12 exceeds source capacity=4 (bottleneck consumer: 't3_d3', hyper-period=4 steps; a container can only be freed after ALL 3 consumers have consumed it)}}
  conduit.link {srcs = ["t3_src"], dsts = ["t3_d1", "t3_d2", "t3_d3"],
                mode = #conduit.link_mode<distribute>, memtile = "tile(0,1)"}
  return
}
