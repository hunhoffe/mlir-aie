// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// M6-dist / M7-dist: 1:N distribute with Denolf Eq. 46/48 composed consume.
//
// Positive tests: distribute patterns where all per-edge checks and the
// cross-conduit composed consume buffer capacity check (Denolf Eq. 48) pass.
//
// Section 1: 1→2 distribute, uniform consumers, sufficient source capacity.
// Section 2: 1→3 distribute, mixed-period consumers, source capacity sufficient.
// Section 3: 1→2 distribute, one fast one slow consumer, source capacity sufficient.

// -----

// Section 1: PASS — 1→2 distribute, both consumers uniform rate [2], source cap=4.
//
// src: P=[2], C=[2], cap=4 → per-edge OK, M7 peak=2 <= 4 ✓
// dst1: P=[2], C=[2], cap=4 → per-edge OK ✓
// dst2: P=[2], C=[2], cap=4 → per-edge OK ✓
//
// Composed consume (Eq. 48): H=lcm(1,1,1)=1
//   t=0: cumProd=2, cumCons1=2, cumCons2=2, composed=2, occ=0
//   Peak=0 <= srcCap=4 ✓

func.func @distribute_composed_uniform_pass() {
  conduit.create {name = "cc_src", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<4xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  conduit.create {name = "cc_d1", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<4xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  conduit.create {name = "cc_d2", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 1, 2>,
                  element_type = memref<4xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  conduit.link {srcs = ["cc_src"], dsts = ["cc_d1", "cc_d2"],
                mode = "distribute", memtile = "tile(0,1)"}
  return
}

// -----

// Section 2: PASS — 1→3 distribute, mixed periods, source cap=6 (sufficient).
//
// src: P=[3], C=[3], cap=6 → per-edge OK, M7 peak=3 <= 6 ✓
// dst1: P=[3], C=[3], cap=6 → per-edge OK ✓
// dst2: P=[1,2], C=[1,2], cap=6 → per-edge OK ✓ (sum=3, period=2)
// dst3: P=[3], C=[3], cap=6 → per-edge OK ✓
//
// Composed consume (Eq. 48): H=lcm(1,1,2,1)=2
//   t=0: cumProd=3, cumCons1=3, cumCons2=1, cumCons3=3, composed=min(3,1,3)=1, occ=2
//   t=1: cumProd=6, cumCons1=6, cumCons2=3, cumCons3=6, composed=min(6,3,6)=3, occ=3
//   Peak=3 <= srcCap=6 ✓

func.func @distribute_composed_mixed_pass() {
  conduit.create {name = "mx_src", capacity = 6 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<6xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 3>,
                  consumer_rates = array<i64: 3>}
  conduit.create {name = "mx_d1", capacity = 6 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<6xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 3>,
                  consumer_rates = array<i64: 3>}
  conduit.create {name = "mx_d2", capacity = 6 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 1, 2>,
                  element_type = memref<6xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 1, 2>,
                  consumer_rates = array<i64: 1, 2>}
  conduit.create {name = "mx_d3", capacity = 6 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 2, 2>,
                  element_type = memref<6xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 3>,
                  consumer_rates = array<i64: 3>}
  conduit.link {srcs = ["mx_src"], dsts = ["mx_d1", "mx_d2", "mx_d3"],
                mode = "distribute", memtile = "tile(0,1)"}
  return
}

// -----

// Section 3: PASS — 1→2 distribute, one slow consumer, source capacity large enough.
//
// src: P=[2], C=[2], cap=4 → per-edge OK, M7 peak=2 <= 4 ✓
// dst1: P=[2], C=[2], cap=4 → per-edge OK ✓
// dst2: P=[1,1], C=[1,1], cap=4 → per-edge OK ✓ (sum=2, period=2)
//
// Composed consume (Eq. 48): H=lcm(1,1,2)=2
//   t=0: cumProd=2, cumCons1=2, cumCons2=1, composed=min(2,1)=1, occ=1
//   t=1: cumProd=4, cumCons1=4, cumCons2=2, composed=min(4,2)=2, occ=2
//   Peak=2 <= srcCap=4 ✓

func.func @distribute_composed_slow_consumer_pass() {
  conduit.create {name = "sl_src", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<4xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  conduit.create {name = "sl_d1", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<4xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  conduit.create {name = "sl_d2", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 1, 2>,
                  element_type = memref<4xi32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 1, 1>,
                  consumer_rates = array<i64: 1, 1>}
  conduit.link {srcs = ["sl_src"], dsts = ["sl_d1", "sl_d2"],
                mode = "distribute", memtile = "tile(0,1)"}
  return
}
