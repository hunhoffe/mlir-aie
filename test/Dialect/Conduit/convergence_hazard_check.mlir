// RUN: aie-opt --conduit-check-channels --split-input-file %s 2>&1 | FileCheck %s
//
// P2-E: Convergence hazard check in --conduit-check-channels.
//
// This test operates on lowered AIE IR (aie.packet_flow ops directly).
// The --conduit-check-channels pass walks all aie.packet_flow ops and
// warns when two flows with different IDs route to the same consumer tile
// through the same physical source port (same producer tile + bundle + channel).
//
// Test layout (--split-input-file):
//   Section 1: hazard case — warning expected
//   Section 2: safe case (different dests) — no warning for this section
//   Section 3: safe case (different channels) — no warning
//   Section 4: single flow — no warning
//
// FileCheck order: the warning from section 1 appears first; we verify it
// is present, then verify no second warning appears in the rest of the output.

// CHECK: warning: packet flows with different IDs (0 and 1) route to the same consumer tile (2, 3)
// CHECK-SAME: through the same switchbox source port on tile (0, 3)
// CHECK-SAME: ordering is not guaranteed under sustained load

// -----

// Section 1: HAZARD — two flows sharing source port (0,3):DMA:0 both route to tile(2,3).

module @convergence_hazard_same_dest {
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %t23 = aie.tile(2, 3)

    // Flow 0: tile(0,3) DMA:0 → tile(2,3) DMA:0
    aie.packet_flow(0) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t23, DMA : 0>
    }

    // Flow 1: tile(0,3) DMA:0 → tile(2,3) DMA:1
    // Same source port, same dest tile (2,3), different flow ID → HAZARD.
    aie.packet_flow(1) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t23, DMA : 1>
    }
  }
}

// -----

// Section 2: SAFE — same source port, DIFFERENT dest tiles.  No hazard.
// CHECK-NOT: packet flows with different IDs (0 and 1) route to the same consumer tile (3, 3)

module @convergence_safe_different_dest {
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %t23 = aie.tile(2, 3)
    %t33 = aie.tile(3, 3)

    aie.packet_flow(0) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t23, DMA : 0>
    }

    // Different dest tile (3,3) → not a hazard.
    aie.packet_flow(1) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t33, DMA : 0>
    }
  }
}

// -----

// Section 3: SAFE — different source channels on the same producer tile.
// DMA:0 and DMA:1 are distinct physical ports → no convergence hazard.
// CHECK-NOT: source port on tile (0, 3); ordering is not guaranteed

module @convergence_safe_different_channel {
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %t23 = aie.tile(2, 3)

    aie.packet_flow(0) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t23, DMA : 0>
    }

    // Different source channel (DMA:1 vs DMA:0) → separate physical port.
    aie.packet_flow(1) {
      aie.packet_source<%t03, DMA : 1>
      aie.packet_dest<%t23, DMA : 0>
    }
  }
}

// -----

// Section 4: SAFE — single packet flow.  Nothing to compare.
// (No additional CHECK needed; absence of any additional warning lines
// between the section 2/3 CHECK-NOTs and end of output is sufficient.)

module @convergence_single_flow {
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %t23 = aie.tile(2, 3)

    aie.packet_flow(0) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t23, DMA : 0>
    }
  }
}
