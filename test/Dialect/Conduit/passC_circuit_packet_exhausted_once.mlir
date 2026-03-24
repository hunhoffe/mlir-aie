// RUN: not aie-opt --conduit-to-dma %s 2>&1 | FileCheck %s
//
// Regression test: B-3 fix at the circuit+packet-fallback exhaustion site.
//
// Specific site: ConduitToDMARoute.cpp Phase 4.5a — when routing_mode="any",
// circuit DMA MM2S channels are full, AND packet fallback also fails (all
// MM2S channels are already circuit-mode, none available for packet
// designation). Before B-3 fix: passFailed=true; continue — loop processes
// the next conduit and emits a SECOND error. After B-3 fix: passFailed=true;
// return — loop exits immediately.
//
// Setup (xcve2302, compute tile(3,3) has max 2 MM2S channels):
//   conduit_a: routing_mode="circuit" → claims MM2S channel 0
//   conduit_b: routing_mode="circuit" → claims MM2S channel 1
//   conduit_c: routing_mode="any" → circuit exhausted + packet fallback fails
//     (tryPacketFallback step 3.5c: nextCh=2>=maxMM2S=2, no free packet slot)
//     → "no DMA resources available" error
//
// conduit_d is present: would trigger a SECOND error IF the loop continued.
// The CHECK-NOT below proves the loop stopped after conduit_c.
//
// CHECK: conduit-to-dma: no DMA resources available for conduit 'conduit_c'
// CHECK-NOT: conduit-to-dma: no DMA resources available for conduit 'conduit_d'

module @passC_circuit_packet_exhausted_once {
  aie.device(xcve2302) {
    // Non-adjacent tiles: tile(3,3) produces, tile(1,3) consumes.
    %prod  = aie.tile(3, 3)
    %cons1 = aie.tile(1, 3)

    // conduit_a and conduit_b: circuit-mode, each consume one MM2S channel.
    conduit.create {name = "conduit_a", capacity = 32 : i64, depth = 1 : i64,
                    element_type = memref<32xi32>,
                    producer_tile = array<i64: 3, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    shim_consumer_tiles = array<i64>,
                                        routing_mode = #conduit.routing_mode<circuit>}
    conduit.create {name = "conduit_b", capacity = 32 : i64, depth = 1 : i64,
                    element_type = memref<32xi32>,
                    producer_tile = array<i64: 3, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    shim_consumer_tiles = array<i64>,
                                        routing_mode = #conduit.routing_mode<circuit>}

    // conduit_c: routing_mode="any" — circuit exhausted + packet fails.
    // This is the B-3 site (passFailed+continue → passFailed+return).
    conduit.create {name = "conduit_c", capacity = 32 : i64, depth = 1 : i64,
                    element_type = memref<32xi32>,
                    producer_tile = array<i64: 3, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    shim_consumer_tiles = array<i64>
                    }

    // conduit_d: would also fail with "no DMA resources" IF the loop continued.
    // CHECK-NOT above verifies the loop returned after conduit_c.
    conduit.create {name = "conduit_d", capacity = 32 : i64, depth = 1 : i64,
                    element_type = memref<32xi32>,
                    producer_tile = array<i64: 3, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    shim_consumer_tiles = array<i64>
                    }

    %cprod = aie.core(%prod)  { aie.end }
    %ccons = aie.core(%cons1) { aie.end }
  }
}
