// RUN: aie-opt --conduit-to-dma --verify-diagnostics %s
//
// P2-A Step 3.5: Mixed-mode design — circuit DMA and packet DMA fallback.
//
// Producer tile (0,3) has 2 MM2S channels. Two circuit conduits consume both.
// The mode=any conduit cannot use circuit mode (both channels taken by
// explicit circuit flows). HOWEVER, both channels are circuit-mode (not
// packet-mode), so Step 3.5c cannot designate a new packet-mode channel
// (all slots are taken). Step 3.5 returns false → Step 4 error.
//
// But producer tile (1,3) has no prior flows: circuit_c takes ch 0 and
// packet_d takes ch 1 (packet-mode). A mode=any conduit from (1,3) finds
// ch 1 packet-mode → Step 3.5 succeeds.
//
// Expected: 2 aie.flow ops (circuit_a, circuit_b from (0,3));
//           2 aie.packet_flow ops (packet_d from (1,3) + fallback from (1,3));
//           Step 4 error for 'circuit_overflow' from (0,3).

module @pkt_fallback_mixed_modes {
  // expected-error @+1 {{conduit-to-dma: no DMA resources available for conduit 'circuit_overflow'}}
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %t13 = aie.tile(1, 3)
    %t23 = aie.tile(2, 3)
    %t33 = aie.tile(3, 3)
    %t15 = aie.tile(1, 5)
    %t25 = aie.tile(2, 5)
    %t35 = aie.tile(3, 5)

    // circuit_a and circuit_b from (0,3): consume both MM2S channels as circuit.
    conduit.create @circuit_a {slot_elems = 4 : i64,
                    element_type = memref<4xi32>, depth = 1 : i64,
                                        routing_mode = #conduit.routing_mode<circuit>}
    conduit.create @circuit_b {slot_elems = 4 : i64,
                    element_type = memref<4xi32>, depth = 1 : i64,
                                        routing_mode = #conduit.routing_mode<circuit>}

    // circuit_overflow: mode=any from (0,3); both circuit-mode channels taken;
    // Step 3.5c fails (no packet-mode channel, no free channel) → Step 4 error.
    conduit.create @circuit_overflow {slot_elems = 4 : i64,
                    element_type = memref<4xi32>, depth = 1 : i64
                    }

    // packet_d from (1,3): explicit packet, designates ch 1 as packet-mode.
    conduit.create @packet_d {slot_elems = 4 : i64,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}

    // anymode_e from (1,3): circuit ch 0 is free → mode=any takes it as circuit.
    // (not a fallback — ch 0 is free, so circuit DMA is used directly)
    conduit.create @anymode_e {slot_elems = 4 : i64,
                    element_type = memref<4xi32>, depth = 1 : i64
                    }

    %core03 = aie.core(%t03) {
      conduit.acquire {name = @circuit_a, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @circuit_b, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @circuit_overflow, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core13 = aie.core(%t13) {
      conduit.acquire {name = @packet_d, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @anymode_e, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core23 = aie.core(%t23) {
      conduit.acquire {name = @circuit_a, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core33 = aie.core(%t33) {
      conduit.acquire {name = @circuit_b, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core15 = aie.core(%t15) {
      conduit.acquire {name = @circuit_overflow, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core25 = aie.core(%t25) {
      conduit.acquire {name = @packet_d, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core35 = aie.core(%t35) {
      conduit.acquire {name = @anymode_e, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
  }
}
