// RUN: aie-opt --conduit-infer-modes --verify-diagnostics %s
//
// Test Step 3.5: circuit DMA exhausted → packet fallback remark.
//
// tile(0,2) has 2 MM2S channels (AIE2 npu1).
// Two circuit-mode conduits consume both channels.
// A third conduit with absent routing_mode (unresolved) finds no circuit
// channel and falls back to "packet", emitting a remark.

module @test_step35_remark {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t13 = aie.tile(1, 3)
    %t14 = aie.tile(1, 4)
    %t15 = aie.tile(1, 5)
    // Two circuit conduits exhaust both MM2S channels on tile(0,2).
    conduit.create @c1 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<circuit>}
    conduit.create @c2 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<circuit>}
    // Third conduit with absent routing_mode (unresolved): circuit exhausted,
    // packet fallback emits remark.
    // expected-remark @+1 {{conduit-infer-modes: resolved unresolved routing_mode to "packet" (circuit DMA exhausted on tile (0,2))}}
    conduit.create @c3_any {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}

    // Structural tile info: tile(0,2) produces all three conduits.
    %core02 = aie.core(%t02) {
      %w1 = conduit.acquire {name = @c1, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      conduit.release %w1 {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      %w2 = conduit.acquire {name = @c2, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      conduit.release %w2 {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      %w3 = conduit.acquire {name = @c3_any, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      conduit.release %w3 {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core13 = aie.core(%t13) {
      %w = conduit.acquire {name = @c1, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core14 = aie.core(%t14) {
      %w = conduit.acquire {name = @c2, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core15 = aie.core(%t15) {
      %w = conduit.acquire {name = @c3_any, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      aie.end
    }
  }
}
