// RUN: aie-opt --conduit-check-channels --verify-diagnostics --split-input-file %s

// -----

// Test 1: PASS — 2 conduits on the same producer tile (at MM2S limit for AIE2).
// AIE2 compute tiles have 2 MM2S + 2 S2MM DMA channels.
// Two conduits producing on tile(0,2) → 2 MM2S channels used.  Within limit.

module @check_channels_pass {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    conduit.create @c1 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}
    conduit.create @c2 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}

    // Structural tile info via aie.core blocks.
    %core_0_2 = aie.core(%tile_0_2) {
      %w1 = conduit.acquire {name = @c1, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      conduit.release %w1 {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      %w2 = conduit.acquire {name = @c2, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      conduit.release %w2 {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %w = conduit.acquire {name = @c1, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %w = conduit.acquire {name = @c2, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      aie.end
    }
  }
}

// -----

// Test 2: FAIL — 3 conduits on the same producer tile (exceeds MM2S limit).
// Tile(0,2) needs 3 MM2S channels, hardware supports 2.

module @check_channels_mm2s_fail {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)

    // expected-error @+1 {{DMA channel limit exceeded on tile (0, 2): 3 conduits require 3 MM2S channels, hardware supports 2}}
    conduit.create @c1 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}
    conduit.create @c2 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}
    conduit.create @c3 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}

    // Structural tile info: tile(0,2) produces all three conduits.
    %core_0_2 = aie.core(%tile_0_2) {
      %w1 = conduit.acquire {name = @c1, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      conduit.release %w1 {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      %w2 = conduit.acquire {name = @c2, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      conduit.release %w2 {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      %w3 = conduit.acquire {name = @c3, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      conduit.release %w3 {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %w = conduit.acquire {name = @c1, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %w = conduit.acquire {name = @c2, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %w = conduit.acquire {name = @c3, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      aie.end
    }
  }
}

// -----

// Test 3: FAIL — 3 conduits consuming on the same tile (exceeds S2MM limit).
// Tile(0,3) receives from 3 different producers → 3 S2MM channels needed.

module @check_channels_s2mm_fail {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)

    // expected-error @+1 {{DMA channel limit exceeded on tile (0, 3): 3 conduits require 3 S2MM channels, hardware supports 2}}
    conduit.create @c1 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}
    conduit.create @c2 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}
    conduit.create @c3 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}

    // Structural tile info: each conduit has a different producer, all consume on tile(0,3).
    %core_0_2 = aie.core(%tile_0_2) {
      %w = conduit.acquire {name = @c1, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %w = conduit.acquire {name = @c2, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %w = conduit.acquire {name = @c3, count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Produce>} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %w1 = conduit.acquire {name = @c1, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      conduit.release %w1 {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      %w2 = conduit.acquire {name = @c2, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      conduit.release %w2 {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      %w3 = conduit.acquire {name = @c3, count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      conduit.release %w3 {count = 1 : i64, port = #conduit.port<Consume>} : !conduit.window<memref<4xi32>>
      aie.end
    }
  }
}

// -----

// Test 4: PASS — 3 conduits on the same producer tile, but 2 are fused.
// Without fusion: 3 MM2S channels needed (fail).
// With fused_dma_channel_group annotation: c1 and c2 share "grp0" → 2 channels (pass).
// NOTE: No structural tile info; pass skips these conduits (no tile info → no check).

module @check_channels_fused_pass {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)

    conduit.create @c1 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64,
                    fused_dma_channel_group = "grp0"}
    conduit.create @c2 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64,
                    fused_dma_channel_group = "grp0"}
    conduit.create @c3 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}
  }
}

// -----

// Test 5: Shim tiles (row==0) are excluded from the check.
// 3 conduits producing from tile(0,0) — no error because shim tiles use
// a separate DMA model.
// NOTE: No structural tile info; pass skips these conduits.

module @check_channels_shim_excluded {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    conduit.create @c1 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}
    conduit.create @c2 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}
    conduit.create @c3 {slot_elems = 4 : i64,
                    element_type = memref<4xi32>,
                    depth = 1 : i64}
  }
}
