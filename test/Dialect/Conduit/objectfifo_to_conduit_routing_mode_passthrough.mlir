// RUN: aie-opt --objectfifo-to-conduit -split-input-file %s 2>&1 | FileCheck %s
//
// Task #105 / Step 2 — explicit routing_mode StringAttr passthrough.
//
// Pass A propagates an explicit `routing_mode` discardable StringAttr
// from the source aie.objectfifo to the resulting conduit.create's
// routing_mode enum.  The explicit attr ALWAYS wins over via_DMA /
// via_cascade / aie_stream / dims/repeat derivations — the user has
// taken explicit control of routing, and Pass A trusts the override.
//
// Allowed values match the RoutingMode enum:
//   circuit, packet, cascade, stream, shared_memory, dma
//
// Six split-input-file scenarios cover each enum value, plus a seventh
// scenario showing that an explicit override beats via_DMA.

// (1) circuit
// CHECK-LABEL: module @rm_circuit
// CHECK: conduit.create @chan
// CHECK-SAME: routing_mode = #conduit.routing_mode<circuit>
module @rm_circuit {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @chan (%tile_0_2, {%tile_0_3}, 2 : i32)
        {routing_mode = "circuit"} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// (2) packet
// CHECK-LABEL: module @rm_packet
// CHECK: conduit.create @chan
// CHECK-SAME: routing_mode = #conduit.routing_mode<packet>
module @rm_packet {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @chan (%tile_0_2, {%tile_0_3}, 2 : i32)
        {routing_mode = "packet"} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// (3) cascade
// CHECK-LABEL: module @rm_cascade
// CHECK: conduit.create @chan
// CHECK-SAME: routing_mode = #conduit.routing_mode<cascade>
module @rm_cascade {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @chan (%tile_0_2, {%tile_0_3}, 1 : i32)
        {routing_mode = "cascade"} : !aie.objectfifo<memref<1xi32>>
  }
}

// -----

// (4) stream
// CHECK-LABEL: module @rm_stream
// CHECK: conduit.create @chan
// CHECK-SAME: routing_mode = #conduit.routing_mode<stream>
module @rm_stream {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @chan (%tile_0_2, {%tile_0_3}, 2 : i32)
        {routing_mode = "stream"} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// (5) shared_memory
// CHECK-LABEL: module @rm_shared_memory
// CHECK: conduit.create @chan
// CHECK-SAME: routing_mode = #conduit.routing_mode<shared_memory>
module @rm_shared_memory {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @chan (%tile_0_2, {%tile_0_3}, 2 : i32)
        {routing_mode = "shared_memory"} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// (6) dma
// CHECK-LABEL: module @rm_dma
// CHECK: conduit.create @chan
// CHECK-SAME: routing_mode = #conduit.routing_mode<dma>
module @rm_dma {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @chan (%tile_0_2, {%tile_0_3}, 2 : i32)
        {routing_mode = "dma"} : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// (7) Explicit override beats via_DMA derivation.
// Without the explicit attr, via_DMA=true would derive routing_mode=circuit.
// With explicit "packet" set, the result must be packet — explicit wins.
//
// CHECK-LABEL: module @rm_explicit_beats_via_dma
// CHECK: conduit.create @chan
// CHECK-SAME: routing_mode = #conduit.routing_mode<packet>
// CHECK-NOT: routing_mode = #conduit.routing_mode<circuit>
module @rm_explicit_beats_via_dma {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    aie.objectfifo @chan (%tile_0_2, {%tile_0_3}, 2 : i32)
        {routing_mode = "packet", via_DMA = true}
        : !aie.objectfifo<memref<16xi32>>
  }
}
