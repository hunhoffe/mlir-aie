// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C test: conduit with BOTH a compute consumer AND a shim consumer.
//
// This exercises the Phase 4b (routePhase) multiConsumer=true path:
//   - (compute) AND shim_consumer_tiles=[0,0] (shim)
//     → multiConsumer=true → shim lock names use indexed suffix "_cons_1"
//       (globalConsIdx = numComputeConsumers=1 + shimConsIdx=0 = 1)
//   - Phase 4b allocates the MM2S channel and records it in conduitMM2SChannel.
//   - Phase 4.5a reuses that same MM2S channel for the compute-consumer flow,
//     broadcasting from one physical MM2S port to both consumers.
//
// Topology: tile(0,2) [producer] → { tile(0,3) [compute, index 0],
//                                     tile(0,0) [shim,    index 1] }
// Device: npu1_1col (AIE2)

// CHECK-LABEL: aie.device(npu1_1col)

// --- Producer buffer and locks on tile(0,2) ---
// CHECK: aie.buffer(%{{.*}}tile_0_2) {sym_name = "chan_buff_0"} : memref<16xi32>
// CHECK: aie.lock(%{{.*}}tile_0_2{{.*}}) {init = 1 : i32, sym_name = "chan_prod_lock_0"}
// CHECK: aie.lock(%{{.*}}tile_0_2{{.*}}) {init = 0 : i32, sym_name = "chan_cons_lock_0"}

// --- Compute consumer buffer and locks on tile(0,3), suffix _cons_0 ---
// CHECK: aie.buffer(%{{.*}}tile_0_3) {sym_name = "chan_cons_0_buff_0"} : memref<16xi32>
// CHECK: aie.lock(%{{.*}}tile_0_3{{.*}}) {{{.*}}sym_name = "chan_cons_0_prod_lock_0"
// CHECK: aie.lock(%{{.*}}tile_0_3{{.*}}) {{{.*}}sym_name = "chan_cons_0_cons_lock_0"

// --- ShimDMAAllocation for shim consumer (emitted before shim locks) ---
// CHECK: aie.shim_dma_allocation @chan_shim_alloc

// --- Shim consumer locks on tile(0,0), suffix _cons_1 (globalConsIdx=1) ---
// Phase 4b emits these after the cores.
// CHECK: aie.lock(%{{.*}}tile_0_0{{.*}}) {{{.*}}sym_name = "chan_cons_1_prod_lock_0"
// CHECK: aie.lock(%{{.*}}tile_0_0{{.*}}) {{{.*}}sym_name = "chan_cons_1_cons_lock_0"

// --- Flows: same MM2S channel (0) on producer to BOTH consumers ---
// Phase 4b assigns channel 0 for shim consumer first; Phase 4.5a reuses it.
// CHECK: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_0, DMA : 0)
// CHECK: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_3, DMA : 0)

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @multi_consumer_shim {
  aie.device(npu1_1col) {
    %shim   = aie.tile(0, 0)
    %tile_2 = aie.tile(0, 2)  // producer
    %tile_3 = aie.tile(0, 3)  // compute consumer

    // One conduit: compute producer → both compute and shim consumers.
    conduit.create @chan {depth = 1 : i64,
                                        element_type = memref<16xi32>,
                    shim_consumer_tiles = array<i64: 0, 0>,
                                        routing_mode = #conduit.routing_mode<circuit>}

    aie.shim_dma_allocation @chan_shim_alloc(%shim, S2MM, 0)

    // Producer core: fill one element and signal done.
    %c_prod = aie.core(%tile_2) {
      %win = conduit.acquire {name = @chan, count = 1 : i64,
                               port = #conduit.port<Produce>}
                 : !conduit.window<memref<16xi32>>
      conduit.release %win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<16xi32>>
      aie.end
    }

    // Compute consumer core: read one element.
    %c_cons = aie.core(%tile_3) {
      %win = conduit.acquire {name = @chan, count = 1 : i64,
                               port = #conduit.port<Consume>}
                 : !conduit.window<memref<16xi32>>
      conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<16xi32>>
      aie.end
    }

    aie.runtime_sequence(%out: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd (%out[0,0,0,0][1,1,1,16][0,0,0,1])
          {metadata = @chan_shim_alloc, id = 0 : i64} : memref<16xi32>
    }
  }
}
