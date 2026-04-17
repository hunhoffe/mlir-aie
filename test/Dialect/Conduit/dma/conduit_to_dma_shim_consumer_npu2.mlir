// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass C test: compute→shim consumer conduit on npu2 (AIE2p, Strix).
//
// Exercises Phase 4b (routePhase) with an npu2 device:
//   - isAIE2Plus() = true for npu2 → shim-side locks emitted with init=0
//   - getNumDestShimMuxConnections() uses npu2 hardware model (budget=2)
//   - Shim lock naming: <name>_cons_prod_lock_0 / <name>_cons_cons_lock_0
//
// This complements conduit_to_dma_shim_lock_init.mlir (npu1) by confirming
// the same behavior on the npu2 (AIE2p) architecture.
//
// Topology: tile(0,2) [producer] → tile(0,0) [shim S2MM consumer]

// CHECK-LABEL: aie.device(npu2)

// --- Producer buffer and locks on tile(0,2) ---
// CHECK: aie.buffer(%{{.*}}tile_0_2) {sym_name = "of_buff_0"} : memref<32xi32>
// CHECK: aie.lock(%{{.*}}tile_0_2{{.*}}) {init = 1 : i32, sym_name = "of_prod_lock_0"}
// CHECK: aie.lock(%{{.*}}tile_0_2{{.*}}) {init = 0 : i32, sym_name = "of_cons_lock_0"}

// --- Shim DMA allocation ---
// CHECK: aie.shim_dma_allocation @of_shim_alloc(%{{.*}}tile_0_0, S2MM, 0)

// --- Shim-side locks: init=0 (AIE2p, host programs via npu.dma_memcpy_nd) ---
// CHECK: aie.lock(%{{.*}}tile_0_0{{.*}}) {init = 0 : i32, sym_name = "of_cons_prod_lock_0"
// CHECK: aie.lock(%{{.*}}tile_0_0{{.*}}) {init = 0 : i32, sym_name = "of_cons_cons_lock_0"

// --- Flow: tile(0,2) MM2S → tile(0,0) S2MM ---
// CHECK: aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_0, DMA : 0)

// --- Producer tile mem with MM2S BD chain ---
// CHECK: aie.mem(%{{.*}}tile_0_2) {
// CHECK:   aie.dma_start(MM2S, 0
// CHECK:   aie.use_lock(%{{.*}}of_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:   aie.dma_bd
// CHECK:   aie.use_lock(%{{.*}}of_prod_lock_0, Release, 1)

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @shim_consumer_npu2 {
  aie.device(npu2) {
    %shim  = aie.tile(0, 0)
    %tile2 = aie.tile(0, 2)

    // Compute tile produces data sent to shim (host-side DMA consumer).
    aie.objectfifo @of (%tile2, {%shim}, 1 : i32)
        : !aie.objectfifo<memref<32xi32>>

    %core = aie.core(%tile2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %sv = aie.objectfifo.acquire @of(Produce, 1)
                  : !aie.objectfifosubview<memref<32xi32>>
        aie.objectfifo.release @of(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%out: memref<128xi32>) {
      aiex.npu.dma_memcpy_nd (%out[0, 0, 0, 0][1, 1, 4, 32][0, 0, 32, 1])
          {metadata = @of, id = 0 : i64} : memref<128xi32>
      aiex.npu.dma_wait {symbol = @of}
    }
  }
}
