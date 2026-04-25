// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression test: shim→MemTile relay (distribute-link source) emits BOTH
// shim-side prod/cons locks AND memtile-side per-slice link locks.
//
// Background:
//   For a conduit.scatter where the source conduit has a shim producer
//   (row==0) and a MemTile consumer (row==1), Phase 4a allocates the shim-
//   side _prod_lock_0/_cons_lock_0 pair (init=0/init=0) for host-runtime
//   back-pressure of the shim S2MM → memtile hop, and linkPhase allocates
//   per-destination memtile slice locks for the memtile MM2S → compute hop.
//   Both are required: memtile slice locks throttle the wrong hop on their
//   own; without shim locks, all repeat_count + 1 BDs fire back-to-back
//   into the memtile S2MM port and compute reads stale data.
//
// Topology:
//   shim(0,0) --[weights_stage1, depth=4]--> MemTile(0,1)
//   conduit.scatter: weights_stage1 → weights  (1-to-1 relay)
//   MemTile(0,1) --[weights, depth=4]--> compute(0,2)
//
// Expected locks:
//   - compute tile(0,2): weights_cons_prod_lock_0 (init=4), weights_cons_cons_lock_0 (init=0)
//   - MemTile(0,1): weights_stage1_link_prod_lock_0 (init=4), weights_stage1_link_cons_lock_0 (init=0)
//   - shim tile(0,0): weights_stage1_prod_lock_0 (init=0), weights_stage1_cons_lock_0 (init=0)
//
// CHECK-LABEL: aie.device(npu2)
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.scatter
//
// Consumer tile (0,2) locks appear first (Phase 3 allocates them before Phase 5).
// prod_lock init=4, cons_lock init=0.
// CHECK: aie.lock(%tile_0_2, {{[0-9]+}}) {init = 4 : i32, sym_name = "weights_cons_prod_lock_0"
// CHECK: aie.lock(%tile_0_2, {{[0-9]+}}) {init = 0 : i32, sym_name = "weights_cons_cons_lock_0"
//
// Shim tile (0,0) prod/cons locks (emitted by Phase 4a): init=0 for both,
// programmed by host runtime aiex.npu.dma_memcpy_nd token signaling.
// CHECK: aie.lock(%shim_noc_tile_0_0, {{[0-9]+}}) {init = 0 : i32, sym_name = "weights_stage1_prod_lock_0"
// CHECK: aie.lock(%shim_noc_tile_0_0, {{[0-9]+}}) {init = 0 : i32, sym_name = "weights_stage1_cons_lock_0"
//
// Flow shim→MemTile (emitted by Phase 4a).
// CHECK: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
//
// MemTile (0,1) link locks (emitted by linkPhase): init=4 (prod) and init=0 (cons).
// CHECK: aie.lock(%mem_tile_0_1, {{[0-9]+}}) {init = 4 : i32, sym_name = "weights_stage1_link_prod_lock_0"
// CHECK: aie.lock(%mem_tile_0_1, {{[0-9]+}}) {init = 0 : i32, sym_name = "weights_stage1_link_cons_lock_0"
//
// Flow MemTile→compute (emitted by linkPhase).
// CHECK: aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)

module @shim_to_memtile_relay_locks {
  aie.device(npu2) {
    %shim    = aie.tile(0, 0)
    %memtile = aie.tile(0, 1)
    %core    = aie.tile(0, 2)

    // Stage 1: shim→MemTile, depth=4 ring.
    conduit.create @weights_stage1 {
            element_type = memref<1xi32>,
      depth = 4 : i64
    }

    // Stage 2: MemTile→compute, depth=4 ring.
    conduit.create @weights {
            element_type = memref<1xi32>,
      depth = 4 : i64
    }

    // Relay: scatter stage1 → weights (1-to-1 forward through MemTile).
    conduit.scatter{src = @weights_stage1, dsts = [@weights] {memtile = "tile(0,1)"}}

    // Shim DMA pre-declaration.
    aie.shim_dma_allocation @weights_stage1_shim_alloc(%shim, MM2S, 0)

    %core_body = aie.core(%core) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %i = %c0 to %c4 step %c1 {
        %win = conduit.acquire {name = @weights, count = 1 : i64,
                                port = #conduit.port<Consume>}
                 : !conduit.window<memref<1xi32>>
        conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<1xi32>>
      }

      aie.end
    } {dynamic_objfifo_lowering = true}

    aie.runtime_sequence(%arg_weights: memref<4xi32>) {
      aiex.npu.dma_memcpy_nd (%arg_weights[0,0,0,0][1,1,1,4][0,0,0,1])
        {metadata = @weights_stage1_shim_alloc, id = 0 : i64} : memref<4xi32>
    }
  }
}
