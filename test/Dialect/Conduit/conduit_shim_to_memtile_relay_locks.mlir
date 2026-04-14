// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression test: shim→MemTile relay (scatter) locks must be on the MemTile,
// NOT on the shim tile.
//
// Background:
//   For a conduit.scatter where the source conduit has a shim producer
//   (row==0) and a MemTile consumer (row==1), Phase 4a was incorrectly
//   allocating shimProdLock/shimConsLock on the shim tile (init=0 for both).
//   The correct behavior: no locks on the shim tile for this conduit name;
//   all synchronization locks belong on the MemTile (allocated by linkPhase
//   as per-destination slice locks, init=depth for prod, init=0 for cons).
//
// Topology:
//   shim(0,0) --[weights_stage1, depth=4]--> MemTile(0,1)
//   conduit.scatter: weights_stage1 → weights  (1-to-1 relay)
//   MemTile(0,1) --[weights, depth=4]--> compute(0,2)
//
// Expected locks:
//   - compute tile(0,2): weights_cons_prod_lock_0 (init=4), weights_cons_cons_lock_0 (init=0)
//   - MemTile(0,1): weights_stage1_link_prod_lock_0 (init=4), weights_stage1_link_cons_lock_0 (init=0)
//   - NO locks on shim tile (0,0) named "weights_stage1_prod_lock_0" or "weights_stage1_cons_lock_0"
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
// Flow shim→MemTile (emitted by Phase 4a).
// CHECK: aie.flow(%shim_noc_tile_0_0, DMA : 0, %mem_tile_0_1, DMA : 0)
//
// MemTile (0,1) link locks (emitted by linkPhase): init=4 (prod) and init=0 (cons).
// These must be on mem_tile_0_1, NOT on shim_noc_tile_0_0.
// CHECK: aie.lock(%mem_tile_0_1, {{[0-9]+}}) {init = 4 : i32, sym_name = "weights_stage1_link_prod_lock_0"
// CHECK: aie.lock(%mem_tile_0_1, {{[0-9]+}}) {init = 0 : i32, sym_name = "weights_stage1_link_cons_lock_0"
//
// Flow MemTile→compute (emitted by linkPhase).
// CHECK: aie.flow(%mem_tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
//
// No "weights_stage1_prod_lock_0" or "weights_stage1_cons_lock_0" anywhere
// (these were the erroneously allocated shim locks before the fix).
// CHECK-NOT: sym_name = "weights_stage1_prod_lock_0"
// CHECK-NOT: sym_name = "weights_stage1_cons_lock_0"

module @shim_to_memtile_relay_locks {
  aie.device(npu2) {
    %shim    = aie.tile(0, 0)
    %memtile = aie.tile(0, 1)
    %core    = aie.tile(0, 2)

    // Stage 1: shim→MemTile, depth=4 ring.
    conduit.create @weights_stage1 {
      slot_elems = 4 : i64,
      element_type = memref<1xi32>,
      depth = 4 : i64
    }

    // Stage 2: MemTile→compute, depth=4 ring.
    conduit.create @weights {
      slot_elems = 4 : i64,
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
