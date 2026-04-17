// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test:  on an objectfifo.link SOURCE
// (the distribute-source side of a 1→N link).
//
// When the source conduit has disable_synchronization, the per-destination
// MemTile slice locks are skipped (ConduitToDMALink.cpp line 154:
// `if (isDistribute && numDsts > 0 && !srcInfo.disableSynchronization)`).
// BD chains on the MemTile must still be emitted — only locks are absent.
//
// This is the COMPLEMENT to objectfifo_disable_sync_link.mlir which tests
// disable_sync on the JOIN destination. This test exercises the distribute
// source path.
//
// Topology:
//   shim(0,0) → link_src(MemTile, 0,1) → {tile_a(0,2), tile_b(0,3)}
//   link_src has 
//   link_dst_a and link_dst_b are regular (synchronized) fifos
//
// Expected:
//   - No MemTile slice locks named link_src_* (skipped due to disable_sync)
//   - link_dst_a and link_dst_b locks ARE emitted (normal fifos)
//   - MemTile DMA BD chains still present (no use_lock in MemTile BDs)
//   - Compute tile aie.mem blocks have use_lock for dst fifos

// CHECK-LABEL: module @disable_sync_distribute_src

// --- Destination fifo locks must be emitted (regular, synchronized) ---
// CHECK: aie.lock({{.*}}) {{{.*}}sym_name = "link_dst_b_cons_prod_lock_0"
// CHECK: aie.lock({{.*}}) {{{.*}}sym_name = "link_dst_b_cons_cons_lock_0"
// CHECK: aie.lock({{.*}}) {{{.*}}sym_name = "link_dst_a_cons_prod_lock_0"
// CHECK: aie.lock({{.*}}) {{{.*}}sym_name = "link_dst_a_cons_cons_lock_0"

// --- No MemTile slice locks for the disabled source ---
// Slice locks would be named link_src_prod_lock_0 / link_src_cons_lock_0.
// CHECK-NOT: link_src_prod_lock
// CHECK-NOT: link_src_cons_lock

// --- Flows must still be emitted (MemTile relay) ---
// CHECK: aie.flow

// --- MemTile DMA must be emitted with BD chains but NO use_lock ---
// The disable_synchronization source means the MemTile S2MM and MM2S BDs
// have no AcquireGreaterEqual / Release use_lock ops.
// CHECK: aie.memtile_dma(
// CHECK:   aie.dma_start
// CHECK:   aie.dma_bd
// CHECK-NOT: aie.use_lock
// CHECK:   aie.dma_start
// CHECK:   aie.dma_bd
// CHECK-NOT: aie.use_lock

// --- Compute tile mem blocks DO have use_lock for their dst fifos ---
// These are synchronized (no disable_sync on link_dst_a/b).
// CHECK: aie.mem({{.*}}tile_0_2
// CHECK:   aie.use_lock

module @disable_sync_distribute_src {
  aie.device(npu1_1col) {
    %shim   = aie.tile(0, 0)
    %mem    = aie.tile(0, 1)
    %tile_a = aie.tile(0, 2)
    %tile_b = aie.tile(0, 3)

    // Source: shim → MemTile on the source.
    aie.objectfifo @link_src (%shim, {%mem}, 1 : i32)
        {disable_synchronization = true}
        : !aie.objectfifo<memref<32xi32>>

    // Destinations: MemTile → compute tiles (regular, synchronized).
    aie.objectfifo @link_dst_a (%mem, {%tile_a}, 1 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @link_dst_b (%mem, {%tile_b}, 1 : i32)
        : !aie.objectfifo<memref<16xi32>>

    // Distribute link: 1 source → 2 destinations.
    aie.objectfifo.link [@link_src] -> [@link_dst_a, @link_dst_b] ([][0, 16])

    %core_a = aie.core(%tile_a) {
      %sv = aie.objectfifo.acquire @link_dst_a(Consume, 1)
                : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @link_dst_a(Consume, 1)
      aie.end
    }

    %core_b = aie.core(%tile_b) {
      %sv = aie.objectfifo.acquire @link_dst_b(Consume, 1)
                : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @link_dst_b(Consume, 1)
      aie.end
    }
  }
}
