// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A → Pass C end-to-end test for the `aie.objectfifo.allocate` MemTile
// delegate path.
//
// Coverage gap (audit GAP D): existing test objectfifo_allocate_memtile.mlir
// exercises Pass A only — it confirms the scatter{N=1} relay is emitted but
// stops there.  Nothing verifies that the relay then lowers via Pass C to a
// real two-hop DMA path (shim → MemTile → consumer).  This test pins the
// end-to-end shape so a future change to either Pass A's relay synthesis
// or Pass C's scatter handling is caught at lit.
//
// Topology: shim(0,0) producer, compute(0,3) consumer, MemTile(0,1)
// allocate delegate.  Pass A rewrites `@of` to terminate at the MemTile and
// emits a scatter{src=@of, dsts=[@of_relay]} forwarding through the
// MemTile.  Pass C lowers the chain to: aie.flow shim→MemTile, an
// aie.memtile_dma block on the MemTile (S2MM ingest + MM2S relay), and an
// aie.flow MemTile→compute.

// CHECK-LABEL: module @objectfifo_allocate_memtile_passa_passc
// CHECK:   aie.device(npu1)

// Tiles must all survive the lowering.
// CHECK-DAG:   aie.tile(0, 0)
// CHECK-DAG:   aie.tile(0, 1)
// CHECK-DAG:   aie.tile(0, 3)

// --- Two flows: shim → MemTile, then MemTile → compute ---
// CHECK-DAG:   aie.flow(%{{.*}}shim{{.*}}, DMA :{{.*}}, %{{.*}}mem{{.*}}, DMA :
// CHECK-DAG:   aie.flow(%{{.*}}mem{{.*}}, DMA :{{.*}}, %{{.*}}tile_0_3{{.*}}, DMA :

// --- MemTile DMA block: ingest (S2MM) + relay-out (MM2S) ---
// CHECK:       aie.memtile_dma(%{{.*}}mem{{.*}}) {
// CHECK-DAG:     aie.dma_start(S2MM, 0,
// CHECK-DAG:     aie.dma_start(MM2S, 0,
// CHECK:         aie.end
// CHECK:       }

// --- All conduit + objectfifo ops fully lowered ---
// CHECK-NOT:   aie.objectfifo
// CHECK-NOT:   aie.objectfifo.allocate
// CHECK-NOT:   conduit.create
// CHECK-NOT:   conduit.scatter
// CHECK-NOT:   conduit.acquire
// CHECK-NOT:   conduit.release

module @objectfifo_allocate_memtile_passa_passc {
  aie.device(npu1) {
    %shim       = aie.tile(0, 0)
    %mem_tile   = aie.tile(0, 1)
    %tile_0_3   = aie.tile(0, 3)

    aie.objectfifo @of (%shim, {%tile_0_3}, 2 : i32)
        : !aie.objectfifo<memref<16xi32>>
    // MemTile delegate: forces a scatter{N=1} relay through the MemTile.
    aie.objectfifo.allocate @of (%mem_tile)

    aie.core(%tile_0_3) {
      %sub = aie.objectfifo.acquire @of (Consume, 1)
          : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %sub[0]
          : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      aie.objectfifo.release @of (Consume, 1)
      aie.end
    }
  }
}
