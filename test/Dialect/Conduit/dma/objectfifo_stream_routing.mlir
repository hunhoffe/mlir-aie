// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s 2>&1 | FileCheck %s
//
// Regression test: aie_stream ObjectFIFO with link (stream_to_link pattern).
//
// The aie_stream attribute routes data from the producer core's AXI stream
// port directly into the consumer tile's DMA. No DMA engine, buffers, or
// locks are needed on the producer tile.
//
// Expected behavior:
//   - Conduit pipeline produces aie.flow(Core:0, ...) for the stream segment
//   - No buffers/locks on the producer tile (3,3)
//   - Buffers/locks on the MemTile (1,1) for the consumer side
//   - MemTile->Shim flow for the linked of_out conduit

// Buffers on MemTile:
// CHECK-DAG: aie.buffer({{.*}}) {sym_name = "of_stream_cons_buff_0"}
// CHECK-DAG: aie.buffer({{.*}}) {sym_name = "of_stream_cons_buff_1"}
// Stream flow uses Core wire bundle; link flow uses DMA wire bundle:
// CHECK-DAG: aie.flow({{.*}}, Core : 0, {{.*}}, DMA : 0)
// CHECK-DAG: aie.flow({{.*}}, DMA : 0, {{.*}}, DMA : 0)
// MemTile DMA with S2MM and MM2S:
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(S2MM
// CHECK: aie.dma_start(MM2S
// No residual conduit ops or objectfifos:
// CHECK-NOT: conduit.
// CHECK-NOT: aie.objectfifo

module @stream_to_link_AIE2 {
  aie.device(xcve2302) {
    %shim_pl_tile_1_0 = aie.tile(1, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_3_3 = aie.tile(3, 3)

    aie.objectfifo @of_stream (%tile_3_3, {%mem_tile_1_1}, 2 : i32) {aie_stream = 0 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out (%mem_tile_1_1, {%shim_pl_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.link [@of_stream] -> [@of_out] ([] [])

    %core_3_3 = aie.core(%tile_3_3) {
      %sv = aie.objectfifo.acquire @of_stream(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of_stream(Produce, 1)
      aie.end
    }
  }
}
