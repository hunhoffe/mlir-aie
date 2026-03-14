// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C test: conduit.create with routing_mode = "packet" should produce
// aie.packet_flow instead of aie.flow for the shim → consumer connection.
//
// This test exercises the routing_mode attribute added in Step 4 of the
// packet switching feature.  The conduit is declared with routing_mode =
// "packet"; Pass C Phase 4 checks this attribute and emits aie.packet_flow
// (with aie.packet_source and aie.packet_dest in its region) rather than
// the default aie.flow.
//
// Resources expected (depth=1, shim tile_0_0 → tile_0_2):
//   aie.buffer:        1  (pkt_fifo_cons_buff_0 on tile_0_2)
//   aie.lock:          4  (cons prod_lock init=1, cons cons_lock init=0 on
//                          tile_0_2; prod_lock, cons_lock on shim tile_0_0)
//   aie.packet_flow:   1  (shim DMA:0 → tile_0_2 DMA:0, with packet ID 0)
//   aie.flow:          0  (must NOT appear — routing_mode = "packet")
//   aie.mem:           1  (S2MM for tile_0_2)

// CHECK-LABEL: module @packet_routing_mode
// CHECK:   aie.device(npu1_1col) {
// CHECK:     aie.tile(0, 0)
// CHECK:     aie.tile(0, 2)
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "pkt_fifo{{.*}}buff_0"
// CHECK:     aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 1
// CHECK:     aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK:     aie.core(%{{.*}}tile_0_2)
//
// --- packet_flow must appear instead of aie.flow ---
// CHECK:     aie.packet_flow
// CHECK-NOT: aie.flow(
//
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM
// CHECK-NOT: conduit.create

module @packet_routing_mode {
  aie.device(npu1_1col) {
    func.func @use_data(%buf: memref<10xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Conduit channel with routing_mode = "packet".
    // Pass C Phase 4 should emit aie.packet_flow instead of aie.flow.
    conduit.create {name = "pkt_fifo", capacity = 10 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<10xi32>,
                    depth = 1 : i64,
                    routing_mode = "packet"}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %arg0 = %c0 to %c4 step %c1 {
        %win = conduit.acquire {name = "pkt_fifo", count = 1 : i64,
                                port = "Consume"}
                   : !conduit.window<memref<10xi32>>
        %elem = conduit.subview_access %win {index = 0 : i64}
                    : !conduit.window<memref<10xi32>> -> memref<10xi32>
        func.call @use_data(%elem) : (memref<10xi32>) -> ()
        conduit.release %win {count = 1 : i64, port = "Consume"}
            : !conduit.window<memref<10xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
