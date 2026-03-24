// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression test (A-7): SubviewAccess driven by WaitWindow must use the
// correct port (Produce or Consume) depending on which side of the conduit
// the enclosing CoreOp is on.
//
// Before the fix, the WaitWindow path hardcoded acquirePort = Port::Consume,
// so a producer-side core would use consumer buffers for the subview_access,
// yielding the wrong buffer reference in the lowered IR.
//
// This test has the PRODUCER-side core (tile_0_2) perform the async acquire +
// wait_window + subview_access pattern.  The conduit has producer_tile=0,2 and
// consumer_tile=0,0 (shim).  In the lowered IR, the subview_access on the
// produce side must reference the producer buffer (fifo_prod_buff_0), not the
// consumer buffer.
//
// Pass C does not allocate separate producer buffers for shim consumers
// (Case B path).  So this test uses a compute-to-compute conduit where
// explicit produce-side buffers exist.
//
// Topology: tile(0,2) [producer] → tile(0,4) [consumer], depth=1.
// Producer core: aie.core(tile_0_2) uses async acquire → wait_window →
//   subview_access → release{port=Produce}.
// Expected: the subview_access result in the producer core references the
//   producer-side buffer (fifo_prod_buff_0), not the consumer buffer.
//
// CHECK-LABEL: module @subview_wait_window_produce_side
// CHECK: aie.device
// CHECK:   %[[PROD_BUFF:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "fifo_prod_buff_0"
// CHECK:   aie.core(%{{.*}}tile_0_2) {
// CHECK:     aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:     func.call @produce(%[[PROD_BUFF]])
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire_async
// CHECK-NOT: conduit.wait_window
// CHECK-NOT: conduit.subview_access

module @subview_wait_window_produce_side {
  aie.device(npu1_1col) {
    func.func @produce(%buf: memref<8xi32>) -> () {
      return
    }
    func.func @consume(%buf: memref<8xi32>) -> () {
      return
    }

    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)

    conduit.create {name = "fifo", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 4>,
                    element_type = memref<8xi32>,
                    depth = 1 : i64}

    // Producer core: uses async acquire + wait_window + subview_access on
    // the produce side.
    %core_prod = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %tok = conduit.acquire_async {name = "fifo", count = 1 : i64}
                   : !conduit.window.token
        %win = conduit.wait_window %tok for "fifo"
                   : !conduit.window.token -> !conduit.window<memref<8xi32>>
        %elem = conduit.subview_access %win {index = 0 : i64}
                    : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @produce(%elem) : (memref<8xi32>) -> ()
        conduit.release %win {count = 1 : i64, port = #conduit.port<Produce>}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    // Consumer core (blocking acquire path).
    %core_cons = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %win = conduit.acquire {name = "fifo", count = 1 : i64,
                               port = #conduit.port<Consume>}
                   : !conduit.window<memref<8xi32>>
        %elem = conduit.subview_access %win {index = 0 : i64}
                    : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @consume(%elem) : (memref<8xi32>) -> ()
        conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
