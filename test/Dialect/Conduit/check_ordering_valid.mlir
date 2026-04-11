// RUN: aie-opt --conduit-check-ordering %s 2>&1 | FileCheck %s
//
// DEFERRED-8: --conduit-check-ordering valid test.
//
// Simple P=1, C=1 SPSC chain — single DMA-only channel per tile.
// Since no tile has two DMA-only rated channels, the ordering is total
// and no warning should be emitted.
//
// CHECK-NOT: warning
// CHECK-NOT: error
// CHECK: module

module {
  aie.device(npu1) {
    conduit.create @ch_single {slot_elems = 64 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 2>,
                    producer_rates = array<i64: 1>,
                    consumer_rates = array<i64: 1>,
                    element_type = memref<64xi32>,
                    depth = 1 : i64}

    func.func @producer() {
      %tok = conduit.put_memref_async {name = @ch_single, num_elems = 1 : i64,
                   offsets = array<i64: 0>, sizes = array<i64: 1>,
                   strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %tok : !conduit.dma.token
      return
    }
  }
}
