// RUN: aie-opt --conduit-check-ordering --verify-diagnostics %s
//
// DEFERRED-8: --conduit-check-ordering ambiguous ordering test.
//
// Two DMA-only channels with rate annotations share the same producer tile
// (0, 2).  In the CSDFa model, both channels' DMA events fire during the
// same actor firing — the relative ordering between them is undefined.
// The pass should emit a warning on the first channel's conduit.create.

module {
  // expected-warning @+1 {{conduit-check-ordering: ambiguous DMA event ordering}}
  conduit.create {name = "ch_a", capacity = 64 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 1, 2>,
                  producer_rates = array<i64: 1>,
                  consumer_rates = array<i64: 1>,
                  element_type = memref<64xi32>,
                  depth = 1 : i64}

  conduit.create {name = "ch_b", capacity = 64 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 2, 2>,
                  producer_rates = array<i64: 1>,
                  consumer_rates = array<i64: 1>,
                  element_type = memref<64xi32>,
                  depth = 1 : i64}

  func.func @producer() {
    %tok_a = conduit.put_memref_async {name = "ch_a", num_elems = 1 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 1>,
                 strides = array<i64: 1>} : !conduit.dma.token
    %tok_b = conduit.put_memref_async {name = "ch_b", num_elems = 1 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 1>,
                 strides = array<i64: 1>} : !conduit.dma.token
    conduit.wait_all %tok_a, %tok_b : !conduit.dma.token, !conduit.dma.token
    return
  }
}
