// RUN: aie-opt %s | FileCheck %s
//
// BUG: The Conduit verifier rejects conduit.acquire with prior_count == count,
// but this is a valid case: it means "all needed elements are already held from
// a prior acquire; no new lock grant is needed."
//
// In the sliding window tail pattern, after 6 middle iterations the core holds
// 2 rows.  The tail needs exactly those 2 rows — prior_count=2, count=2,
// delta=0.  Pass C should emit no AcquireGreaterEqual for this acquire.
//
// Current verifier error:
//   error: 'conduit.acquire' op prior_count (2) must be less than count (2)
//
// Expected (after fix):
//   The verifier allows prior_count == count.
//   Pass C emits no use_lock for this acquire (delta = 0).
//
// This test XFAIL until the verifier is fixed.
//
// XFAIL: *
//
// CHECK-LABEL: module @acquire_prior_count_equals_count
// CHECK: aie.core
// CHECK-NOT: use_lock(%{{.*}}, AcquireGreaterEqual, {{.*}})

module @acquire_prior_count_equals_count {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    conduit.create {name = "fifo", capacity = 512 : i64, depth = 4 : i64,
                    element_type = memref<128xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>}

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      // First acquire 2 elements.
      %win1 = conduit.acquire {name = "fifo", count = 2 : i64,
                                port = #conduit.port<Consume>}
                  : !conduit.window<memref<128xi32>>

      // Re-acquire same 2 elements (prior_count == count, delta = 0).
      // Semantics: "I already hold 2; I need 2; nothing to acquire from DMA."
      // BUG: verifier rejects prior_count (2) == count (2).
      %win2 = conduit.acquire {name = "fifo", count = 2 : i64,
                                prior_count = 2 : i64,
                                port = #conduit.port<Consume>}
                  : !conduit.window<memref<128xi32>>

      conduit.release %win2 {count = 2 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xi32>>

      aie.end
    }

    aie.runtime_sequence(%in: memref<256xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,256][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<256xi32>
    }
  }
}
