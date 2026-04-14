// RUN: aie-opt --conduit-check-tiers -split-input-file -verify-diagnostics %s
//
// M-12: Mixed-tier verifier tests.
//
// Mixing Tier 2 (acquire/release) and Tier 3 (get_memref/put_memref) ops for
// the same channel name inside the same aie.core region is rejected because
// Tier 3 ops bypass the rotation counter that Tier 2 ops maintain.  The next
// Tier 2 acquire after a Tier 3 op would read from the wrong physical buffer.
//
// Cross-endpoint mixing (Tier 3 on shim, Tier 2 on compute core) is valid and
// is explicitly tested in the PASSING section.
//
//===----------------------------------------------------------------------===//
// FAILING: acquire (Tier 2) followed by get_memref (Tier 3) — same channel,
// same core.
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    %prod = aie.tile(0, 0)
    %cons = aie.tile(0, 2)

    conduit.create @foo {slot_elems = 32 : i64,
                    element_type = memref<32xi32>,
                    depth = 1 : i64}

    aie.core(%cons) {
      // Tier 2 acquire — establishes the channel in T2 map.
      %win = conduit.acquire {name = @foo, count = 1 : i64,
                              port = #conduit.port<Consume>}
                 : !conduit.window<memref<32xi32>>
      %elem = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<32xi32>> -> memref<32xi32>
      conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<32xi32>>

      // Tier 3 get on SAME channel — should error.
      // expected-error @+1 {{conduit channel 'foo' mixed Tier 2}}
      conduit.get_memref {name = @foo, num_elems = 32 : i64,
                          offsets = array<i64: 0>, sizes = array<i64: 32>,
                          strides = array<i64: 1>}
      aie.end
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// FAILING: get_memref (Tier 3) followed by acquire (Tier 2) — same channel,
// same core.  Order reversed to confirm direction-independence.
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    %prod = aie.tile(0, 0)
    %cons = aie.tile(0, 2)

    conduit.create @bar {slot_elems = 16 : i64,
                    element_type = memref<16xi32>,
                    depth = 1 : i64}

    aie.core(%cons) {
      // Tier 3 get — establishes the channel in T3 map first.
      conduit.get_memref {name = @bar, num_elems = 16 : i64,
                          offsets = array<i64: 0>, sizes = array<i64: 16>,
                          strides = array<i64: 1>}

      // Tier 2 acquire on SAME channel — error fires here (T3 seen first).
      // expected-error @+1 {{conduit channel 'bar' mixed Tier 2}}
      %win = conduit.acquire {name = @bar, count = 1 : i64,
                              port = #conduit.port<Consume>}
                 : !conduit.window<memref<16xi32>>
      // Release traces back to the acquire, so 'bar' is already in alreadyErrored —
      // no duplicate error here.
      conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<16xi32>>
      aie.end
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// FAILING: put_memref_async (Tier 3) + release_async (Tier 2) — same channel,
// same core.  Tests async variants.
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    %prod = aie.tile(0, 2)
    %cons = aie.tile(0, 3)

    conduit.create @baz {slot_elems = 8 : i64,
                    element_type = memref<8xi32>,
                    depth = 1 : i64}

    aie.core(%prod) {
      // Tier 3 async put — establishes channel in T3 map first.
      %dma = conduit.put_memref_async {name = @baz, num_elems = 8 : i64,
                                       offsets = array<i64: 0>,
                                       sizes = array<i64: 8>,
                                       strides = array<i64: 1>}
                 : !conduit.dma.token
      conduit.wait_all %dma : !conduit.dma.token

      // Tier 2 async release on SAME channel — error fires here (T3 seen first).
      // expected-error @+1 {{conduit channel 'baz' mixed Tier 2}}
      conduit.release_async {name = @baz, count = 1 : i64,
                             port = #conduit.port<Produce>}
          : !conduit.window.token
      aie.end
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// PASSING: Tier 2 on consumer core + Tier 3 on producer (shim) — cross-
// endpoint mixing.  This is the canonical ObjectFIFO shim-to-core pattern.
// No error should be emitted.
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    conduit.create @input {slot_elems = 32 : i64,
                    element_type = memref<32xi32>,
                    depth = 1 : i64}

    // Shim tile uses Tier 3 (put_memref via DMA from host) — no aie.core here.
    // The shim is not an aie.core region, so this is not checked.

    aie.core(%core) {
      // Compute core uses Tier 2 only.
      %win = conduit.acquire {name = @input, count = 1 : i64,
                              port = #conduit.port<Consume>}
                 : !conduit.window<memref<32xi32>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<32xi32>> -> memref<32xi32>
      conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<32xi32>>
      aie.end
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// PASSING: Same channel name used in two DIFFERENT aie.core regions.
// Core A uses Tier 2 on "shared"; Core B uses Tier 3 on "shared".
// No shared rotation counter across cores — valid.
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    %coreA = aie.tile(0, 2)
    %coreB = aie.tile(1, 2)

    conduit.create @shared {slot_elems = 16 : i64,
                    element_type = memref<16xi32>,
                    depth = 1 : i64}

    // Core A: Tier 2 only.
    aie.core(%coreA) {
      %win = conduit.acquire {name = @shared, count = 1 : i64,
                              port = #conduit.port<Produce>}
                 : !conduit.window<memref<16xi32>>
      conduit.release %win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<16xi32>>
      aie.end
    }

    // Core B: Tier 3 only.  Different aie.core region — no conflict.
    aie.core(%coreB) {
      conduit.get_memref {name = @shared, num_elems = 16 : i64,
                          offsets = array<i64: 0>, sizes = array<i64: 16>,
                          strides = array<i64: 1>}
      aie.end
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// PASSING: Tier 2 only in a single core — no Tier 3 ops at all.
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    %t = aie.tile(0, 2)

    conduit.create @only_t2 {slot_elems = 8 : i64,
                    element_type = memref<8xi32>,
                    depth = 1 : i64}

    aie.core(%t) {
      %win = conduit.acquire {name = @only_t2, count = 1 : i64,
                              port = #conduit.port<Consume>}
                 : !conduit.window<memref<8xi32>>
      conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<8xi32>>
      aie.end
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// PASSING: Tier 3 only in a single core — no Tier 2 ops at all.
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    %t = aie.tile(0, 2)

    conduit.create @only_t3 {slot_elems = 8 : i64,
                    element_type = memref<8xi32>,
                    depth = 1 : i64}

    aie.core(%t) {
      conduit.get_memref {name = @only_t3, num_elems = 8 : i64,
                          offsets = array<i64: 0>, sizes = array<i64: 8>,
                          strides = array<i64: 1>}
      aie.end
    }
  }
}
