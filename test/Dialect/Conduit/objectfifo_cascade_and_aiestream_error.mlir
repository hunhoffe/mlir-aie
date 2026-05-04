// RUN: not aie-opt --objectfifo-to-conduit %s 2>&1 | FileCheck %s
//
// Regression test: B-8 — objectfifo with both via_cascade=true and aie_stream
// routing is a mutually exclusive combination that must produce a hard error.
//
// Before the B-8 fix, the aie_stream branch silently overwrote the "cascade"
// routing_mode with "stream", resulting in a stream conduit that was missing
// the cascade hardware configuration — silent semantic corruption.
//
// CHECK: objectfifo-to-conduit: objectfifo 'bad_fifo' has both via_cascade=true and aie_stream routing
// CHECK: mutually exclusive

module @objectfifo_cascade_and_aiestream_error {
  aie.device(npu1_1col) {
    %prod_tile = aie.tile(0, 2)
    %cons_tile = aie.tile(0, 3)

    // This objectfifo has via_cascade=true AND aie_stream=0 (aie_stream on
    // producer tile). aie_stream=0 puts the fifo into aieStreamFifoPort map.
    // The combination is semantically impossible — reject it.
    aie.objectfifo @bad_fifo(%prod_tile, {%cons_tile}, 1 : i32)
        {via_cascade = true, aie_stream = 0 : i32, aie_stream_port = 0 : i32} : !aie.objectfifo<memref<16xi32>>

    aie.core(%prod_tile) {
      aie.end
    }
    aie.core(%cons_tile) {
      aie.end
    }
  }
}
