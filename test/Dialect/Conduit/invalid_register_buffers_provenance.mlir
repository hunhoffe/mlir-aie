// RUN: aie-opt --verify-diagnostics -split-input-file %s
//
// Negative test: conduit.register_buffers provenance check.
//
// Buffer operands must be defined by aie.buffer or aie.external_buffer.
// A buffer from memref.alloc (or any other op) must be rejected.

// -----

//===----------------------------------------------------------------------===//
// FAILING: buffer comes from memref.alloc — not aie.buffer/aie.external_buffer.
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    %tile02 = aie.tile(0, 2)

    conduit.create @chan {slot_elems = 32 : i64,
                    element_type = memref<32xi32>,
                    depth = 1 : i64}

    aie.core(%tile02) {
      %bad = memref.alloc() : memref<32xi32>
      // expected-error @+1 {{'conduit.register_buffers' op buffer operand must be defined by aie.buffer or aie.external_buffer, got memref.alloc}}
      conduit.register_buffers {name = @chan, buffers = [%bad]} : memref<32xi32>
      aie.end
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// PASSING: buffer comes from aie.buffer — valid provenance.
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    %tile02 = aie.tile(0, 2)

    conduit.create @chan2 {slot_elems = 32 : i64,
                    element_type = memref<32xi32>,
                    depth = 1 : i64}

    %good = aie.buffer(%tile02) : memref<32xi32>

    aie.core(%tile02) {
      conduit.register_buffers {name = @chan2, buffers = [%good]} : memref<32xi32>
      aie.end
    }
  }
}
