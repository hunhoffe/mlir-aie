// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Pass A test: objectfifo with via_cascade=true.
//
// Verifies that aie.objectfifo with via_cascade=true is lowered to:
//   - conduit.create with routing_mode = #conduit.routing_mode<cascade> and depth = 1
//   - conduit.put_cascade in the producer core body
//     (acquire + memref.store + release → put_cascade with the stored vector)
//   - conduit.get_cascade in the consumer core body
//     (acquire + memref.load + release → get_cascade; load replaced by get result)
//
// Uses memref<1xvector<16xi32>> as the element type so the inner vector type
// (vector<16xi32> = 512 bits) matches the AIE2 cascade stream width.
// The verifier rejects cascade ops whose value type is not 384 or 512 bits.

// CHECK-LABEL: module

// --- Cascade conduit.create ---
// Attributes are printed alphabetically by MLIR's attr-dict printer.
// CHECK:   conduit.create
// CHECK-SAME: depth = 1 : i64
// CHECK-SAME: name = "cas_fifo"
// CHECK-SAME: routing_mode = #conduit.routing_mode<cascade>

// --- Producer core: acquire+store+release → put_cascade ---
// CHECK:   aie.core
// CHECK:     conduit.put_cascade "cas_fifo"
// CHECK-SAME:   vector<16xi32>
// CHECK-NOT:   conduit.acquire
// CHECK-NOT:   conduit.release

// --- Consumer core: acquire+load+release → get_cascade ---
// CHECK:   aie.core
// CHECK:     conduit.get_cascade "cas_fifo"
// CHECK-SAME:   vector<16xi32>

// No objectfifo ops remain.
// CHECK-NOT: aie.objectfifo

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    // Cascade objectfifo: depth=1, via_cascade=true.
    // Element type memref<1xvector<16xi32>>: element is vector<16xi32> = 512 bits (AIE2).
    aie.objectfifo @cas_fifo(%tile03, {%tile13}, 1 : i32) {via_cascade = true}
        : !aie.objectfifo<memref<1xvector<16xi32>>>

    // Producer core: stores a vector into the element buffer, then releases.
    // Pass A should pattern-match the store and emit put_cascade with the
    // stored value, then erase the store, subview, and acquire in order.
    aie.core(%tile03) {
      %subview = aie.objectfifo.acquire @cas_fifo(Produce, 1)
          : !aie.objectfifosubview<memref<1xvector<16xi32>>>
      %elem0 = aie.objectfifo.subview.access %subview[0]
          : !aie.objectfifosubview<memref<1xvector<16xi32>>> -> memref<1xvector<16xi32>>
      %c0 = arith.constant 0 : index
      %v = arith.constant dense<42> : vector<16xi32>
      memref.store %v, %elem0[%c0] : memref<1xvector<16xi32>>
      aie.objectfifo.release @cas_fifo(Produce, 1)
      aie.end
    }

    // Consumer core: acquires, loads from the element buffer, releases.
    // Pass A should emit get_cascade and replace the memref.load with the
    // cascade value, then erase the load, subview, and acquire.
    aie.core(%tile13) {
      %subview = aie.objectfifo.acquire @cas_fifo(Consume, 1)
          : !aie.objectfifosubview<memref<1xvector<16xi32>>>
      %elem0 = aie.objectfifo.subview.access %subview[0]
          : !aie.objectfifosubview<memref<1xvector<16xi32>>> -> memref<1xvector<16xi32>>
      %c0 = arith.constant 0 : index
      %r = memref.load %elem0[%c0] : memref<1xvector<16xi32>>
      // Use %r so it's not DCE'd.
      vector.print %r : vector<16xi32>
      aie.objectfifo.release @cas_fifo(Consume, 1)
      aie.end
    }
  }
}
