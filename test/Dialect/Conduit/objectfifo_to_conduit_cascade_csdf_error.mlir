// RUN: aie-opt --objectfifo-to-conduit --verify-diagnostics %s
//
// P0-C: negative test — cascade objectfifo with CSDF access pattern.
//
// A cascade stream is a hardware rendezvous register with no FIFO backing.
// Only a 1:1 (SDF) rate is physically possible; any CSDF pattern (varying
// acquire counts) produces deadlock or data corruption at runtime.
//
// Pass A must detect this via the Phase 1.5 CSDF scanner:
//   - Phase 1.5 scans all ObjectFifoAcquireOps for this fifo.
//   - If the consume-port acquire counts are not all equal, a CSDF
//     accessPattern is stored in FifoInfo.
//   - Phase 2 checks: if routing_mode would be "cascade" AND accessPattern
//     is non-empty, emit a hard error and fail the pass.
//
// The consumer core here has two acquire ops with counts 1 and 2 respectively,
// so Phase 1.5 records accessPattern = {1, 2} (non-uniform → CSDF detected).
//
// FileCheck: the expected-error annotation on the objectfifo op confirms the
// correct error message.  The surrounding SDF cascade objectfifo
// (objectfifo_to_conduit_cascade.mlir) must continue to pass unchanged.

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    // Element type: memref<1xvector<16xi32>> → inner type vector<16xi32> = 512 bits
    // (AIE2 cascade width). Depth = 1 satisfies the depth constraint.
    // CSDF pattern from the consumer core triggers the new guard.
    // expected-error @+1 {{cascade conduit requires SDF rate (1,1); CSDF patterns require buffering which cascade cannot provide}}
    aie.objectfifo @cas_csdf(%tile03, {%tile13}, 1 : i32) {via_cascade = true}
        : !aie.objectfifo<memref<1xvector<16xi32>>>

    // Producer: uniform acquire/release pattern (1,1) — not CSDF.
    aie.core(%tile03) {
      %sv0 = aie.objectfifo.acquire @cas_csdf(Produce, 1)
          : !aie.objectfifosubview<memref<1xvector<16xi32>>>
      %elem0 = aie.objectfifo.subview.access %sv0[0]
          : !aie.objectfifosubview<memref<1xvector<16xi32>>> -> memref<1xvector<16xi32>>
      %c0 = arith.constant 0 : index
      %v = arith.constant dense<42> : vector<16xi32>
      memref.store %v, %elem0[%c0] : memref<1xvector<16xi32>>
      aie.objectfifo.release @cas_csdf(Produce, 1)
      aie.end
    }

    // Consumer: CSDF pattern {1, 2} — two different acquire counts.
    // Phase 1.5 detects the non-uniform sequence and sets accessPattern = {1, 2}.
    aie.core(%tile13) {
      %c0 = arith.constant 0 : index

      // First iteration: acquire 1 element.
      %sv0 = aie.objectfifo.acquire @cas_csdf(Consume, 1)
          : !aie.objectfifosubview<memref<1xvector<16xi32>>>
      %elem0 = aie.objectfifo.subview.access %sv0[0]
          : !aie.objectfifosubview<memref<1xvector<16xi32>>> -> memref<1xvector<16xi32>>
      %r0 = memref.load %elem0[%c0] : memref<1xvector<16xi32>>
      vector.print %r0 : vector<16xi32>
      aie.objectfifo.release @cas_csdf(Consume, 1)

      // Second iteration: acquire 2 elements — makes this CSDF.
      %sv1 = aie.objectfifo.acquire @cas_csdf(Consume, 2)
          : !aie.objectfifosubview<memref<1xvector<16xi32>>>
      %elem1 = aie.objectfifo.subview.access %sv1[0]
          : !aie.objectfifosubview<memref<1xvector<16xi32>>> -> memref<1xvector<16xi32>>
      %r1 = memref.load %elem1[%c0] : memref<1xvector<16xi32>>
      vector.print %r1 : vector<16xi32>
      aie.objectfifo.release @cas_csdf(Consume, 2)

      aie.end
    }
  }
}
