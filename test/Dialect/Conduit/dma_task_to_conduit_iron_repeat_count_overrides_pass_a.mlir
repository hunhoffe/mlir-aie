// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --verify-diagnostics %s | FileCheck %s
//
// =============================================================================
// REGRESSION: when Pass A (--objectfifo-to-conduit) infers `dma_repeat = X`
// AND IRON explicitly emits `aiex.dma_configure_task_for {repeat_count = Y}`
// for the same channel, the IRON value WINS — `--dma-task-to-conduit`
// overwrites the Pass A stamp (verbatim) and emits a remark documenting the
// override.
// =============================================================================
//
// Rationale: IRON is closer to the source-of-truth about the runtime
// sequence's actual replay count.  Pass A's inference is a derived quantity
// from the core-side acquire count and the per-channel emit count; the
// IRON-emitted attribute is the authored-by-the-operator-author value.
// When both are present, defer to IRON.
//
// Geometry (cribbed from passC_shim_bd_dma_repeat_uses_configure_task_repeat):
//   * Core loop trip count = 8; 2 configure_task emissions for @chan
//   * Pass A inferDmaRepeatForChannel = (8 / 2) / 1 = 4
//   * IRON explicit: repeat_count = 7  (Y != X to make the override visible)
// Expected: dma_repeat = 7 on the conduit.create (IRON wins).
//
// Note: the remark fires once per IRON-emitted configure_task that triggers
// an override.  With two emissions both setting repeat_count = 7, the second
// is a no-op overwrite (both sides equal), so only the FIRST one fires the
// remark.

// CHECK-LABEL: module @dma_task_to_conduit_iron_repeat_count_overrides_pass_a
module @dma_task_to_conduit_iron_repeat_count_overrides_pass_a {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<256xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      // 8 acquires total; 2 configure_task emissions below → emit.count = 2.
      // Pass A would infer dma_repeat = (8 / 2) / 1 = 4.
      scf.for %i = %c0 to %c8 step %c1 {
        %sub = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<256xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
        aie.objectfifo.release @chan (Consume, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%a0: memref<256xbf16>) {
      // expected-remark @below {{dma-task-to-conduit: IRON explicit repeat_count = 7 on @chan overrides Pass A inferred dma_repeat = 4}}
      %t0 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%a0 : memref<256xbf16>, 0, 256,
            [<size = 1, stride = 0>,
             <size = 1, stride = 0>,
             <size = 1, stride = 0>,
             <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_free_task(%t0)
      %t1 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%a0 : memref<256xbf16>, 0, 256,
            [<size = 1, stride = 0>,
             <size = 1, stride = 0>,
             <size = 1, stride = 0>,
             <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t1)
    }
  }
}

// IRON-explicit value (7) wins over Pass A inference (4).
// CHECK:       conduit.create @chan
// CHECK-SAME:  dma_repeat = 7
