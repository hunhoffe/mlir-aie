// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --verify-diagnostics %s | FileCheck %s
//
// =============================================================================
// REGRESSION: when IRON explicitly emits `aiex.dma_configure_task_for
// {repeat_count = N}` for a shim-bearing channel, `--dma-task-to-conduit`
// stamps `dma_repeat = N` on the conduit.create (verbatim).
// =============================================================================
//
// HISTORY: this fixture originally pinned a CONFLICT path — it expected
// Pass A's `inferDmaRepeatForChannel` to stamp `dma_repeat = 4` on the
// conduit.create (because emit.count > 1), and then `--dma-task-to-conduit`
// to OVERRIDE that with IRON's explicit `repeat_count = 7`, emitting a
// remark documenting the override.  Per Task #40 / #42 (2026-04-28), Pass A
// no longer infers `dma_repeat` for ANY shim-bearing channel — the
// host-side `num_invocations` is invisible to the IR and the inference
// over-fires the shim BD by exactly that factor.  So the conflict path
// is gone (nothing for IRON to override), but the IRON-explicit
// surfacing path still must work — that's what this fixture now pins.
//
// Geometry (cribbed from passC_shim_bd_dma_repeat_uses_configure_task_repeat):
//   * Core loop trip count = 8; 2 configure_task emissions for @chan
//   * Pass A inference: SKIPPED (shim-bearing channel) — no stamp.
//   * IRON explicit:    repeat_count = 7
// Expected: `dma_repeat = 7` on the conduit.create (IRON-explicit value
// flows through unaltered).

// CHECK-LABEL: module @dma_task_to_conduit_iron_repeat_count_overrides_pass_a
module @dma_task_to_conduit_iron_repeat_count_overrides_pass_a {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (shim-bearing channel); deferring dma_repeat to runtime}}
    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<256xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      // 8 acquires total; 2 configure_task emissions below → emit.count = 2.
      // Post Task #42: Pass A SKIPS inference for any shim-bearing channel
      // (regardless of emit.count) since host-side num_invocations is
      // invisible.  IRON-explicit repeat_count surfaces unconditionally.
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
      // No override remark post Task #42: Pass A no longer stamps
      // dma_repeat on shim-bearing channels, so IRON's value lands fresh.
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
