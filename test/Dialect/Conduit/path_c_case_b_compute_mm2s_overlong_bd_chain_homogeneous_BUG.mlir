// RUN: aie-opt --verify-diagnostics --conduit-to-dma %s
// RUN: aie-opt --verify-diagnostics --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// Pass C Case B (compute MM2S → shim) overlong-BD-chain BUG pin (homogeneous).
//
// Bug location: ConduitToDMALink.cpp Case B, lines 2030-2202.
//   * L2037: caseBBdRepeat = info.bdRepeat > 1 ? info.bdRepeat : 1;
//   * L2038: caseBEffectiveBDs = info.nConsumerBuffers() * caseBBdRepeat;
//   * L2060/L2139: emit caseBEffectiveBDs sequential aie.dma_bd blocks
//                  inside aie.mem on the compute tile.
//   * L2123/L2198: NextBD chain is `bdBlocks[(i+1) % caseBEffectiveBDs]`
//                  — always-circular, no terminator, no rotation cap.
//   * NO targetModel.getNumBDs(...) cap before block emission.
//
// Root cause: nConsumerBuffers() returns putCount when putCount > 1 and
//   dmaRepeat == 0 (Common.h:315-326).  putCount is a *host-emit accounting*
//   metric (count of conduit.put_memref_async ops on this channel) — it
//   should not influence compute-tile BD-emit decisions.  Same risk class
//   as today's HIGH bug at Case A (Link.cpp:2369-2370): putCount leaking
//   into hardware-tile BD-chain length without a HW cap.
//
// On AIE2 compute tile, getNumBDs(...) = 16.  Any putCount > 16 produces
// > 16 aie.dma_bd blocks → AIEDialect.cpp:328-343 HasValidBDs verifier
// rejects with `'aie.mem' op has more than 16 blocks`.
//
// Fixture geometry:
//   * compute(0,2) → shim(0,0) via @chan
//   * @chan: depth = 2, memref<8xi32>; conduit.create authored directly
//     (post-Pass-A IR; no objectfifo).
//   * aie.shim_dma_allocation @chan(%shim, S2MM, 0) — shim is consumer.
//     Drives routePhase to populate info.shimConsumerTileCoords (compute
//     producer + empty consumerTileCoords + non-empty shimConsumerTileCoords
//     = Case B trigger at Link.cpp:2002-2003).
//   * Compute core: scf.for trip=∞ conduit.acquire/release(Produce, 1).
//   * 17 hand-authored conduit.put_memref_async ops on @chan in a regular
//     func.func (NOT aie.runtime_sequence — keeps Pass C's runtime-step
//     out of the path).  All 17 are STRUCTURALLY IDENTICAL (same offsets,
//     sizes, strides, num_elems, no producer_dimensions).  Each is paired
//     with conduit.wait_all{token=true} (await) + conduit.wait_all{token=
//     false} (free) — matching the `--conduit-canonicalize-loop-unroll-puts`
//     match shape.
//
// Why hand-authored puts on a compute→shim channel:
//   For compute→shim channels, runtime aiex.dma_configure_task_for ops
//   become S2MM (shim receives) → conduit.get_memref_async after
//   --dma-task-to-conduit, NOT puts.  putCount > 1 is therefore not
//   reachable from IRON's natural compile path today — but IS reachable
//   via (a) --conduit-fuse-channels merging multiple compute→shim
//   channels into one canonical name, or (b) future canon symmetric
//   collapse / expansion patterns.  The hand-authored shape pins the
//   structural latent bug at the LINK pass independent of how the state
//   arises upstream.
//
// Today's behavior (this RUN line, with canon ABSENT from pipeline):
//   Pass C emits 17 sequential aie.dma_bd blocks in aie.mem on compute(0,2).
//   AIE verifier rejects with `'aie.mem' op has more than 16 blocks`.
//
// Post-canon-stabilization flip plan (homogeneous variant):
//   --conduit-canonicalize-loop-unroll-puts (Task #11) collapses the 17
//   structurally-identical puts → 1 put + dma_repeat = 17 on conduit.create.
//   With canon in pipeline, putCount drops to 1 and dmaRepeat = 17 → Case B
//   takes nConsumerBuffers() = depth = 2 path; caseBBdRepeat = 1; emits
//   2 BDs (depth-many circular chain).  Compute tile DMAs cycle infinitely
//   (repeat_count = 0), matching upstream stateful's compute-tile rotation.
//   Flip this fixture (rename, drop _BUG suffix) to FileCheck pinning:
//     CHECK: conduit.create @chan
//     CHECK-SAME: dma_repeat = 17
//     CHECK: aie.mem(%{{.*}}tile_0_2)
//     CHECK:   aie.dma_start(MM2S,
//     CHECK:   aie.dma_bd
//     CHECK:   aie.next_bd
//     CHECK:   aie.dma_bd
//     CHECK:   aie.next_bd
//     CHECK-NOT: aie.dma_bd
//     CHECK:   aie.end
//
// (See companion `..._heterogeneous_BUG.mlir` — canon refuses on
// non-identical puts, so that variant pins the cap-helper need; Task #15.)
//
// Sprint N+2 Tier 0 status (2026-04-30):
//   Tier 0 only adds a loop-context discriminator to nConsumerBuffers() in
//   ConduitToDMACommon.h so the putCount override is suppressed when the
//   consumer gets are inside a loop (op7/op11 GEMV pattern, fixing the Case A
//   consumer S2MM crash).  Case B is the PRODUCER side (compute MM2S) — its
//   chain length comes from `caseBEffectiveBDs = nConsumerBuffers() *
//   caseBBdRepeat` at Link.cpp L2038, which still over-fires when the
//   consumer is a memtile/shim with NO consumer-side gets at all (so
//   consumerGetsInLoop = false → override fires → BD chain = putCount).
//   Tier 0 therefore does NOT fix this BUG pin.  Flip → `_canonical.mlir`
//   only after canon (`--conduit-canonicalize-channel-puts`) lands in the
//   Pass C pre-pipeline AND a per-call-site cap helper (Task #15) is
//   applied at Case B's BD-emit sites.

module @path_c_case_b_compute_mm2s_overlong_bd_chain_homogeneous {
  // expected-error@+1 {{conduit-to-dma: BD chain length}}
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @chan {
      element_type = memref<8xi32>,
      depth = 2 : i64
    }

    // Shim is consumer (S2MM) — makes compute(0,2) the producer.
    aie.shim_dma_allocation @chan_shim_alloc(%tile_0_0, S2MM, 0) {conduit_channel = @chan}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %win = conduit.acquire {name = @chan, port = #conduit.port<Produce>, count = 1 : i64}
            : !conduit.window<memref<8xi32>>
        conduit.release %win {port = #conduit.port<Produce>, count = 1 : i64}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    }

    // 17 structurally-identical puts on @chan.  All offsets = 0; all sizes = 8;
    // all strides = 1; all num_elems = 8 — IRON-loop-unroll match shape that
    // --conduit-canonicalize-loop-unroll-puts WOULD collapse if it were in
    // the pipeline (it is not, on this RUN line).  Each paired with
    // wait_all{token=true} (await) + wait_all{token=false} (free) so the
    // shape matches the canon match predicate exactly.
    func.func @sequence(%arg0: memref<8xi32>) {
      %t0 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token
      %t1 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token
      %t2 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token
      %t3 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t3 {token = true} : !conduit.dma.token
      conduit.wait_all %t3 {token = false} : !conduit.dma.token
      %t4 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t4 {token = true} : !conduit.dma.token
      conduit.wait_all %t4 {token = false} : !conduit.dma.token
      %t5 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t5 {token = true} : !conduit.dma.token
      conduit.wait_all %t5 {token = false} : !conduit.dma.token
      %t6 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t6 {token = true} : !conduit.dma.token
      conduit.wait_all %t6 {token = false} : !conduit.dma.token
      %t7 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t7 {token = true} : !conduit.dma.token
      conduit.wait_all %t7 {token = false} : !conduit.dma.token
      %t8 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t8 {token = true} : !conduit.dma.token
      conduit.wait_all %t8 {token = false} : !conduit.dma.token
      %t9 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t9 {token = true} : !conduit.dma.token
      conduit.wait_all %t9 {token = false} : !conduit.dma.token
      %t10 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t10 {token = true} : !conduit.dma.token
      conduit.wait_all %t10 {token = false} : !conduit.dma.token
      %t11 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t11 {token = true} : !conduit.dma.token
      conduit.wait_all %t11 {token = false} : !conduit.dma.token
      %t12 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t12 {token = true} : !conduit.dma.token
      conduit.wait_all %t12 {token = false} : !conduit.dma.token
      %t13 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t13 {token = true} : !conduit.dma.token
      conduit.wait_all %t13 {token = false} : !conduit.dma.token
      %t14 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t14 {token = true} : !conduit.dma.token
      conduit.wait_all %t14 {token = false} : !conduit.dma.token
      %t15 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t15 {token = true} : !conduit.dma.token
      conduit.wait_all %t15 {token = false} : !conduit.dma.token
      %t16 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t16 {token = true} : !conduit.dma.token
      conduit.wait_all %t16 {token = false} : !conduit.dma.token
      return
    }
  }
}
