// RUN: aie-opt --verify-diagnostics --conduit-to-dma %s
// RUN: aie-opt --verify-diagnostics --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// Pass C Case B (compute MM2S → shim) overlong-BD-chain BUG pin (heterogeneous).
//
// Same Pass C bug location as `..._homogeneous_BUG.mlir`:
//   ConduitToDMALink.cpp Case B (lines 2030-2202).
//   * L2037: caseBBdRepeat = info.bdRepeat > 1 ? info.bdRepeat : 1;
//   * L2038: caseBEffectiveBDs = info.nConsumerBuffers() * caseBBdRepeat;
//   * L2060/L2139: emit caseBEffectiveBDs sequential aie.dma_bd blocks.
//   * L2123/L2198: NextBD chain `bdBlocks[(i+1) % caseBEffectiveBDs]`.
//   * NO targetModel.getNumBDs(...) cap — chain length is putCount-driven.
//
// Heterogeneous variant: 17 puts have DIFFERENT per-put offsets (offset = i*8).
// `--conduit-canonicalize-loop-unroll-puts`'s match predicate requires:
//   name + num_elems + offsets + sizes + strides + producer_dimensions all
//   identical across N puts.  Different offsets per put → predicate FAILS,
//   canon REFUSES to collapse, even after canon lands.
//
// Why this variant matters (Task #15 cap-helper need):
//   The IRON `for batch in range(N)` Python-unroll case is HOMOGENEOUS — canon
//   collapses it.  But several other reach paths produce HETEROGENEOUS puts
//   that canon won't touch:
//     (a) `--conduit-fuse-channels` merging multiple distinct compute→shim
//         channels into one canonical name post-fusion (per-channel offsets
//         remain distinct).
//     (b) Future canon symmetric expansion patterns.
//     (c) Hand-authored Conduit IR for non-IRON harnesses.
//   For these, the only fix is a `targetModel.getNumBDs(tile)` cap in Pass C
//   itself — that's Task #15's `checkBDChainCap` helper.  Until that lands,
//   the structural latent bug at Case B remains exposed.
//
// Fixture geometry:
//   * compute(0,2) → shim(0,0) via @chan
//   * @chan: depth = 2, memref<8xi32>; conduit.create authored directly.
//   * aie.shim_dma_allocation @chan(%shim, S2MM, 0) — shim is consumer.
//   * 17 puts with offsets = 0, 8, 16, 24, ..., 128 (= i*8).  All other
//     attrs (sizes, strides, num_elems, no producer_dimensions) match.
//   * Each put paired with wait_all{token=true} (await) + wait_all{token=
//     false} (free).  Per-put waits are tied to the per-put token, so wait
//     identity matches puts.
//
// Today's behavior (this RUN line):
//   Pass C emits 17 sequential aie.dma_bd blocks in aie.mem on compute(0,2).
//   AIE verifier rejects with `'aie.mem' op has more than 16 blocks`.
//
// Post-canon flip plan:
//   `--conduit-canonicalize-loop-unroll-puts` REFUSES (heterogeneous offsets
//   fail predicate) — IR is unchanged, Pass C still crashes.  This fixture
//   STAYS as expected-error pin until Task #15's cap helper lands at every
//   Pass C BD-emit site.  Then the cap helper either:
//     (a) emits a Pass C structured diagnostic naming the channel + cap
//         (preferred — clearer error than the verifier), or
//     (b) emits dma_repeat / temporal-mux equivalent that fits in the cap.
//   Whichever (a)/(b) lands, this fixture's CHECK lines flip then.

module @path_c_case_b_compute_mm2s_overlong_bd_chain_heterogeneous {
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

    // 17 HETEROGENEOUS puts on @chan.  offsets vary per put (i*8) — canon
    // refuses to collapse because its match predicate requires identical
    // offsets across all candidates.
    func.func @sequence(%arg0: memref<136xi32>) {
      %t0 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token
      %t1 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 8>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token
      %t2 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 16>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token
      %t3 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 24>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t3 {token = true} : !conduit.dma.token
      conduit.wait_all %t3 {token = false} : !conduit.dma.token
      %t4 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 32>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t4 {token = true} : !conduit.dma.token
      conduit.wait_all %t4 {token = false} : !conduit.dma.token
      %t5 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 40>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t5 {token = true} : !conduit.dma.token
      conduit.wait_all %t5 {token = false} : !conduit.dma.token
      %t6 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 48>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t6 {token = true} : !conduit.dma.token
      conduit.wait_all %t6 {token = false} : !conduit.dma.token
      %t7 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 56>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t7 {token = true} : !conduit.dma.token
      conduit.wait_all %t7 {token = false} : !conduit.dma.token
      %t8 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 64>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t8 {token = true} : !conduit.dma.token
      conduit.wait_all %t8 {token = false} : !conduit.dma.token
      %t9 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 72>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t9 {token = true} : !conduit.dma.token
      conduit.wait_all %t9 {token = false} : !conduit.dma.token
      %t10 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 80>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t10 {token = true} : !conduit.dma.token
      conduit.wait_all %t10 {token = false} : !conduit.dma.token
      %t11 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 88>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t11 {token = true} : !conduit.dma.token
      conduit.wait_all %t11 {token = false} : !conduit.dma.token
      %t12 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 96>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t12 {token = true} : !conduit.dma.token
      conduit.wait_all %t12 {token = false} : !conduit.dma.token
      %t13 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 104>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t13 {token = true} : !conduit.dma.token
      conduit.wait_all %t13 {token = false} : !conduit.dma.token
      %t14 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 112>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t14 {token = true} : !conduit.dma.token
      conduit.wait_all %t14 {token = false} : !conduit.dma.token
      %t15 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 120>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t15 {token = true} : !conduit.dma.token
      conduit.wait_all %t15 {token = false} : !conduit.dma.token
      %t16 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 128>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t16 {token = true} : !conduit.dma.token
      conduit.wait_all %t16 {token = false} : !conduit.dma.token
      return
    }
  }
}
