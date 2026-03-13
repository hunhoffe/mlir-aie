# ObjectFIFO vs. AIR Channel: Divergence Analysis

**Last updated:** 2026-03-12 (revised: acquire_async bridge added; gap table regenerated)
**Status:** Research findings; empirical verification pending on Strix hardware.

This document records the committee-reviewed analysis of divergence between
`aie.objectfifo.*` (mlir-aie) and `air.channel.*` (mlir-air), and documents
how the Conduit dialect addresses each gap today or plans to address it.

**Evidence levels:**
- **[CODE]** — Confirmed by reading implementation source in this repo
- **[DOC]** — Confirmed by official documentation text
- **[INFER]** — Logical inference from the model; not yet experimentally verified
- **[UNVERIFIED]** — Plausible but requires a concrete failing program to confirm

---

## Research Statement

> **Conduit IR is a unified intermediate dialect that (1) enables a new class
> of programs combining ObjectFIFO's zero-copy window semantics with AIR
> Channel's N-D DMA descriptor efficiency — patterns expressible in neither
> source dialect alone — and (2) enables compiler optimizations previously
> available only in one framework to be applied to programs from both, through
> a shared lowering path.**

This is a two-part contribution. Part (1) is the **expressiveness claim**:
the cross-tier `conduit.put_memref_async` + `conduit.acquire_async` +
`conduit.wait_all` pattern expresses programs that neither mlir-aie nor
mlir-air can express directly. Part (2) is the **optimization claim**:
passes like `--conduit-depth-promote` operate on Conduit IR regardless of
whether the program originated from ObjectFIFO or AIR Channel.

---

## Master Gap Table

**Legend for Conduit columns:**

| Symbol | Meaning |
|---|---|
| ✓ | Supported today (compiled, in dialect or pipeline) |
| ◑ | Partially supported (in dialect but lowering incomplete, or annotation only) |
| ✗ | Not supported |
| 📋 | Planned (tracked in RESEARCH_PLAN.md) |
| — | Not applicable (this is a pass, not a dialect feature) |

### A. Expressiveness

| Feature | ObjectFIFO | AIR Channel | Conduit dialect | Conduit pipeline | Conduit roadmap |
|---|---|---|---|---|---|
| Sliding window (partial release) | ✓ `[CODE]` | ✗ | ✓ | ◑ subview placeholder | Fix subview in Pass C |
| Cyclostatic patterns | ✓ `[CODE]` | ✗ | ✓ | ◑ same | Same |
| Acquire-as-ceiling semantics | ✓ (implicit) | — | ✓ (documented) | ✓ Pass A correct | Done |
| N-D strided DMA descriptor | ◑ manual annotation | ✓ `[CODE]` | ✓ put/get_memref | ✗ Pass B needed | Track C |
| Async dependency graph | ✗ | ✓ `[CODE]` | ✓ acquire_async, wait_all | ✗ async lowering needed | Track B + C |
| SPMD herd indexing | ✗ | ✓ `[CODE]` | ✗ | ✗ | Future (beyond scope) |
| M:N producer:consumer | ✗ | ✓ `[INFER]` | ✗ | ✗ | Future |
| **DMA fill + window consumption (cross-tier)** | **✗** | **✗** | **✓ NEW** | **✗** | **Track B + C (priority)** |
| Packet-switched routing | ✓ flag `[CODE]` | ✗ | ◑ annotate hint | ✗ | Track B (medium effort) |
| repeat_count BD reuse | ✓ depth=1 only `[CODE]` | ✗ | ✗ | ✗ | Track B |

The **cross-tier** row is the key expressiveness contribution. With
`conduit.acquire_async` and `conduit.put_memref_async` combined in a
`conduit.wait_all`, a program can simultaneously use AIR's N-D DMA
efficiency (Tier 3 producer) and ObjectFIFO's zero-copy sliding window
(Tier 2 consumer). See Section 2 for details.

### B. Optimization Infrastructure

| Optimization | ObjectFIFO | AIR Channel | Conduit pipeline | Conduit roadmap |
|---|---|---|---|---|
| Automatic ping-pong / depth promotion | ✗ `[CODE]` | ✓ 7-pass | ✗ | `--conduit-depth-promote` Track B |
| Wrap/stride absorption from loops | ✗ `[CODE]` | ✓ | ✗ | `--conduit-wrap-stride` Track C |
| Channel time-multiplexing / fusion | ✗ `[CODE]` | ✓ | ✗ | `--conduit-fuse-channels` Track B |
| DMA hoist in accumulation loops | ✗ `[CODE]` | ✓ | ✗ | Future |
| BD count / repeat_count folding | ✗ `[CODE]` | ◑ shim only | ✗ | Track B |
| Memory footprint minimization | ✗ `[CODE]` | ✓ | ✗ | Future |
| Congestion-aware routing | ✓ PathFinder | ✗ | Delegated to `--aie-create-pathfinder-flows` | Done |

### C. Hardware Mapping

| Feature | ObjectFIFO | AIR Channel | Conduit today | Conduit roadmap |
|---|---|---|---|---|
| Lock ID assignment | ✗ opaque `[CODE]` | N/A | ✓ by design (no IDs in dialect) | Done |
| DMA channel budget management | ✗ fail if exhausted `[CODE]` | ✓ fuse-channels | ✗ | `--conduit-fuse-channels` Track B |
| Packet routing for AIR programs | N/A | ✗ | ◑ annotate hint | Track B |
| repeat_count for AIR programs | N/A | ✗ | ✗ | Track B |
| objectfifo.link composability | ✗ single op `[INFER]` | N/A | ✓ multiple link ops composable | Test in Track A |
| Kernel state introspection | ✗ | ✗ | ✓ conduit.status | Pass C lowering (low effort) |

### D. Documented Pain Points Addressed

| Pain Point | Evidence | Conduit status |
|---|---|---|
| DMA channel exhaustion (3+ streams) | `[CODE]` DMAChannelAnalysis | Roadmap: `--conduit-fuse-channels` |
| Lock ID collision when mixing custom sync | `[CODE]` AIEAssignLockIDs | ✓ Solved by design |
| "Not all patterns expressible in ObjectFIFO" | `[DOC]` section-2g README | ✓ Cross-tier pattern is the new class |
| repeat_count incompatible with depth > 1 | `[DOC]` section-2b/04_Repeat | Roadmap: decouple repeat_count from depth |
| Static unrolling code size explosion | `[CODE]` stateful transform LCM | Roadmap: depth-promote normalizes to LCM=2 |
| Irregular N:1 gather without MemTile relay | `[INFER]` | Roadmap: `--conduit-fuse-channels` + objectfifo_link |
| Acquire-as-ceiling semantics non-obvious | `[CODE]` AIE2_delayed_release.mlir | ✓ Documented in conduit.acquire description |
| Packet routing unavailable for AIR | `[CODE]` AIR always circuit-switched | Roadmap: Track B |

---

## 2. The Cross-Tier Expressiveness Contribution (New)

### What neither dialect can express

A convolution kernel that:
1. Reads a weight tensor from DDR using a 2D strided DMA descriptor
   (e.g., a 3×3 filter window at stride 16 over a 512×512 feature map)
2. Slides that filter over input buffers using a window-hold pattern
   (acquire 2 windows, compute, release 1 to advance by 1)
3. Overlaps the DMA fill of the *next* window with the compute on the
   *current* window, using async tokens to express the dependency

**mlir-aie ObjectFIFO**: cannot express the 2D strided DMA without manual
`dimensionsToStream` annotation; cannot express async overlap at the source level.

**mlir-air Channel**: cannot express the sliding window (each `get` is an
atomic copy; no partial-release semantics).

**Conduit**: can express the full program today at the dialect level:

```mlir
conduit.create {name = "weights", capacity = 18 : i64}  // 2 windows × 9 elems

// Non-blocking: DMA fills next weight window from DDR (Tier 3 / AIR strength)
%dma = conduit.put_memref_async {name = "weights",
           num_elems = 9 : i64,
           offsets = [0, 0], sizes = [3, 3], strides = [16, 1]}
           : !conduit.async.token

// Non-blocking: request output buffer window (Tier 2 / ObjectFIFO strength)
%acq = conduit.acquire_async {name = "output", count = 1 : i64}
           : !conduit.async.token

// Hardware satisfies both in parallel — DMA engine + lock arbiter
conduit.wait_all %dma, %acq

// Access both (window semantics on both, AnyType result)
%wt  = conduit.subview_access {name = "weights", index = 0 : i64}
           : memref<3x3xi16>
%out = conduit.subview_access {name = "output",  index = 0 : i64}
           : memref<16xi32>
// ... compute stencil on %wt and %out ...
conduit.release {name = "weights", count = 1 : i64}  // slide weight window
conduit.release {name = "output",  count = 1 : i64}
```

### The reconciling concept

A `!conduit.async.token` from `conduit.acquire_async` IS pending permission
to access a buffer window. Both DMA completion tokens (`put_memref_async`)
and window-grant tokens (`acquire_async`) are the same type and can be
combined in `conduit.wait_all`. The hardware satisfies them concurrently:
the DMA engine fills buffers autonomously while the lock arbiter processes
the window request.

`conduit.acquire` (blocking) = `conduit.acquire_async` + `conduit.wait`.

This makes the entire blocking window model a special case of the async
token model, and makes the two models composable rather than competing.

### Lowering path (planned)

| Conduit op | AIE hardware lowering |
|---|---|
| `conduit.put_memref_async` | Submit `aie.dma_bd` to DMA engine; token = in-flight BD |
| `conduit.acquire_async` | Issue `aie.use_lock(AcquireGreaterEqual, N)` in non-blocking poll form |
| `conduit.wait_all %dma, %acq` | Wait for both BD completion and lock grant |

Both operations run in hardware concurrently. The software polls for both
to complete, whichever takes longer. This is the minimum possible stall time
for a compute kernel that needs data from DMA and a buffer window simultaneously.

---

## 3. Confirmed ObjectFIFO Usability Pain Points

(Unchanged from original; see Section 4 of original document for full list.)

| Pain Point | Evidence | Source |
|---|---|---|
| DMA channel exhaustion | `[CODE]` | `DMAChannelAnalysis::getDMAChannelIndex` |
| Lock ID opacity | `[CODE]` | `AIEAssignLockIDs`, stateful transform |
| Not all patterns expressible | `[DOC]` | `programming_guide/section-2/section-2g/README.md` |
| repeat_count + depth > 1 unsupported | `[DOC]` | `programming_guide/section-2/section-2b/04_Repeat/README.md` |
| Static unrolling LCM code size | `[CODE]` | Stateful transform static lowering |
| objectfifo.link not composable | `[INFER]` | ObjectFifoLinkOp design |
| Acquire-as-ceiling non-obvious | `[CODE]` | `test/.../AIE2_delayed_release.mlir` |

---

## 4. Open Questions Requiring Empirical Verification

1. Does the cross-tier pattern (Tier 3 producer + Tier 2 consumer with
   `acquire_async`) compile and run correctly on Strix? What is the latency
   reduction vs. blocking acquire?

2. Does ping-pong promotion (depth 1→2 via `--conduit-depth-promote`)
   reduce compute stall cycles on Strix? By how much?

3. Does channel fusion (`--conduit-fuse-channels`, when implemented)
   allow 3-stream programs to compile that currently fail due to DMA
   channel exhaustion?

4. Does the Conduit pipeline produce output semantically equivalent to
   `--aie-objectFifo-stateful-transform` for the benchmark suite?
   (Correctness gate for all other measurements.)

5. Does the cross-tier expressiveness demo (convolution with 2D DMA + sliding
   window) produce measurably better throughput than the equivalent program
   written without the async overlap?

---

## 5. Conduit Dialect Status (as of 2026-03-12)

| Component | Status |
|---|---|
| All ops in Conduit.td | Compiled, registered in aie-opt |
| Roundtrip lit test | Passing (includes acquire_async, release_async, AnyType subview) |
| Pass A (--objectfifo-to-conduit) | 7/7 lit tests passing |
| Pass C (--conduit-to-dma) | 7/7 lit tests passing; depth-1 single-consumer correct |
| Pass B (--air-channel-to-conduit) | Not yet implemented |
| --conduit-depth-promote | Not yet implemented |
| acquire_async lowering in Pass C | Not yet implemented |

*This document was produced through combined code analysis of mlir-aie and
mlir-air and design review of the Conduit dialect. See `RESEARCH_PLAN.md`
for the implementation roadmap.*
