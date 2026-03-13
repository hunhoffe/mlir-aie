# Conduit Research Plan — Three Parallel Tracks

**Last updated:** 2026-03-12 (revised: two-part hypothesis; cross-tier expressiveness added)
**Team:** 3 students, 2-year timeline
**Target hardware:** Strix (npu2, AIE2p) primary; Phoenix (npu1, AIE2) secondary
**Goal:** Empirical support for the Conduit research statement on real hardware

---

## Research Statement

> **Conduit IR is a unified intermediate dialect that (1) enables a new class
> of programs combining ObjectFIFO's zero-copy window semantics with AIR
> Channel's N-D DMA descriptor efficiency — patterns expressible in neither
> source dialect alone — and (2) enables compiler optimizations previously
> available only in one framework to be applied to programs from both, through
> a shared lowering path.**

This is a two-part contribution with two corresponding claims to validate:

**Expressiveness claim (new):** The cross-tier `conduit.put_memref_async` +
`conduit.acquire_async` + `conduit.wait_all` pattern expresses programs that
neither mlir-aie nor mlir-air can express directly. Specifically: a program
that simultaneously uses AIR's N-D strided DMA (Tier 3 producer) and
ObjectFIFO's partial-release sliding window (Tier 2 consumer) with async
overlap between them. The `acquire_async` op bridges the two models: a
`!conduit.async.token` from an acquire request is the same type as a token
from a DMA submission, and `conduit.wait_all` over both allows the hardware
to satisfy both in parallel.

**Optimization claim (original):** Optimization passes operating on Conduit IR
(e.g., `--conduit-depth-promote`) apply without modification to programs
originating from either ObjectFIFO or AIR Channel, demonstrating a shared
optimization path that does not exist today.

This research statement requires three things to be validated, each owned
by one track:

1. **Track A:** The divergence between ObjectFIFO and AIR Channel is real and
   measurable on hardware (not just theoretical); the gap table in
   `conduit_ir/DIVERGENCE_ANALYSIS.md` is confirmed with concrete failing programs.
2. **Track B:** The Conduit → DMA lowering is correct; `--conduit-depth-promote`
   improves performance on Strix; `conduit.acquire_async` lowers correctly.
3. **Track C:** The cross-tier expressiveness claim is validated on hardware:
   a program using Tier 3 DMA fill + Tier 2 window consumption via `acquire_async`
   compiles and runs, and shows better throughput than either source dialect
   can achieve alone.

Tracks A and B can proceed in parallel from day one. Track C depends on Track B
reaching correctness parity (around month 4).

---

## Track A: Gap Verification and Measurement Infrastructure

**Owner:** Student 1
**Primary output:** A paper-quality table of confirmed, hardware-validated gaps
between ObjectFIFO and AIR Channel, plus the benchmark infrastructure to
reproduce the measurements.

### Motivation

The divergence table in `conduit_ir/DIVERGENCE_ANALYSIS.md` was produced by
reading implementation source. Code analysis confirms which gaps *exist in the
model* but not which gaps *matter in practice*. Track A converts code-level
claims into hardware-level evidence.

### Deliverables

**Month 1–3: Gap Verification Suite**

Create `test/gap_verification/` with one `.mlir` file per divergence row:

| File | Claim being tested | Expected result |
|---|---|---|
| `gap_dma_channel_exhaustion.mlir` | 3 ObjectFIFOs on one compute tile fails to compile | Compile error: "number of output DMA channel exceeded" |
| `gap_lock_id_collision.mlir` | Manual lock + ObjectFIFO on same tile silently collides | Hardware deadlock or wrong output |
| `gap_no_packet_routing_air.mlir` | AIR program with 2 channels between same tiles uses 2 physical flows | aie.flow ×2 in output (no aie.packet_flow) |
| `gap_no_pingpong_objectfifo.mlir` | ObjectFIFO depth=1 program does not auto-promote | No depth=2 BD chain in stateful transform output |
| `gap_repeat_count_air.mlir` | AIR channel with replay pattern generates per-transfer lock ops | N lock ops where 1+hardware-repeat would suffice |
| `gap_wrap_stride_objectfifo.mlir` | ObjectFIFO accessing 2D matrix tile requires manual dimensionsToStream | No auto-stride derivation from surrounding loop |
| `gap_sliding_window_air.mlir` | AIR channel cannot express partial-release sliding window | Requires manual scratch-buffer copy |
| `gap_no_channel_fusion_objectfifo.mlir` | ObjectFIFO: 3 inputs to one tile → compile failure | Compare to mlir-air with --air-fuse-channels |

For each file: document (a) what the program tries to do, (b) what the
ObjectFIFO path produces, (c) what the AIR Channel path produces (if applicable),
(d) the compile error or wrong-output evidence.

**Month 3–6: Hardware Benchmarking Infrastructure**

Extend `tools/compare_lowering_resources.py` to also measure hardware performance:

```python
# compare_hardware_performance.py
# For each benchmark: run on Strix via XRT, record:
#   - wall-clock time (XRT timestamps)
#   - BD count (from compiled output)
#   - lock count
#   - DMA channel count
#   - binary size
```

Set up the measurement pipeline:
1. Compile via `--aie-objectFifo-stateful-transform` (baseline)
2. Compile via `--objectfifo-to-conduit --conduit-to-dma` (Conduit, no optimization)
3. Record output IR and run on Strix
4. Verify outputs are byte-identical for the same inputs (correctness gate)

Three benchmark programs from `programming_examples/basic/`:
- `passthrough_dmas` (no compute, baseline)
- `vector_vector_add` (depth-1, compute between acquire/release)
- `matrix_scalar_add` (depth-1, scalar compute)

**Month 6–12: Extended Benchmark Suite**

Add the Track B optimization passes as they become available and add measurement
columns for each:
- `--conduit-depth-promote` (from Track B)
- `--conduit-fuse-channels` (from Track B, if completed)

Write a comparison table documenting all measurements. This becomes the
evaluation section of the paper.

**Month 12: Workshop Paper Draft**

Target venue: MLIR Workshop at LLVM DevMtg, or CGO 2027.
Title candidate: "Measuring the ObjectFIFO–AIR Channel Expressiveness Gap
on Commercial NPU Hardware."

### Hardware Access

Use **Strix (npu2)** as primary because:
- Newest hardware (AIE2p), most relevant for publication
- Full wrap/stride descriptor support
- MemTile present (enables link pattern testing)

Use **Phoenix (npu1)** as secondary for cross-validation. Document any
differences between npu1 and npu2 behavior.

### Key Skills Required

- Reading compiled MLIR IR (aie.dma_bd, aie.lock, aie.flow)
- XRT runtime API for NPU kernel execution
- Python scripting for benchmark automation

---

## Track B: Conduit Correctness and Optimization for ObjectFIFO Programs

**Owner:** Student 2
**Primary output:** A correct `--conduit-to-dma` pipeline plus at least one
hardware-validated optimization pass (`--conduit-depth-promote`).

### Current Status

Pass A (`--objectfifo-to-conduit`) and Pass C (`--conduit-to-dma`) compile and
7/7 lit tests pass. Pass C is correct for depth-1, single-consumer programs
only. Known gaps:

1. Depth > 1 BD chain (only `buffers[0]` used; depth-N buffers allocated but never connected)
2. Producer-side lock protocol for shim DMA (consumer locks correctly managed; no producer-side lock release in BD chain)
3. Multi-consumer broadcast (only `consumerTiles[0]` allocated)
4. Buffer type placeholder (`memref<capacity/depth x i32>` instead of real element type)
5. Lock ID collision risk (currently fixed IDs 0/1; needs delegation to `AIEAssignLockIDs`)

### Deliverables

**Month 1–4: Pass C Correctness + acquire_async lowering**

Fix the 5 known correctness gaps. The reference implementation for all
correctness questions is `AIEObjectFifoStatefulTransform.cpp`. For each
fix, add a lit test.

Priority order:
1. Fix lock ID: remove explicit ID from `aie.lock` builder; let `AIEAssignLockIDs` run downstream
2. Fix producer-side shim lock: shim DMA path must emit `use_lock(prodLock, AcquireGreaterEqual)` in the BD chain send block
3. Fix depth-2 BD chain: two-buffer circular chain (bd0 → bd1 → bd0) with alternating lock acquire/release
4. Fix buffer type: parse `element_type` annotation from `conduit.annotate` instead of using placeholder
5. Fix multi-consumer: iterate over all `consumer_tile_N` annotations, allocate buffer+lock set per consumer
6. **NEW: Lower `conduit.acquire_async` in Pass C.** Map to a non-blocking
   poll form of `aie.use_lock(AcquireGreaterEqual, N)`. The token becomes
   complete when the lock is granted. This is required for Track C's
   cross-tier expressiveness demo and for the `--conduit-depth-promote`
   async variant. Implementation: create a new Phase 6b in Pass C that walks
   `Acquire` ops and checks for a paired async token use; if found, emits the
   non-blocking form.

Correctness gate: for each program in Track A's benchmark suite, the output
of `--objectfifo-to-conduit --conduit-to-dma --aie-assign-lock-ids
--aie-assign-buffer-addresses --aie-assign-bd-ids --aie-create-pathfinder-flows`
must produce semantically equivalent BD sequences to `--aie-objectFifo-stateful-transform`
plus downstream passes. Verify by running both outputs on Strix and comparing
hardware output (byte-identical result tensors).

**Month 4–7: --conduit-depth-promote**

New file: `lib/Dialect/Conduit/Transforms/ConduitDepthPromotion.cpp`

Algorithm:
```
for each conduit.create with conduit.annotate {key="depth", value="1"}:
    collect all conduit.acquire / conduit.release for this conduit
    check: are they inside an scf.for body inside an aie.core body?
    check: acquire count == 1?
    check: at least one non-conduit op between acquire and release?
    if all pass:
        update conduit.annotate {key="depth", value="2"}
        update conduit.create capacity = capacity * 2
```

This is ~150 lines. Register it as `--conduit-depth-promote` in `Passes.td`.

Add lit tests in `test/Dialect/Conduit/`:
- `conduit_depth_promote_simple.mlir` (should promote)
- `conduit_depth_promote_no_compute.mlir` (should NOT promote — passthrough)
- `conduit_depth_promote_multi_conduit.mlir` (each conduit evaluated independently)

Provide hardware measurements to Track A.

**Month 7–12: --conduit-fuse-channels (stretch goal)**

If correctness and depth-promote are complete, implement channel fusion:
detect pairs of `conduit.create` on the same tile whose acquire/release windows
are non-overlapping (determined by static analysis of the `scf.for` loop
structure), and annotate them to share one DMA channel in Pass C.

This directly addresses the DMA channel exhaustion gap (Track A, gap 1).

**Month 12: Technical Report / Conference Paper Section**

Document: (a) the known correctness gaps in Pass C and how they were fixed,
(b) the depth-promote algorithm and hardware results, (c) comparison against
the stateful transform baseline.

### Key Skills Required

- MLIR pass development (OpBuilder, walk, DenseMap)
- Reading AIE BD/lock IR (aie.dma_bd, aie.lock, aie.use_lock)
- Understanding the AIE2 lock protocol (producer/consumer init values, AcquireGreaterEqual semantics)

---

## Track C: Pass B (AIR Channel → Conduit) and Cross-Dialectical Validation

**Owner:** Student 3
**Primary output:** A working `--air-channel-to-conduit` pass (Pass B) that
enables AIR Channel programs to benefit from ObjectFIFO-derived optimizations
via the shared Conduit lowering path.

### Dependency

Track C requires Track B to reach correctness parity (month 4) before
cross-dialectical validation is meaningful. Track C can begin with Pass B
design and implementation while Track B finishes correctness work.

### The Cross-Dialectical Claim

mlir-air has `--air-ping-pong-transform` (7-pass, 7000 lines). mlir-aie has
nothing equivalent. Via Conduit:

1. An AIR Channel program → Pass B → Conduit IR → `--conduit-depth-promote` → Pass C → hardware
2. The `--conduit-depth-promote` pass applies without modification — it sees the same `conduit.create`/`conduit.acquire`/`conduit.release` structure regardless of whether the program originated from ObjectFIFO or AIR Channel.
3. **This is the thesis:** one optimization pass, written once, applies to programs from both frameworks.

### Deliverables

**Month 1–4: Pass B Design and Prototype**

Study `air.channel.put` / `air.channel.get` API:
- `air.channel @name [dims]` — channel declaration with SPMD dimensions
- `air.channel.put @name[indices] (%buf[offsets][sizes][strides]) {id=N}`
- `air.channel.get @name[indices] (%buf[offsets][sizes][strides]) {id=N}`

New file: `lib/Dialect/Conduit/Transforms/AirChannelToConduit.cpp`

Phase 1 (months 1–2): Handle the simplest case — non-indexed, non-async, blocking channels:
```
air.channel @foo [1,1]
air.channel.put @foo[] (%buf[0][N][1]) → conduit.create + conduit.put_memref
air.channel.get @foo[] (%buf[0][N][1]) → conduit.get_memref
```

Phase 2 (months 3–4): Handle indexed channels and async tokens:
```
air.channel.put async [%tok] @foo[%i][%j] (%buf[...][...][...])
→ conduit.put_memref_async {name="foo_i_j", offsets=..., sizes=..., strides=...}
```

For indexed channels: either flatten the 2D channel array to individual named
conduits (one per index), or extend `conduit.create` with a `shape=[dims]`
attribute. Discuss with the team; flattening is simpler for the prototype.

Add lit tests in `test/Dialect/Conduit/`:
- `air_channel_to_conduit_simple.mlir`
- `air_channel_to_conduit_indexed.mlir`
- `air_channel_to_conduit_async.mlir`

**Month 4–6: Cross-Tier Expressiveness Demo (priority deliverable)**

This is now the primary Track C deliverable and validates the expressiveness
claim in the research statement — distinct from (and earlier than) the
cross-dialectical optimization demo.

Write a benchmark program in Conduit IR directly (not through Pass A or Pass B)
that uses the cross-tier pattern:

```mlir
// Tier 3 producer: DMA fill with N-D descriptor
%dma = conduit.put_memref_async {name = "weights", num_elems = 9 : i64,
           offsets = [0, 0], sizes = [3, 3], strides = [16, 1]}
           : !conduit.async.token

// Tier 2 consumer: sliding window acquisition
%acq = conduit.acquire_async {name = "output", count = 1 : i64}
           : !conduit.async.token

conduit.wait_all %dma, %acq
// ... compute ...
conduit.release {name = "weights", count = 1 : i64}
conduit.release {name = "output",  count = 1 : i64}
```

To make this work, Track B must first implement `acquire_async` lowering in
Pass C (month 3-4). Track C then:
1. Writes the cross-tier benchmark in Conduit IR
2. Runs it through `--conduit-to-dma` on Strix
3. Measures throughput vs. equivalent programs in pure ObjectFIFO (blocking
   acquire, no async overlap) and pure AIR (opaque copy, no sliding window)
4. Documents the result as Empirical Result #1 for the expressiveness claim

**Month 4–8: Cross-Dialectical Optimization Demo**

Select an mlir-air benchmark that uses blocking channels in a for loop:
- Look in `mlir-air/mlir/test/` for programs that resemble `vector_vector_add`
- Target: a program that currently goes through `--air-to-aie` with depth-1 equivalent channels

Run it through:
```bash
# AIR path (existing):
aie-opt --air-to-aie --aie-objectFifo-stateful-transform ... input.mlir

# Conduit path (new):
aie-opt --air-channel-to-conduit \
        --conduit-depth-promote \
        --conduit-to-dma \
        --aie-assign-lock-ids ... input.mlir
```

Compare on Strix: does the Conduit path (with depth-promote) outperform
the AIR path (without automatic ping-pong)?

This comparison is the empirical cross-dialectical result. If yes, you have
hardware evidence that Conduit enables an optimization that the AIR path
could not achieve without Conduit.

**Month 8–12: Validation at Scale**

Run Pass B on a subset of mlir-air's existing test files (those that use
non-async blocking channels in for loops). Document:
- How many programs Pass B handles correctly
- How many require async token support (deferred)
- How many require SPMD indexed channel support (deferred)

This becomes the "coverage" section of the paper: "Pass B correctly handles
X% of mlir-air programs in the test suite."

**Month 12: Cross-Dialectical Section of Main Paper**

Document the cross-dialectical demo with hardware results. This is the
centerpiece of the thesis's contribution: one optimization, zero modification,
two input dialects.

### Key Skills Required

- mlir-air dialect API (air.channel ops, async tokens)
- MLIR pass development
- Understanding AIR → AIE lowering (study `--air-to-aie` pass as reference)

---

## Convergence: The Paper

At month 12, the three tracks converge into one paper with two claims:

**Claim 1 (expressiveness):** Conduit expresses programs combining ObjectFIFO
window semantics and AIR Channel N-D DMA descriptors with async overlap, which
neither source dialect alone can express. Hardware result: cross-tier benchmark
on Strix achieves better throughput than equivalent programs in either dialect.

**Claim 2 (optimization):** A single `--conduit-depth-promote` pass applies
to programs from both dialects without modification, demonstrating the shared
optimization path. Hardware result: X% stall reduction on Strix.

| Paper section | Contributed by |
|---|---|
| Motivation (gap table, confirmed pain points) | Track A |
| Conduit design (three-tier IR, acquire_async bridge) | All tracks |
| Expressiveness claim + cross-tier benchmark | Track B (lowering) + Track C (demo) |
| Correctness evaluation (Conduit == stateful transform) | Track A + B |
| Optimization claim (depth-promote on Strix) | Track A + B |
| Cross-dialectical optimization demo (AIR program benefits) | Track C |
| Hardware measurement table | Track A |

Target venues:
- **Year 1 (month 12):** Workshop paper at LLVM DevMtg, CGO Workshop, or LCTES
- **Year 2:** Full paper at CGO, PLDI, or MLSys

---

## Coordination Points

| Month | Sync event |
|---|---|
| 1 | All tracks: agree on benchmark programs; Track B starts correctness fixes |
| 3 | Track A delivers gap verification suite; Track B delivers depth-2 BD chain + acquire_async lowering |
| 6 | Track B delivers depth-promote; Track C delivers cross-tier expressiveness demo on Strix |
| 9 | Track A delivers full hardware measurements; Track C delivers cross-dialectical optimization demo |
| 12 | All tracks: paper draft with both claims validated |

---

## Reference: File Locations

| What | Path |
|---|---|
| Divergence analysis (this research's foundation) | `conduit_ir/DIVERGENCE_ANALYSIS.md` |
| Gap verification test suite (Track A creates) | `test/gap_verification/` |
| Hardware benchmarking script (Track A extends) | `tools/compare_hardware_performance.py` |
| Pass C source (Track B owns) | `lib/Dialect/Conduit/Transforms/ConduitToDMA.cpp` |
| Depth-promote pass (Track B creates) | `lib/Dialect/Conduit/Transforms/ConduitDepthPromotion.cpp` |
| Pass B source (Track C creates) | `lib/Dialect/Conduit/Transforms/AirChannelToConduit.cpp` |
| mlir-air ping-pong reference | `mlir-air/mlir/lib/Transform/AIRDependencyScheduleOpt.cpp:AIRLabelScfForLoopForPingPongPattern` |
| mlir-air channel fusion reference | `mlir-air/mlir/lib/Transform/AIRDependencyScheduleOpt.cpp:AIRFuseChannels` |
| ObjectFIFO stateful transform (correctness oracle) | `mlir-aie/lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp` |
