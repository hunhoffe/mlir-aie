# Conduit: A Unified Bridge IR for AIE Streaming Abstractions

**Workshop paper draft — MLIR Workshop at PLDI/CGO**
**Target length: 4–6 pages**
**Status: Draft v6 (2026-03-14); Evaluation pending hardware results**

---

## Abstract

AMD AIE spatial processors are programmed through two incompatible MLIR
dialects: `mlir-aie`'s ObjectFIFO, which provides blocking buffer-window
acquisition with partial-release sliding-window semantics, and `mlir-air`'s
Channel, which provides N-D strided DMA transfers with async token
dependencies. Neither dialect can express programs that combine both — for
example, overlapping an asynchronous DMA fill with a non-blocking window
acquisition.

We present Conduit, a 16-op MLIR dialect that bridges both abstractions
through a unified async token model. The key mechanism is
`conduit.acquire_async`, which returns a `!conduit.window.token` that can be
combined with a `!conduit.dma.token` (from `conduit.put_memref_async`) in a
single `conduit.wait_all` fan-in call, enabling the AIE hardware to satisfy
DMA transfers and lock grants concurrently. Three compiler passes lower
ObjectFIFO programs (Pass A), AIR Channel programs (Pass B, static shapes),
and Conduit IR to raw BD/DMA hardware ops (Pass C).

We evaluate Conduit on a 130-file ObjectFIFO corpus: 103 files compile
end-to-end through the Conduit pipeline (the remainder require
still-unimplemented features such as packet switching). Five ObjectFIFO
attributes — `repeat_count`, `iter_count`, `disable_synchronization`,
`viaDMA`, and N-D DMA dimensions — are fully lowered through Pass C. For
depth-1 single-consumer programs, the generated IR is structurally equivalent
to the production stateful transform across all six AIE op types. Hardware
validation on AMD VCK5000 is the next milestone; no hardware results are
reported in this draft.

---

## 1. Introduction

AMD AIE tiles are spatial-architecture processors arranged in a 2D array with
dedicated DMA engines, hardware lock arbiters, and tile-local scratchpad memory.
The IRON framework [1] introduced a Python-level API over
the `mlir-aie` ObjectFIFO programming model and established the baseline
expressiveness profile for this hardware. Conduit extends IRON's work by
addressing the expressiveness boundary between ObjectFIFO and the companion
`mlir-air` dialect, introducing a common intermediate representation that
lowers both to raw BD/DMA hardware ops.

Programming AIE hardware requires precise control over DMA buffer descriptors
(BDs), lock acquire/release sequences, MemTile relay routing, and DMA channel
allocation (each compute tile has only 2 MM2S + 2 S2MM channels). Two MLIR
dialects provide higher-level abstractions over these primitives:

**ObjectFIFO (`mlir-aie`)** models data movement as named channels with
buffer-window semantics. A producer `acquire`s N buffer slots, fills them,
and `release`s; a consumer mirrors this sequence. Partial-release sliding
windows — where the consumer advances by fewer slots than it acquired,
retaining overlap across iterations — are natively expressible.

**AIR Channel (`mlir-air`)** models data movement as channel declarations
with N-D strided memref transfers. A `channel.put async [deps]` operation
submits an N-D DMA descriptor (offsets/sizes/strides) and returns an async
token that can be chained with downstream dependencies, enabling explicit
overlap between DMA transfers and compute.

The two dialects are incompatible in two ways. First, ObjectFIFO cannot express
async overlap at the source level — there is no token model. Second, AIR Channel
cannot express sliding-window partial-release semantics — each `channel.get` is
an atomic copy with no subview. A program requiring both strengths — a
convolution kernel that slides a filter window using partial-release while
overlapping the DMA fill of the next window — cannot be written in either dialect.

This expressiveness gap is not theoretical. Stencil computations such as
SPARTA [3] require cyclostatic access patterns when tiled
across AIE cores — boundary tiles process fewer elements than interior tiles.
The AIE hardware supports this natively through multi-BD chains, but no
existing MLIR dialect exposes cyclostatic rates at the IR level. Conduit
represents cyclostatic rate annotations on `conduit.create` and verifies CSDF
balance at compile time (§3.2b).

**Conduit** is a 16-op MLIR dialect that serves as a common intermediate
representation for both. This paper makes three contributions:

1. **A cross-tier async token model for hardware-concurrent DMA and window
   acquisition.** `conduit.acquire_async` returns a `!conduit.window.token`;
   `conduit.put_memref_async` returns a `!conduit.dma.token`. Both token types
   are accepted by `conduit.wait_all`, enabling the AIE DMA engine and lock
   arbiter to operate concurrently. This pattern — overlapping a DMA fill with
   a window grant — is inexpressible in either source dialect (§3.4). The
   actual performance benefit must be measured on AIE hardware (§6.4,
   Experiment 4); no hardware results are available yet.

2. **A unified dialect enabling cross-dialectical optimization.** The 16-op
   dialect covers ObjectFIFO's window semantics (Tier 2) and AIR Channel's
   N-D DMA descriptor semantics (Tier 3). Optimization passes such as
   `--conduit-depth-promote` operate on `conduit.acquire`/`conduit.release`
   and apply without modification to programs from either source dialect.
   Three compiler passes are implemented: Pass A (ObjectFIFO → Conduit),
   Pass B (AIR Channel → Conduit, static shapes only), and Pass C (Conduit →
   raw BD/DMA). Pass C queries the AIE device target model and emits
   architecture-correct lock operations for AIE1 (value-based) and AIE2/AIE2p
   (counting semaphores) from the same Conduit IR.

3. **An ownership-typed buffer window model.** `!conduit.window<T>` is a typed
   grant of exclusive buffer access that makes the acquire→use→release
   protocol statically verifiable through SSA def-use ordering. No existing
   AIE or spatial-fabric MLIR dialect enforces this protocol at the type level.

---

## 2. Background

### 2.1 AIE Tile Architecture

This work targets three AMD AIE hardware generations: AIE1 (xcvc1902,
VCK1902/VCK5000), AIE2 (npu1, Phoenix), and AIE2p (npu2, Strix). VCK5000
(AIE1) is the locally available validation platform; Phoenix (AIE2) and Strix
(AIE2p) are available via collaboration. Pass C queries the device target model
and emits architecture-specific lock operations for each generation. AIE tiles
consist of:
- **Compute tiles**: 64KB data memory (AIE2/AIE2p; 32KB on AIE1), VLIW vector
  processor, local DMA engine with 2 MM2S + 2 S2MM channels and 16 buffer
  descriptors, 16 hardware locks
- **MemTiles** (AIE2/AIE2p only; not present on AIE1): 512KB data memory,
  used as relay buffers for fan-in/fan-out patterns via
  `aie.objectfifo.link`; 6 MM2S + 6 S2MM DMA channels, 48 buffer
  descriptors, 64 locks
- **Shim tiles**: interface to host DDR via DMA channels
- **Cascade streams**: dedicated nearest-neighbor data paths (north↔south)
  that bypass the DMA engine entirely; used for systolic-array patterns.
  Conduit does not currently model cascade connections (see §5).

Data movement is configured at compile time as **buffer descriptors (BDs)**: a
BD specifies a source/destination address, transfer size, wrap/stride layout for
N-D transfers (up to 3 dimensions on AIE2, plus a repeat count for BD-level
replay), and a next-BD pointer for chaining. BDs are chained via successors; the
DMA engine traverses the chain autonomously.

Synchronization between DMA engines and compute cores uses **hardware locks**.
The lock model differs across generations:
- **AIE1** (xcvc1902): value-based locks. `Acquire(lock, val)` blocks until the
  lock holds the specified value; `Release(lock, val)` sets the lock to the
  specified value. The programmer must manage lock values explicitly.
- **AIE2/AIE2p** (npu1/npu2): counting semaphores.
  `AcquireGreaterEqual(lock, count)` blocks until the lock value ≥ count, then
  decrements by count. `Release(lock, count)` increments by count. This
  eliminates the value-management burden and enables multi-slot operations.

In both models, a producer lock guards free buffer slots (initialized to
`depth`); a consumer lock guards filled slots (initialized to `0`). A compute
core acquires the consumer lock before reading and releases the producer lock
afterward; the DMA engine performs the symmetric protocol.

### 2.2 ObjectFIFO Semantics

`aie.objectfifo @name(producerTile, {consumerTiles}, depth)` declares a named
FIFO channel. `aie.objectfifo.acquire(Produce, N)` acquires N buffer slots on
the producer side (acquires producer lock); `aie.objectfifo.acquire(Consume, N)`
acquires on the consumer side (acquires consumer lock). The port argument
determines which lock is acquired; this distinction is load-bearing for
hardware correctness.

`aie.objectfifo.subview.access %sv[i]` returns a `memref<T>` pointing to slot
`i` of the acquired window — this is an in-place reference to the DMA-filled
buffer, not a copy.

`aie.objectfifo.link [@srcs] -> [@dsts] ([join] [dist])` routes tokens through
a MemTile relay at byte offsets, enabling distribute (1 → N) and join (N → 1)
patterns. 98 link operations appear across 53 mlir-aie source files.

**Key limitation.** ObjectFIFO has no async token model. Programs are
synchronous: `acquire` blocks until N slots are available. There is no way to
overlap a buffer-window acquisition with a concurrent DMA fill.

### 2.3 AIR Channel Semantics

`air.channel @name [dims]` declares a channel with an optional SPMD shape.
`air.channel.put async [%dep0, %dep1] @name[%i, %j] (%buf[offsets][sizes][strides])`
submits a strided DMA transfer and returns an `!air.async.token`. Tokens can be
threaded into subsequent ops via the dependency list, forming an explicit async
dependency graph.

`air.wait_all [%t0, %t1]` provides fan-in synchronization. The AIR compilation
stack includes an extensive ping-pong promotion pass (`--air-ping-pong-transform`:
7 passes, ~7,000 lines) that doubles depth to overlap DMA fill with compute.

**Key limitation.** AIR Channel has no partial-release window semantics. Each
`channel.get` is an atomic copy of the full buffer. Sliding-window patterns
(acquire 2 slots, release 1) must be implemented via explicit scratch-buffer
management, defeating the zero-copy model.

### 2.4 The Expressiveness Gap

A program requiring both N-D strided DMA fill and partial-release sliding
window with async overlap is not expressible in either dialect. This gap
motivates Conduit as an intermediate layer rather than a pass within either
existing stack.

---

## 3. The Conduit Dialect

### 3.1 Design Goals

1. **Minimal op count.** The dialect should contain only ops with a hardware
   lowering target or a clear structural role. Sugar ops (blocking = async +
   wait) are acceptable to improve readability; ops with no lowering path are
   not. `conduit.wait` is technically sugar for `conduit.wait_all` on a single
   token and is a candidate for removal in a future cleanup pass; it is retained
   in the current dialect for readability in simple programs.

2. **Cross-tier composability.** Tier 2 (window) and Tier 3 (DMA-descriptor)
   ops must interoperate without a separate protocol. `!conduit.window.token`
   (from `acquire_async`) and `!conduit.dma.token` (from `put_memref_async`)
   are distinct types; both are accepted by `conduit.wait_all` (`Variadic<AnyType>`),
   enabling hardware-concurrent fan-in across both event kinds.

3. **Name-based channel reference.** AIE cores live in `IsolatedFromAbove`
   regions; SSA values cannot cross region boundaries. Channel references use
   string names. A future redesign using `FlatSymbolRefAttr` would give MLIR's
   symbol verifier visibility into these references.

### 3.2 Type System

The dialect defines three types:

**`!conduit.dma.token`** — an opaque handle representing a pending DMA transfer.
Produced by `conduit.put_memref_async`, `conduit.get_memref_async`, and
`conduit.wait_all_async`. Consumed by `conduit.wait` (blocking) or accepted
by `conduit.wait_all` / `conduit.wait_all_async` for fan-in.

**`!conduit.window.token`** — an opaque handle representing a pending
buffer-window lock grant. Produced by `conduit.acquire_async` and
`conduit.release_async`. Consumed by `conduit.wait_window` (which resolves
it into a `!conduit.window<T>` SSA value) or accepted by `conduit.wait_all`
for cross-tier fan-in alongside DMA tokens.

Both token types are intentionally opaque with no runtime executor model.
Unlike MLIR's built-in `!async.token` (which implies CPU-managed coroutines
via `async.execute`), these types represent events satisfied by AIE hardware
engines: DMA completion or lock grant. No software thread management is implied.

The cross-tier fan-in — hardware satisfying a DMA fill and a window grant
concurrently — works because `conduit.wait_all` accepts `Variadic<AnyType>`,
not because the two token types are identical. They are distinct MLIR types;
the type system prevents passing a `!conduit.window.token` to `conduit.wait`
(which requires `!conduit.dma.token`) and vice versa (§3.2a).

**`!conduit.window<T>`** — a typed grant of exclusive access to a DMA-filled
buffer window, where `T` is a `memref<N x element_type>` capturing both the
window size `N` and the element type. Produced by `conduit.acquire` (blocking
form) and by `conduit.wait_window` (async form). Consumed by
`conduit.subview_access`, which returns a `memref<T>` in-place reference to
the acquired buffer slot, and by `conduit.release`, which takes the window as
an SSA operand to establish the def-use chain in the verifier.

The `memref<N x element_type>` encoding mirrors ObjectFIFO's
`!aie.objectfifosubview<memref<N x T>>` pattern and statically encodes the
window size so the verifier can catch out-of-bounds `subview_access[i]` where
`i >= N`. It is the key to sliding-window correctness: a program can `acquire`
N slots and `release` K < N to advance the window by K while retaining N-K
slots of overlap. Pass C maps the window SSA value directly to the allocated
`aie.buffer`; `conduit.subview_access` returns that buffer reference.

**`conduit.wait_window` — the async form.** The blocking `conduit.acquire`
returns `!conduit.window<T>` directly. The async pattern uses a separate op
that converts the resolved `!conduit.window.token` to a typed window handle:

```mlir
%tok = conduit.acquire_async {name = "output", count = 1 : i64}
           : !conduit.window.token
// ... overlap: DMA fills buffer while compute runs on previous window ...
conduit.wait_all %dma_tok, %tok        // hardware satisfies both in parallel
%win = conduit.wait_window %tok for "output"
           : !conduit.window.token -> !conduit.window<memref<16xi32>>
%buf = conduit.subview_access %win {index = 0 : i64}
           : !conduit.window<memref<16xi32>> -> memref<16xi32>
conduit.release %win {count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<16xi32>>
```

The type transition from `!conduit.window.token` to `!conduit.window<T>`
via `conduit.wait_window` is a structural proof that synchronization has
occurred — it is statically impossible to call `conduit.subview_access`
with a raw token. `conduit.wait_window` emits no hardware ops; it maps the
resolved token to the allocated `aie.buffer` in Pass C bookkeeping.

### 3.2a Token Type Safety Properties

The split into `!conduit.dma.token` and `!conduit.window.token` provides
three static guarantees:

**Type separation.** The two token types are distinct MLIR type definitions.
`conduit.wait` accepts only `!conduit.dma.token`; `conduit.wait_window`
accepts only `!conduit.window.token`; passing the wrong type is a compile-time
error. The cross-tier `conduit.wait_all` uses `Variadic<AnyType>` in the ODS
definition but a runtime verifier (M8c) narrows acceptance to the two token
types, rejecting non-token operands.

**Acquire-before-use.** `conduit.subview_access` requires a `!conduit.window<T>`
SSA value, which can only be produced by `conduit.acquire` or
`conduit.wait_window`. The SSA def-use chain enforces a strict
acquire→use→release protocol without a custom verifier.

**Partial linearity.** MLIR does not natively enforce linear types, so
double-release (using the same window in two `conduit.release` ops) is
syntactically valid. Verifier M8a catches the most common case (cumulative
releases exceeding acquired count). Full use-after-release detection remains
future work.

### 3.2b CSDF Formal Properties

Conduit's `conduit.create` op carries optional `producer_rates` and
`consumer_rates` attributes for cyclostatic dataflow (CSDF) rate annotations.
Two compile-time verifiers check consistency:

**CSDF balance (M6).** A producer with rate sequence P (period q) and consumer
with rate sequence C (period r) must satisfy the Lee-Messerschmitt balance
equation [9]: `sum(P) * len(C) == sum(C) * len(P)`. This is the necessary
and sufficient condition for a consistent periodic schedule. The special case
`q = r = 1` recovers standard SDF. Verifier M6 checks this condition on
every `conduit.create` op with rate annotations.

**Buffer capacity (M7).** The buffer must absorb the peak token accumulation
over one hyper-period H = lcm(q, r). M7 simulates the producer and consumer
over H firings and verifies that `capacity >= peak_occupancy`. For the common
SDF case (uniform rates), this reduces to `capacity >= max(p, c)`. For CSDF
patterns like HDIFF's sliding window (acquire=5, release=1, depth=6), the
net consumption rate is 1 (the release count), not 5 (the acquire count);
the acquire count determines the minimum simultaneous buffer occupancy, not
the SDF rate.

**Multi-consumer buffer sizing.** For broadcast (shared buffers), the buffer
depth is governed by the slowest consumer: `buf_size >= max_i(peak_occupancy(C_i))`.
For distribute (independent buffers), each consumer's buffer is sized
independently. Pass C's multi-consumer allocation emits per-consumer buffer
and lock pairs accordingly.

The CSDF model and balance equation follow Bilsen et al. [9], who extended
Lee and Messerschmitt's SDF balance theory to cyclostatic rates. Buffer size
bounds follow Koh and Bodin [10].

### 3.3 The Three-Tier Organization

The 16 ops are organized into infrastructure and two tiers:

**Infrastructure ops** (tier-agnostic):

| Op | Summary |
|---|---|
| `conduit.create {name, capacity, producer_tile?, consumer_tiles?, element_type?, depth?, repeat_count?, iter_count?, disable_synchronization?, viaDMA?, dimensionsToStream?, dimensionsFromStream?}` | Declare a named FIFO channel with typed lowering attributes |
| `conduit.register_external_buffers {name, num_buffers, base_addr}` | Register host-side shim DMA buffers |
| `conduit.objectfifo_link {srcs, dsts, mode, memtile, offsets}` | MemTile relay (distribute or join) |

**Tier 2 — Buffer-window ops (ObjectFIFO path):**

| Op | Summary |
|---|---|
| `%win = conduit.acquire {name, count, port} : !conduit.window<T>` | Reserve N buffer slots; `port="Produce"\|"Consume"` selects the hardware lock |
| `conduit.release %win {count, port} : !conduit.window<T>` | Release M slots; window SSA operand establishes def-use chain; port selects the lock |
| `%buf = conduit.subview_access %win {index} : !conduit.window<T> -> memref<T>` | Access element N of the held window |
| `%tok = conduit.acquire_async {name, count} : !conduit.window.token` | Non-blocking acquire; window token accepted by `conduit.wait_all` alongside DMA tokens |
| `%win = conduit.wait_window %tok for "name" : !conduit.window.token -> !conduit.window<T>` | Materialize window handle after async acquire resolves |
| `%tok = conduit.release_async {name, count} : !conduit.window.token` | Non-blocking release |

**Tier 3 — Memref-DMA ops (AIR Channel path):**

| Op | Summary |
|---|---|
| `conduit.put_memref {name, num_elems, offsets, sizes, strides}` | DMA put of strided tile (blocking) |
| `conduit.get_memref {name, num_elems, offsets, sizes, strides}` | DMA get of strided tile (blocking) |
| `%tok = conduit.put_memref_async {...} : !conduit.dma.token` | Non-blocking DMA put |
| `%tok = conduit.get_memref_async {...} : !conduit.dma.token` | Non-blocking DMA get |
| `conduit.wait %tok : !conduit.dma.token` | Block until one DMA token completes |
| `conduit.wait_all %tok0, %tok1, ...` | Block until all tokens complete |
| `%tok = conduit.wait_all_async %tok0, ... : (...) -> !conduit.dma.token` | Merge tokens non-blocking |

**Op count rationale.** The compiled dialect has 16 ops. The irreducible minimum
is 10; the remaining 6 are sugar ops (blocking forms = async + wait) that
improve readability. `conduit.wait_window` is not sugar — it fills a structural
gap: without it, `conduit.acquire_async` produces a token with no typed
consumer that yields a window handle, making the async acquire pattern
incomplete at the type-system level.

### 3.4 The Cross-Tier Bridge: `conduit.acquire_async`

The central design insight is that `conduit.wait_all` accepts both
`!conduit.window.token` (from `acquire_async`, representing a pending lock
grant) and `!conduit.dma.token` (from `put_memref_async`, representing a
pending DMA transfer) in a single variadic call. The hardware satisfies both
in parallel. This makes it possible to write:

```mlir
// ObjectFIFO async acquire produces a window token; DMA put produces a DMA token:
%dma_tok = conduit.put_memref_async {name = "weights",
               offsets = array<i64: 0>, sizes = array<i64: 64>, strides = array<i64: 1>}
               : !conduit.dma.token
%acq_tok = conduit.acquire_async {name = "output", count = 1 : i64}
               : !conduit.window.token

// Cross-tier fan-in — both token types are accepted by conduit.wait_all:
conduit.wait_all %dma_tok, %acq_tok

// Materialize window after both hardware operations complete:
%win = conduit.wait_window %acq_tok for "output"
           : !conduit.window.token -> !conduit.window<memref<64xi32>>
%result = conduit.subview_access %win {index = 0 : i64}
           : !conduit.window<memref<64xi32>> -> memref<64xi32>
conduit.release %win {count = 1 : i64, port = "Consume"}
           : !conduit.window<memref<64xi32>>
```

ObjectFIFO cannot express this: `aie.objectfifo.acquire` always blocks, so a
DMA fill and a window grant cannot overlap. AIR Channel cannot express it
either: `air.channel.put` is an atomic transfer with no acquire/release
lifecycle. The combination requires dropping to raw hardware ops, forfeiting
composability and verifiability.

When both operations run concurrently, the critical path is
`max(DMA_latency, lock_wait)` rather than their sum. The actual benefit depends
on transfer size, lock contention, and interconnect load — a hardware
measurement question (§6.4). Pass C defers `aie.use_lock` emission for
`conduit.acquire_async` to the `conduit.wait_window` site, allowing
instructions between the two to execute without blocking.

The mechanism generalizes: multiple `conduit.acquire_async` tokens (e.g., an
input slice and a weight tensor) combine in a single `conduit.wait_all`, and
the hardware lock arbiter satisfies all grants in parallel.

### 3.5 Key Design Decisions

**Name-as-string channel reference.** AIE cores live in `IsolatedFromAbove`
regions; SSA values cannot cross region boundaries. Channels are referenced
by name string. The alternative — `FlatSymbolRefAttr` — would give MLIR's
symbol verifier visibility into dangling references; this is planned but not
yet implemented.

**`!conduit.async.token` is not `!async.token`.** MLIR's `!async.token`
implies CPU-managed coroutines (`async.execute`). AIE satisfies token
conditions via DMA engines and lock arbiters with no CPU involvement.

**`conduit.wait_window` separates synchronization from buffer access.**
An alternative design would have `acquire_async` produce `!conduit.window<T>`
directly, becoming valid only after `wait`. This creates a "valid after wait"
semantic condition that the type system cannot enforce. The chosen design
uses a type transition: `!conduit.window.token` → `!conduit.window<T>` via
`conduit.wait_window`, making the synchronization proof structural.

**Port-dispatched lock selection.** The `port = "Produce" | "Consume"`
attribute on `conduit.acquire`/`conduit.release` dispatches to the correct
hardware lock. A consumer releasing its window frees the *producer* lock
(`prodLock`), not the consumer lock — matching the AIE dual-lock protocol.

**Known vulnerability:** The `port` attribute is a plain string. A typo
(e.g., `"produce"` instead of `"Produce"`) silently emits wrong-polarity lock
operations, causing hardware deadlock with no compile-time diagnostic. This
should be replaced with an ODS `enum` attribute (`PortDirection::Produce |
PortDirection::Consume`) that makes invalid values a parse error.

| Op | `port` | Lock operation (AIE2) | Lock operation (AIE1) |
|---|---|---|---|
| `conduit.acquire` | `"Consume"` | `AcquireGreaterEqual(consLock, N)` | `Acquire(consLock, 1)` |
| `conduit.acquire` | `"Produce"` | `AcquireGreaterEqual(prodLock, N)` | `Acquire(prodLock, 0)` |
| `conduit.release` | `"Consume"` | `Release(prodLock, N)` | `Release(prodLock, 0)` |
| `conduit.release` | `"Produce"` | `Release(consLock, N)` | `Release(consLock, 1)` |

Note: the port-to-lock polarity inversion (consumer release frees the
producer lock) is the same across both generations; only the lock operation
semantics differ (counting semaphore vs. value-based).

**DMA channel fusion.** AIE compute tiles have only 2 MM2S + 2 S2MM DMA
channels. The `--conduit-fuse-channels` pass uses interval coloring to group
non-overlapping conduits on the same tile, reducing channel pressure via
BD chain fusion.

**CSDF-correct `repeat_count` lowering.** `conduit.acquire {count = 1}` in
the core body expresses logical intent — one buffer slot per iteration. When
`repeat_count = N > 1`, Phase 6 lowering scales the hardware lock operation
by `bdChainRepeatCount`: `AcquireGreaterEqual(lock, N)` and
`Release(lock, N)`. This separates logical semantics (the program acquires
one slot) from hardware implementation (the lock must account for
N BD replays per iteration), preserving CSDF correctness while keeping
the source IR readable.

---

## 4. Implementation

### 4.1 Pass A: ObjectFIFO to Conduit

Pass A (`--objectfifo-to-conduit`, `ObjectFifoToConduit.cpp`) converts
`aie.objectfifo.*` ops to Conduit Tier 2 ops in five phases:

1. **Fifo scan**: walks `aie.objectfifo` ops, records name, producer tile,
   consumer tiles, depth, and element type.
2. **Channel creation**: emits `conduit.create` for each fifo with typed
   attributes `producer_tile`, `consumer_tiles`, `element_type`, `depth`,
   and — when present on the source ObjectFIFO — `repeat_count`,
   `iter_count`, `disable_synchronization`, `viaDMA`,
   `dimensionsToStream`, and `dimensionsFromStream`. No separate
   `conduit.annotate` ops are emitted.
3. **Link rewriting**: converts `aie.objectfifo.link` to `conduit.objectfifo_link`,
   inferring `mode` from the src/dst count (1 src = distribute; 1 dst = join).
4. **Acquire/release/subview rewriting**: converts `acquire`, `release`, and
   `subview.access` ops. Pass A reads `op.getPort()` and propagates it as a
   `port = "Produce" | "Consume"` string attribute on `conduit.acquire` and
   `conduit.release`. `aie.objectfifo.subview.access` is rewritten to
   `conduit.subview_access %window {index}`, where `%window` is the SSA result
   of the corresponding `conduit.acquire`. No `memref.alloc` placeholders are
   emitted; data connectivity is preserved through the window SSA value.
5. **Deferred erase**: ObjectFIFO ops are erased after the acquire/release
   walk, keeping the objectfifo symbol alive for the AIE verifier.

**Status**: compiled; 62 PASS / 0 XFAIL / 0 FAIL in lit suite. Tests cover
depth-1 single fifo (AIE1 and AIE2 variants), 1→3 distribute link, broadcast,
N→1 join link structure, producer acquire, window type, multi-conduit lock
non-collision, depth-2 BD ring + rotation counter, shared memory, async
acquire end-to-end, channel fusion, MemTile relay, the five new ObjectFIFO
attributes (`repeat_count`, `iter_count`, `disable_synchronization`, `viaDMA`,
N-D DMA dimensions), and interaction tests (feature combinations, edge cases,
link-distribute + repeat_count). Corpus coverage: see §6.2.

### 4.2 Pass C: Conduit to BD/DMA

Pass C (`--conduit-to-dma`, split across 7 source files) converts Conduit
Tier 2 ops to raw hardware ops in seven phases:

1. **Attribute collection**: reads typed attributes (`producer_tile`,
   `consumer_tiles`, `element_type`, `depth`) directly from `conduit.create`
   ops. No string parsing is required.
2. **Tile cache**: walks `aie.tile` ops and builds a name→tile map.
3. **Buffer/lock allocation**: for each channel, allocates `depth` buffers
   on the consumer tile and two locks (producer and consumer).
4. **Flow creation**: emits `aie.flow` and `aie.shim_dma_allocation` for
   shim-to-tile transfers.
5. **ObjectFIFO_link lowering**: emits the BD chain for MemTile relay
   (depth-1 only; depth-N has known gap).
6. **Acquire/release lowering**: maps `conduit.acquire` to
   `aie.use_lock(consLock, AcquireGreaterEqual, N)` and `conduit.release`
   to `aie.use_lock(prodLock, Release, M)`.
7. **Cleanup erase**: erases `conduit.create`, `conduit.annotate`,
   `conduit.acquire`, `conduit.release`, and `conduit.subview_access`.

Five ObjectFIFO attributes are handled across these phases:
`repeat_count > 1` unrolls the BD chain N times with producer lock init
scaled to `depth × N`; core `AcquireGreaterEqual`/`Release` counts are
scaled by N in Phase 6 (§3.5). For link-distribute patterns, the MemTile
MM2S BD chain for each destination is also unrolled `linkDepth × repeat_count`
times with per-slice lock inits scaled accordingly. `iter_count` sets
`DMAStartOp.repeat_count` to `K − 1` and terminates the last BD with
`aie.end` (non-circular chain). `disable_synchronization` suppresses lock
allocation in Phase 3 and all `use_lock` emission; BD chains are still
emitted. `viaDMA` forces the DMA path for adjacent tiles (auto-set when
`dimensionsToStream` or `dimensionsFromStream` is non-empty, preventing
silent dimension loss on the shared-memory path). `dimensionsToStream` and
`dimensionsFromStream` apply N-D DMA traversal descriptors to MM2S and S2MM
BDs respectively; for distribute links, the destination conduit's
`producerDimensions` are applied to MemTile MM2S send BDs.

**Status**: compiled; 62 PASS / 0 XFAIL / 0 FAIL in lit suite. Correct for
depth=1, depth-2, broadcast (all consumer tiles), distribute, join, shared
memory, and async acquire programs across both AIE1 and AIE2 targets. All
five ObjectFIFO attributes verified in both isolated and combined configurations.
Tier 3 DMA op erasure (Steps 8e-8h) is implemented and tested.

### 4.3 Pass B: AIR Channel → Conduit

Pass B (`--air-channel-to-conduit`, `AirChannelToConduit.cpp`, 488 lines)
implements the structural rewrite from AIR Channel ops to Conduit Tier 3 ops.
The mapping is direct:

| AIR op | Conduit op |
|---|---|
| `air.channel @name [dims]` | `conduit.create {name, capacity}` |
| `air.channel.put async [deps] @name[] (...)` | `conduit.put_memref_async {name, offsets, sizes, strides, num_elems}` |
| `air.channel.get async [deps] @name[] (...)` | `conduit.get_memref_async {name, offsets, sizes, strides, num_elems}` |
| `air.wait_all [%deps]` | `conduit.wait_all %deps` |
| `air.wait_all async [%deps]` | `conduit.wait_all_async %deps` |

Pass B operates on post-specialized AIR, where the upstream
`specializeChannelBundle` step has already resolved SPMD channel indices to
scalar `[1, 1]` channels with compile-time constant positions. Descriptor field
extraction is correct for static integer constants; dynamic-shape programs
produce incorrect descriptors (a hard error is emitted).

**Key missing feature:** `air.channel.put async [%dep0, %dep1]` carries an
explicit dependency list forming the async DMA schedule graph. The current
Conduit Tier 3 async ops lack a variadic `deps` operand. Without this, Pass B
drops dependency ordering — losing the pipeline overlap information that makes
AIR programs fast. Adding `Variadic<Conduit_AsyncTokenType>:$deps` to both
Tier 3 async ops is the priority fix (§8).

**Status:** structural mapping tested (`air_channel_to_conduit.mlir` lit test
passes); not validated against the full mlir-air corpus.

---

## 5. Scope and Known Limitations

The current implementation handles depth-1 and depth-2 single-consumer
programs correctly (confirmed by IR structural comparison, §6). Depth-2
programs use a `memref<1xi32>` rotation counter with `scf.index_switch` for
ping-pong buffer selection, matching the stateful transform structure.
Broadcast, distribute, join, shared memory, async acquire, and CSDF patterns
all compile through the pipeline.

**Corpus coverage.** Of the 130 real objectFIFO files in the mlir-aie corpus,
103 compile end-to-end through the Conduit pipeline (79%). The 27 failures
comprise negative-test files, files using features Conduit does not yet
implement (packet switching, `register_process`, external buffers), known
bugs (`join-L2` resource parity), and 1 file correctly rejected by verifier
M8a. Five previously unimplemented attributes — `repeat_count`, `iter_count`,
`disable_synchronization`, `viaDMA`, and N-D DMA dimensions — are now fully
lowered, including the link-distribute case for `repeat_count`.

**Open correctness gaps.**

| Gap | Impact | Severity |
|---|---|---|
| Port attribute is a plain string | Typo (`"produce"` vs `"Produce"`) silently emits wrong-polarity locks → hardware deadlock with no diagnostic | High |
| Shared memory adjacency not detected | Unnecessary DMA for adjacent tiles | Efficiency |
| Packet switching not represented | Cannot lower packet-mode programs | High |
| Pass B dynamic strides | Incorrect descriptors for dynamic shapes (hard error emitted) | High |
| No DMA channel count validation | Programs exceeding 2+2 channels per tile compile without error | High |

**Recently fixed gaps (no longer open):** Broadcast depth>1 zero-flow (P0-B),
`repeat_count` (full BD chain unrolling + scaled locks, including link-distribute
MemTile MM2S: `linkDepth × repeat_count` BDs),
`iter_count` (non-circular BD chains), `disable_synchronization` (lock
suppression with BD chains intact), `viaDMA` (forced DMA path with
auto-inference from N-D dims), `dimensionsToStream`/`dimensionsFromStream`
(N-D DMA traversal on MM2S/S2MM BDs; distribute links use destination
fifo's `producerDimensions`), external buffer lowering in Pass A,
`nd_dma_distribute` duplicate BD emission in Phase 5.5 Case C,
`signalPassFailure` lambda-scope bug.

These gaps are documented in full detail in the project repository. No
hardware results are available yet (§6.4).

---

## 6. Evaluation

We evaluate Conduit along three dimensions: compile-time validity (L1),
semantic equivalence with the production stateful transform (L2), and
functional correctness on hardware (L3). No L3 results are available yet;
all claims below are L1 or L2.

### 6.1 Correctness Levels

**L1 — Compile-time structural validity.** The pipeline completes without
error; the AIE dialect verifier passes; no dangling Conduit ops remain.

**L2 — Semantic equivalence.** Verifiable by static IR inspection: (a) correct
`aie.flow` count, (b) BD chain length ≥ depth, (c) consistent lock polarity,
(d) buffers allocated on reachable tiles. Resource count differences between
Conduit and the stateful transform are expected when Conduit emits equivalent
but leaner structures (e.g., omitting dead ring-index counters at depth-1).

**L3 — Functional correctness.** Byte-identical outputs on hardware. This is
the only ground truth; L1 and L2 are compile-time proxies.

### 6.2 Corpus Coverage

The mlir-aie corpus contains 130 real objectFIFO files.

| Metric | Result |
|---|---|
| L1: compile success | 103/130 (79%) |
| L1: files requiring unimplemented features | 27/130 |
| L1: correctly rejected by M8a verifier | 1/130 |
| L2: `aie.flow` and `aie.dma_bd` match stateful transform | 66/66 comparable files |
| L2: full 6-op-type match | 1 file (`shim_dma_alloc_test.mlir`) |
| L3: byte-identical hardware output | 0 (pending) |

The 27 failing files use features not yet implemented (packet switching,
`register_process`, external buffers) or contain pre-existing join-L2
resource parity issues. The lit suite (62 tests, all pass) verifies IR
structural presence via FileCheck — covering isolated feature tests,
feature interaction tests, and edge cases — not lock polarity,
schedulability, or hardware execution. A FileCheck PASS does not imply
hardware correctness. Pass C emits `Acquire` for AIE1 and
`AcquireGreaterEqual` for AIE2 from the same Conduit IR, demonstrating
cross-generation portability.

### 6.3 Structural Comparison with Stateful Transform

For depth-1 single-consumer programs, the Conduit pipeline produces
structurally identical DMA ops (`aie.dma_bd`, `aie.next_bd`, `aie.flow`).
The `aie.use_lock` count is lower in Conduit by 4 per channel: IR inspection
confirms that the stateful transform's extra lock calls manage a ring-index
counter that is dead code at depth-1 (the counter value is never used to
index into the single-element buffer array). Conduit omits this dead code.

| Op type | Depth-1 | Depth-2 |
|---|---|---|
| `aie.dma_bd` | Match | Match |
| `aie.next_bd` | Match | Match |
| `aie.flow` | Match | Match |
| `aie.buffer` | -1 (no rotation counter needed) | Match |
| `aie.use_lock` | -4 (dead ring-index code omitted) | -8 (loop unrolling difference; runtime ops identical) |

For depth-2, the BD ring structure matches (ring closure, lock init values,
chain counts). Conduit emits 8 fewer `aie.use_lock` calls, which is
confirmed to be a loop-structure difference: Conduit uses a rotation counter
with `scf.index_switch` at step=1 (11 core use_locks) while the stateful
transform unrolls the loop at step=2 (19 core use_locks). Runtime lock
operation counts are identical (43 ops each). The lock count now matches
the stateful transform exactly; an earlier −2 lock deficit was a bug (missing
shim consumer locks for compute→shim conduits, fixed in `ConduitToDMARoute.cpp`).

| Program class | DMA structure | Lock delta | Status |
|---|---|---|---|
| Depth-1 SPSC | Match | -4 (dead code) | Verified (L2) |
| Depth-2 SPSC | Match | -8 use_locks (loop structure; runtime identical) | L1+L2 |
| Broadcast 1→N | Match (depth-1) | N/A | L1+L2a |
| Distribute 1→N | Per-dest locks/flows correct | N/A | L1+L2a |
| Async overlap | Not comparable | N/A | Lowering tested |

### 6.4 Planned Hardware Experiments

Hardware is available locally (VCK5000, AIE1). Four experiments are planned:

1. **Depth-1 correctness:** Verify byte-identical output vs. stateful transform
   on a passthrough kernel. IR analysis predicts full equivalence.
2. **Depth-2 double-buffering:** Verify rotation counter produces correct
   ping-pong alternation on hardware.
3. **Distribute-link:** Verify all N destination tiles receive correct slices.
4. **Async stall reduction:** Measure stall cycles for `acquire_async` +
   `wait_all` vs. blocking `acquire`, using AIE performance counters.

Experiment 4 is the primary performance result: it validates the core
contribution (§3.4) on hardware. Experiments 1-3 are correctness prerequisites.
Note: AIE performance counter event numbers and stall-cycle measurement
capabilities differ between AIE1 (VCK5000) and AIE2 (Phoenix/Strix);
experiment methodology will be adapted to the target generation.

---

## 7. Related Work

### 7.1 AMD AIE Compilation

**IRON** [1] is the direct predecessor from the same research group,
providing a Python-level API over `mlir-aie`'s ObjectFIFO model. Conduit
differs in two ways: it targets the MLIR IR level (a new dialect, not a new
frontend), and its scope includes `mlir-air` channel semantics. The primary
contribution relative to IRON is `conduit.acquire_async`: the bridge op that
lets a window grant and a DMA completion be waited on in parallel — a
limitation IRON identifies but does not address.

**ARIES** [2] proposes a unified MLIR-based compilation flow using an ADF
dialect, achieving 87% AIE efficiency. The key difference: ARIES replaces both
ObjectFIFO and AIR with its own graph model; Conduit preserves both and
provides a shared lowering path.

**CHARM** [12] and **CHARM 2.0** [13] are full-stack AIE+PL compilation systems
targeting matrix multiplication on AMD Versal ACAP. ARIES reports 1.17–1.59×
improvement over CHARM. These systems illustrate the ecosystem Conduit targets:
full-stack compilers exist; Conduit's contribution is a portable intermediate
dialect that unifies two source IRs below those stacks.

**MLIR-AIR** [4] defines the AIR dialect, achieving 78.7% compute efficiency
for matmul on AMD NPUs. MLIR-AIR is the source dialect for Conduit's Pass B;
its static SPMD specialization simplifies Pass B's scope to post-specialized
scalar channels. Conduit's `--conduit-depth-promote` aims to perform comparable
depth promotion in a single dialect-agnostic pass.

**SPARTA** [3] demonstrates horizontal diffusion stencil
on 384 AIE cores using sliding-window objectfifos (acquire=5, release=1) and
broadcast — precisely the cyclostatic and multi-consumer patterns Conduit
must support for real-world stencil workloads.

### 7.2 Async Token Models in MLIR

**IREE Stream dialect** [7] is the most mature async scheduling IR in
the MLIR ecosystem. IREE's `stream.async.*` → `stream.cmd.*` pipeline and
`stream.timepoint.join` are structurally analogous to Conduit's token model.
Key difference: IREE targets device-agnostic ML runtimes with CPU-managed
async execution; Conduit targets AIE hardware events (DMA engines, lock arbiters)
with no CPU async model. This motivates the separate `!conduit.dma.token` /
`!conduit.window.token` types rather than reusing `!async.token`.

**NVGPU dialect** demonstrates typed DMA descriptor modeling in MLIR via
`nvgpu.tma.descriptor`, using typed operands rather than string attributes — a
design direction Conduit should adopt via `FlatSymbolRefAttr`.

### 7.3 Dataflow and Streaming IRs

**Dato** [6] targets AMD Ryzen AI NPU with a virtual-machine
graph chain model, achieving 84% utilization for GEMM with 2.81× speedup on
attention. Dato's hardware results define the performance bar Conduit must
approach on the same hardware.

**Allo** [5] provides composable streaming with a `compose` primitive
analogous to `conduit.objectfifo_link`. **SPADA** [11] targets Cerebras WSE
with async-await constructs and formal deadlock semantics, independently
validating the async token approach for a different spatial vendor.
**"Towards Scheduling of Pipelined Dataflow Graphs in MLIR"** [8] is the
closest MLIR-community analog to `--conduit-depth-promote`: both use
token-based analysis to promote pipeline stages to overlapped execution.
**AXI4MLIR** [14] and **TL** [15] address the same core problem — mapping
tile programs to on-chip DMA — for different vendor architectures.

### 7.4 Positioning

Conduit does not compete with ARIES (which replaces both source dialects)
or IREE (which targets device-agnostic runtimes). It occupies the bridge
position between two existing AIE dialects, enabling shared optimization
passes. A main-track submission must match the hardware numbers of MLIR-AIR
and Dato.

---

## 8. Future Work

The workshop paper contributes the dialect design and IR-level equivalence
evidence. A main-track submission requires hardware validation of at least
two of the three novel elements (§1). We identify three priority items:

**Hardware stall reduction measurement.** The primary open question: does
`conduit.acquire_async` + `conduit.wait_all` measurably reduce stall cycles
vs. blocking `conduit.acquire`? The experiment compares the async pattern
(DMA fill and lock grant concurrent in hardware) against the blocking
equivalent on the same benchmark, measuring stall cycles via AIE performance
counters. The lowering is implemented and tested; hardware measurement is
pending. No existing system (ARIES, MLIR-AIR, Dato) measures this pattern.

**Cross-dialectical depth promotion.** Does `--conduit-depth-promote` on an
AIR-origin program (via Pass B) achieve comparable throughput improvement to
MLIR-AIR's 7-pass ping-pong transform? This validates contribution 2 and
is the strongest argument for a shared intermediate IR.

**CSDF BD chain specialization.** Cyclostatic programs compile through the
pipeline using `AcquireGreaterEqual(count)`, but BD chains are not yet
specialized per CSDF phase. An `access_pattern` attribute on `conduit.create`
would enable Pass C to generate `LCM(period)` BD blocks exploiting hardware
BD-level replay, reducing lock arbitration overhead.

**Design-level items.** `FlatSymbolRefAttr` channel references (replacing
name-as-string), async dependency operands on Tier 3 ops (required for Pass B
fidelity), packet switching support (`routing_mode` attribute), and
per-consumer depth lists are tracked in the repository.

---

## 9. Conclusion

We have presented Conduit, a 16-op MLIR dialect that bridges the ObjectFIFO
and AIR Channel programming models for AMD AIE spatial processors. Conduit's
key mechanism — cross-tier async token fan-in via `conduit.wait_all` —
enables hardware-concurrent DMA transfer and buffer-window acquisition, a
pattern inexpressible in either source dialect. The ownership-typed
`!conduit.window<T>` model makes the acquire-use-release protocol statically
verifiable through SSA def-use ordering.

Three compiler passes lower ObjectFIFO programs, AIR Channel programs
(static shapes), and Conduit IR to raw BD/DMA hardware ops. On a 130-file
ObjectFIFO corpus, 103 files compile end-to-end (79%). Five ObjectFIFO
attributes — `repeat_count`, `iter_count`, `disable_synchronization`,
`viaDMA`, and N-D DMA dimensions — are fully lowered through Pass C,
including the link-distribute `repeat_count` case. For depth-1
single-consumer programs, the generated IR is structurally equivalent to the
production stateful transform across all
six AIE op types.

Conduit is a design contribution with compile-time validation. No hardware
results are available yet. The primary open question — whether
`conduit.acquire_async` + `conduit.wait_all` reduces stall cycles on AIE
hardware — requires the planned hardware experiments (§6.4). Hardware
validation on the locally available VCK5000 is the immediate next step.

---

## Acknowledgments

[To be completed.]

---

## References

1. Hunhoff, Melber, Denolf, Bisca, Bayliss, Neuendorffer, Fifield, Lo, Vasireddy,
   James-Roxby, Keller. "Efficiency, Expressivity, and Extensibility in a
   Close-to-Metal NPU Programming Interface." IEEE FCCM 2025. arXiv:2504.18430.

2. "ARIES: An Agile MLIR-Based Compilation Flow for Reconfigurable Devices with
   AI Engines." FPGA '25. ACM DL:10.1145/3706628.3708870.

3. Singh, Khodamoradi, Denolf, Lo, Gómez-Luna, Melber, Bisca, Corporaal,
   Mutlu. "SPARTA: Spatial Acceleration for Efficient and Scalable Horizontal
   Diffusion Weather Stencil." ICS '23. arXiv:2303.03509.

4. Wang, Bayliss, Bisca, Blair, Denolf, Fifield, Melber, Neuendorffer, Richter
   et al. "From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with
   MLIR-AIR." arXiv:2510.14871, 2025.

5. Chen et al. "Allo: A Programming Model for Composable Accelerator Design."
   PLDI 2024. arXiv:2404.04815.

6. Fang et al. "Dato: A Task-Based Programming Model for Dataflow Accelerators."
   arXiv:2509.06794.

7. IREE Stream Dialect Documentation. iree.dev/reference/mlir-dialects/Stream/

8. "Towards Scheduling of Pipelined Dataflow Graphs in MLIR." FPGA 2026.
   ACM DL:10.1145/3748173.3779568.

9. Bilsen, Engels, Lauwereins, Peperstraete. "Cycle-Static Dataflow." IEEE
   Transactions on Signal Processing, 44(2):397–408, March 1996.
   DOI: 10.1109/78.485935.

10. Koh, Bodin. "K-Periodic Scheduling for Throughput-Buffering Trade-Off
    Exploration of CSDF." ACM Transactions on Embedded Computing Systems,
    22(1), October 2022. DOI: 10.1145/3559760.

11. Gianinazzi et al. "SPADA: A Spatial Dataflow Architecture Programming
    Language." arXiv:2511.09447, November 2025.

12. Zhuang, Lau, Ye, Yang, Lo, Denolf, Neuendorffer, Jones, Hu, Chen, Cong,
    Zhou. "CHARM: Composing Heterogeneous AcceleRators for Matrix Multiply on
    Versal ACAP Architecture." FPGA 2023, pp. 153–164.
    DOI: 10.1145/3543622.3573210.

13. Zhuang, Lau, Ye, Yang, Ji, Lo, Denolf, Neuendorffer, Jones, Hu, Shi,
    Chen, Cong, Zhou. "Composing Heterogeneous Accelerators for Deep Learning
    on Versal ACAP Architecture." ACM TRETS 17(3):51:1–51:31, 2024.
    (CHARM 2.0)

14. Agostini, Haris, Gibson, Jayaweera, Rubin, Tumeo, Abellán, Cano, Kaeli.
    "AXI4MLIR: User-Driven Automatic Host Code Generation for Custom
    AXI-Based Accelerators." CGO 2024. arXiv:2312.14821.

15. Li, Bai, Wang, Dangi, Zhang, Tan, Lan, Wong, Mitra. "TL: Automatic
    End-to-End Compiler of Tile-Based Languages for Spatial Dataflow
    Architectures." arXiv:2512.22168, December 2025.

---

## Appendix: Comparison Matrix

The table below shows selected rows from the 22-primitive × 13-system comparison matrix. The full matrix (all 22 primitives across all 13 systems, with adoption rationale and urgency ratings) is available at `artifacts/comparison/extended_matrix.csv` in the repository.

| Primitive | Conduit | mlir-aie | mlir-air | Dato | Allo | IREE Stream | Urgency |
|---|---|---|---|---|---|---|---|
| `channel.create` / `conduit.create` | op | op (`objectfifo`) | op (`air.channel`) | type (`Stream`) | implicit (`compose`) | op (`resource.alloca`) | must-have |
| `put_memref` / N-D DMA descriptor | op | attr (`dimensionsToStream`) | op (`channel.put [off][sz][st]`) | type (`Layout + DMA-fold`) | implicit (`buffer_at`) | op (`resource.map`) | must-have |
| `objectfifo.link` / join-distribute routing | op | op (`objectfifo.link`) | none | implicit (`VMG chain`) | none | none | must-have |
| `acquire / release` (windowed in-place) | op | op (`objectfifo.acquire`) | none (emulated) | none | attr (`buffer_at`) | none | must-have |
| `put_async` / async token returns | op | none (lock emulation) | op (`channel.put async`) | implicit (task dep) | none | op (`async.execute`) | high |
| `wait / wait_all` / timepoint join | op | op (`use_lock Acquire`) | op (`air.wait_all`) | implicit (task ordering) | none | op (`timepoint.join`) | must-have |
| `register_external_buffers` (host DDR) | op | op | none | none | none | op (`resource.import`) | high |
| `subview_access` / indexed buffer extraction | op | op (`objectfifo.subview.access`) | none | none | none | op (`resource.map subrange`) | must-have |
| `routing_mode = "packet"` / packet multiplexing | planned | attr (via `viaDMA`) | attr (`channel_type = "dma_packet"`) | implicit | none | none | medium |
| CSDF / `access_pattern` | planned | internal (`--aie-objectFifo-stateful-transform`) | none | none | none | none | medium |

**Key differentiators:** Conduit is the only system in this matrix that
(1) enables cross-tier fan-in across DMA completion and lock-grant events
in a single `wait_all` call, (2) uses ownership-typed buffer handles
(`!conduit.window<T>`) with SSA-enforced acquire-before-use, and
(3) makes MemTile relay a first-class verifiable IR construct.
