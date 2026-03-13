# Conduit IR — Dialect Design and RFC Extensions

Three first-class extensions to the core Conduit dialect that cover
critical gaps identified in the upstream mlir-aie and mlir-air corpora.

---

## Dialect Design — The 16-op minimal dialect

The compiled Conduit dialect contains 16 ops organized into three groups.
This section documents the current compiled state; the RFC sections below
describe additional ops whose lowering is planned but not yet implemented.

### Infrastructure ops (tier-agnostic)

These ops declare channels and attach lowering metadata.  Used by all programs.

```mlir
conduit.create {name = "c1", capacity = 64 : i64}

conduit.annotate {name = "c1", key = "producer_tile", value = "tile(0,0)"}

conduit.register_external_buffers {name = "shim_chan",
                                   num_buffers = 2 : i64,
                                   base_addr   = 0 : i64}

conduit.objectfifo_link {srcs = ["of_in"],
                         dsts = ["of_out0", "of_out1"],
                         mode = "distribute",
                         memtile = "tile(0,1)",
                         offsets = array<i64: 0, 1024>}
```

### Tier 2 — Buffer-window ops (ObjectFIFO path)

Produced by Pass A (`--objectfifo-to-conduit`).  Lowered by Pass C
(`--conduit-to-dma`) to `aie.use_lock` + `aie.buffer` + `aie.dma_bd`.

```mlir
conduit.acquire {name = "input", count = 1 : i64}              // blocking
conduit.release {name = "input", count = 1 : i64}              // blocking
%buf = conduit.subview_access {name = "input", index = 0 : i64} : memref<64xi32>

%tok = conduit.acquire_async {name = "input", count = 1 : i64}
           : !conduit.async.token                              // non-blocking bridge op
%tok = conduit.release_async {name = "output", count = 1 : i64}
           : !conduit.async.token                              // non-blocking
```

### Tier 3 — Memref-DMA ops (AIR Channel path)

Will be produced by Pass B (`--air-channel-to-conduit`, not yet implemented).

```mlir
// Blocking forms
conduit.put_memref {name = "input", num_elems = 64 : i64,
                    offsets = array<i64: 0, 0>, sizes = array<i64: 8, 8>,
                    strides = array<i64: 16, 1>}
conduit.get_memref {name = "output", num_elems = 64 : i64,
                    offsets = array<i64: 0>, sizes = array<i64: 64>,
                    strides = array<i64: 1>}

// Async forms (return !conduit.async.token)
%t0 = conduit.put_memref_async {name = "input", num_elems = 64 : i64,
                                 offsets = array<i64: 0>, sizes = array<i64: 64>,
                                 strides = array<i64: 1>} : !conduit.async.token
%t1 = conduit.get_memref_async {name = "output", num_elems = 64 : i64,
                                 offsets = array<i64: 0>, sizes = array<i64: 64>,
                                 strides = array<i64: 1>} : !conduit.async.token

// Synchronization
conduit.wait %t0 : !conduit.async.token
conduit.wait_all %t0, %t1

%merged = conduit.wait_all_async %t0, %t1
              : (!conduit.async.token, !conduit.async.token) -> !conduit.async.token
```

### The cross-tier bridge: acquire_async

`conduit.acquire_async` returns a `!conduit.async.token` that represents
*pending permission to access a buffer window*.  This token is compatible
with Tier 3 DMA tokens in `conduit.wait_all`, so the hardware can satisfy
a buffer-window grant (lock wait) and a DMA transfer in parallel.
This is the unifying mechanism of the dialect.

**What was deliberately excluded:**
- `conduit.put` / `conduit.get` / `conduit.peek` / `conduit.advance` /
  `conduit.prefill` — Tier 1 scalar token ops used only in the DSL interpreter
  (`tools/conduit_interpreter.py`).  Removed from the compiled MLIR dialect.
  The DSL interpreter retains its own `PUT`/`GET`/`PEEK` syntax.
- `conduit.put_async` / `conduit.get_async` — scalar async variants; removed
  with the scalar model.
- `conduit.merge` / `conduit.fork` — no hardware lowering, no implementation
  plans.  Use `conduit.objectfifo_link` with `mode="join"` / `mode="distribute"`
  for the hardware-mediated fan-in/fan-out pattern.
- `conduit.chain` — semantically broken in SSA: SSA values are immutable;
  cannot add deps to an existing token.  Use `conduit.wait_all_async` instead.
- `conduit.status` — no lowering target; removed for minimality.

### Cross-tier MLIR example

```mlir
// Producer endpoint: shim DMA fills from DDR (Tier 3, non-blocking)
%dma_tok = conduit.put_memref_async {name = "input",
               num_elems = 9 : i64,
               offsets = array<i64: 0, 0>, sizes = array<i64: 3, 3>,
               strides = array<i64: 16, 1>} : !conduit.async.token

// Consumer endpoint: core requests output window (Tier 2, non-blocking)
%acq_tok = conduit.acquire_async {name = "output", count = 1 : i64}
               : !conduit.async.token

// Hardware satisfies both in parallel; software waits for both
conduit.wait_all %dma_tok, %acq_tok

// Both conditions met: access both endpoints in-place
%in  = conduit.subview_access {name = "input",  index = 0 : i64} : memref<3x3xi16>
%out = conduit.subview_access {name = "output", index = 0 : i64} : memref<9xi32>
// ... compute stencil ...
conduit.release {name = "input",  count = 1 : i64}
conduit.release {name = "output", count = 1 : i64}
```

---

## Extension 1 — `conduit.objectfifo_link`

### Signature

```
conduit.objectfifo_link srcs=[%<name>, ...]
                        dsts=[%<name0>, %<name1>, ...]
                        mode="distribute" | "join"
                        memtile=<string>
                        [offsets=[<int>, ...]]
                        [lock_id=<int>]
```

`srcs` is always a list (even in distribute mode where there is exactly one
source).  This matches the TableGen stub, the text-transform tools, and RFC-000.
Earlier versions of this spec used `src=` (singular) for distribute mode; that
form is deprecated.

### Semantics

`conduit.objectfifo_link` connects one or more source conduits to one or more
destination conduits through a MemTile relay buffer, enabling:

- **distribute** (`1 src → N dsts`): tokens from the single source are
  dispatched to each destination conduit at byte offsets specified by the
  `offsets` list.  Each destination receives a contiguous sub-region of the
  source buffer.  The MemTile is responsible for arbitrating BD chains.

- **join** (`N srcs → 1 dst`): tokens from multiple source conduits are
  concatenated, at byte offsets from the `offsets` list, into a single
  destination conduit.

**Invariants:**
- In distribute mode: `len(dsts) == len(offsets)` (one offset per destination).
  Offsets denote byte start positions within the source buffer.
- In join mode: `len(srcs) == len(offsets)` (one offset per source).
  Offsets denote byte insertion positions within the destination buffer.
- `memtile` names the tile string (e.g. `"tile(0,1)"`) that performs the relay.
- `lock_id`, when present, passes through to the generated `aie.objectfifo.link`
  to preserve synchronisation semantics.
- All named conduits must have been created with `CREATE` or `conduit.create`
  before the link is declared.

### Example — distribute (1 → 2)

**Conduit DSL:**
```
CREATE of_in  capacity=2048
CREATE of_out0 capacity=1024
CREATE of_out1 capacity=1024
ANNOTATE of_in   lower_to=objectfifo tile="(0,0)"
ANNOTATE of_out0 lower_to=objectfifo tile="(0,2)"
ANNOTATE of_out1 lower_to=objectfifo tile="(1,2)"

conduit.objectfifo_link srcs=[%of_in]
                        dsts=[%of_out0, %of_out1]
                        mode="distribute"
                        memtile="(0,1)"
                        offsets=[0, 1024]
```

**AIE MLIR lowering:**
```mlir
aie.objectfifo @of_in  (%tile_0_0, {%tile_0_1}, 4 : i32) : !aie.objectfifo<memref<2048xi32, 1>>
aie.objectfifo @of_out0(%tile_0_1, {%tile_0_2}, 4 : i32) : !aie.objectfifo<memref<1024xi32, 1>>
aie.objectfifo @of_out1(%tile_0_1, {%tile_1_2}, 4 : i32) : !aie.objectfifo<memref<1024xi32, 1>>
aie.objectfifo.link [@of_in] -> [@of_out0, @of_out1] ([] [0, 1024])
```

### Example — join (2 → 1)

**Conduit DSL:**
```
CREATE of_src0 capacity=16
CREATE of_src1 capacity=16
CREATE of_dst  capacity=32
ANNOTATE of_src0 lower_to=objectfifo tile="(0,0)"
ANNOTATE of_src1 lower_to=objectfifo tile="(1,0)"
ANNOTATE of_dst  lower_to=objectfifo tile="(0,2)"

conduit.objectfifo_link srcs=[%of_src0, %of_src1]
                        dsts=[%of_dst]
                        mode="join"
                        memtile="(0,1)"
                        offsets=[0, 16]
```

**AIE MLIR lowering:**
```mlir
aie.objectfifo @of_src0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16xi32, 1>>
aie.objectfifo @of_src1(%tile_1_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16xi32, 1>>
aie.objectfifo @of_dst (%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32xi32, 1>>
aie.objectfifo.link [@of_src0, @of_src1] -> [@of_dst] ([0, 16] [])
```

### Lowering Notes

**First-class op approach:**
The lowering pass matches `conduit.objectfifo_link` and emits:
1. `aie.objectfifo` declarations for src and all dsts (with memtile as producer
   tile for the dst fifos).
2. `aie.objectfifo.link [@srcs] -> [@dsts] ([join_offsets] [dist_offsets])`.
   - distribute: join_offsets = `[]`, dist_offsets = link `offsets` attribute.
   - join: join_offsets = link `offsets` attribute, dist_offsets = `[]`.

**Annotation / fallback approach:**
Without the first-class op, `ANNOTATE %of lower_to=objectfifo_link` on a
conduit that acts as a relay causes the lowering script to emit a
`aie.objectfifo.link` using positional defaults (offset 0 for each destination).
This is correct only for non-overlapping equal-size splits and emits a warning.

**AIR mapping:**
`conduit.objectfifo_link` has no direct AIR equivalent. In AIR, fan-out is
expressed by declaring a channel with `broadcast_shape` or by multiple
independent `air.channel` declarations. The fallback emits one `air.channel`
per destination and a comment flagging the missing MemTile relay.

---

## Extension 2 — `conduit.put_memref` / `conduit.bulk_put`

### Signatures

```
conduit.put_memref %chan, %buf
    {offsets=[<int|SSA>, ...],
     sizes=[<int|SSA>, ...],
     strides=[<int|SSA>, ...],
     num_elems=<int>}

conduit.get_memref %chan, %buf
    {offsets=[<int|SSA>, ...],
     sizes=[<int|SSA>, ...],
     strides=[<int|SSA>, ...],
     num_elems=<int>}
```

`conduit.bulk_put` is a shorthand alias for `conduit.put_memref` when all
offsets/strides are compile-time constants.

### Semantics

`conduit.put_memref` transfers a strided tile from a memref into a conduit,
matching the N-D DMA descriptor form used by `air.channel.put`:

```
air.channel.put async [...] @chan[%i, %j]
    (%buf[%o0, %o1] [%s0, %s1] [%r0, %r1]) : (memref<MxNxT>)
```

Fields:
- `offsets`: starting indices into the source memref for each dimension.
- `sizes`: number of elements to transfer per dimension.
- `strides`: stride in elements between successive transfers per dimension.
- `num_elems`: total flat element count (`product(sizes)`); used for BD
  descriptor generation and validation.
- `%buf`: the source memref SSA value.
- `%chan`: the target conduit.

The op is **non-blocking by default** (returns void). Use
`conduit.put_memref_async` for the async token-returning variant (see
Extension 3).

`conduit.get_memref` is the matching consumer-side op.  Fields are identical;
`%buf` is the destination memref.

### Example

**Conduit DSL:**
```
CREATE ch0 capacity=4
ANNOTATE ch0 lower_to=channel dims=[2,2]

// Transfer a 32x32 tile starting at offset (r*32, 0) with column-major stride:
conduit.put_memref %ch0, %A {
    offsets=[%row_off, 0],
    sizes=[32, 32],
    strides=[64, 1],
    num_elems=1024
}

conduit.get_memref %ch0, %C {
    offsets=[],
    sizes=[],
    strides=[],
    num_elems=1024
}
```

**AIR MLIR lowering:**
```mlir
air.channel @ch0 [2, 2]

// put side:
%tok = air.channel.put async [%dep]  @ch0[%i, %j]
    (%A[%row_off, %c0] [%c32, %c32] [%c64, %c1]) {id = 1 : i32}
    : (memref<128x64xf32>)

// get side:
%tok2 = air.channel.get async [%tok] @ch0[%i, %j]
    (%C[] [] []) {id = 2 : i32}
    : (memref<32x32xf32, 2>)
```

**AIE MLIR lowering (objectfifo backend):**
When lowering to objectfifo, `conduit.put_memref` with non-trivial strides maps
to `aie.objectfifo.create` with `dimensionsToStream`:
```mlir
aie.objectfifo @ch0(%src_tile dimensionsToStream
    [<size = 32, stride = 64>, <size = 32, stride = 1>],
    {%dst_tile}, 4 : i32) : !aie.objectfifo<memref<1024xi32>>
```

### Lowering Notes

**First-class op approach:**
The lowering pass reads `offsets`, `sizes`, `strides` from the op attributes
and emits either:
- `air.channel.put` with the full `(%buf[offsets] [sizes] [strides])` syntax.
- `aie.objectfifo` `dimensionsToStream`/`dimensionsFromStream` attribute lists
  built from the descriptor.

If `num_elems` mismatches `product(sizes)`, the lowering aborts with an error.

**Annotation / fallback approach:**
`ANNOTATE %c put_descriptor="<off>;<sz>;<stride>"` on a conduit causes the
lowering script to parse the semicolon-separated descriptor and emit the
matching AIR syntax. This is lossy for SSA offsets (replaced with `%c0`
placeholders and a warning).

**Empty descriptor:**
An empty descriptor (`offsets=[], sizes=[], strides=[]`) lowers to
`(%buf[] [] [])` in AIR — the full-buffer transfer form — with `num_elems`
used only for BD length.

---

## Extension 3 — Async Token Handles

### Signatures (surviving ops after dialect trimming)

```
// Async Tier 3 ops:
%tok = conduit.put_memref_async {name = "ch", num_elems = N : i64,
           offsets = [...], sizes = [...], strides = [...]} : !conduit.async.token
%tok = conduit.get_memref_async {name = "ch", num_elems = N : i64,
           offsets = [...], sizes = [...], strides = [...]} : !conduit.async.token

// Async Tier 2 bridge ops:
%tok = conduit.acquire_async {name = "ch", count = N : i64} : !conduit.async.token
%tok = conduit.release_async {name = "ch", count = N : i64} : !conduit.async.token

// Synchronization:
conduit.wait %tok : !conduit.async.token
conduit.wait_all %tok0, %tok1, ...
%merged = conduit.wait_all_async %tok0, %tok1, ... :
    (!conduit.async.token, ...) -> !conduit.async.token

// Removed (dialect trimming 2026-03-12):
//   conduit.put_async, conduit.get_async — scalar async; Tier 1 removed
//   conduit.chain — broken SSA semantics; use wait_all_async instead
```

All `*_async` variants return `!conduit.async.token`.

### Semantics

**`conduit.put_memref_async` / `conduit.get_memref_async`:**
Non-blocking N-D DMA submission.  The returned token represents the completion
of the DMA BD submission.  The caller must not touch the source/destination
buffer until the token is waited on.

**`conduit.acquire_async` (Tier 2 bridge):**
Non-blocking buffer-window acquisition.  The returned token represents pending
permission to access the buffer window.  After `conduit.wait` or
`conduit.wait_all` on this token, `conduit.subview_access` is valid.
This token is compatible with Tier 3 DMA tokens in `conduit.wait_all` —
the hardware can satisfy a lock grant and a DMA transfer in parallel.

**`conduit.release_async` (Tier 2 bridge):**
Non-blocking buffer-window release.  The returned token signals when the
release has propagated to the producer's lock.

**`conduit.wait %tok`:**
Blocking wait until the operation represented by `%tok` completes.  Equivalent
to `conduit.wait_all %tok` (blocking, no result).

**`conduit.wait_all [%tok0, ...]`:**
Blocking wait on a list of tokens.  All specified operations must complete
before execution continues.

**`conduit.wait_all_async [%tok0, ...]`:**
Returns a new token that is signaled when all input tokens complete.
Non-blocking; useful for constructing dependency chains.  Subsumes the
removed `conduit.chain` op: use `conduit.wait_all_async [%tp] -> %tc_tok`
where the result token is then used as a dependency for subsequent ops.

### Example

**MLIR (cross-tier unified pattern):**
```mlir
// Non-blocking DMA transfer (Tier 3):
%t0 = conduit.put_memref_async {name = "ch_a", num_elems = 128 : i64,
                                 offsets = array<i64: 0>,
                                 sizes   = array<i64: 128>,
                                 strides = array<i64: 1>} : !conduit.async.token

// Non-blocking window acquisition (Tier 2 bridge):
%t1 = conduit.acquire_async {name = "ch_b", count = 1 : i64}
          : !conduit.async.token

// Wait for both in parallel:
conduit.wait_all %t0, %t1

// Merged token for downstream dep:
%t_all = conduit.wait_all_async %t0, %t1 :
             (!conduit.async.token, !conduit.async.token) -> !conduit.async.token
conduit.wait %t_all : !conduit.async.token
```

**AIR MLIR lowering:**
```mlir
air.channel @ch_a [2, 2]
air.channel @ch_b [2, 2]

// Non-blocking puts (async deps encoded in bracket list):
%t0 = air.channel.put async [] @ch_a[%i, %j]
    (%src_a[] [] []) {id = 1 : i32} : (memref<128xi32>)

%t1 = air.channel.put async [%t0] @ch_b[%i, %j]
    (%src_b[] [] []) {id = 2 : i32} : (memref<128xi32>)

// Wait (air.wait_all with no result = blocking sync):
air.wait_all [%t0, %t1]

// Merged token:
%t_all = air.wait_all async [%t0, %t1]
air.wait_all [%t_all]
```

**AIE lock-based token ordering:**
AIE does not have an explicit async token type; dependency ordering is encoded
via lock sequencing.  `conduit.acquire_async` and `conduit.release_async` lower
to non-blocking poll forms of `aie.use_lock`:
```mlir
aie.use_lock(%consLock, AcquireGreaterEqual, 1)  // acquire_async result
aie.use_lock(%prodLock, Release, 1)              // release_async result
```
On AIE, DMA BDs and lock operations run in hardware concurrently; software waits
on whichever completes last.

### Lowering Notes

**First-class op approach:**
- All `conduit.put_memref_async` / `conduit.get_memref_async` ops are collected
  in a dataflow analysis pass.
- `conduit.wait` emits `air.wait_all [...]` (blocking) or a lock-release
  sequence for the AIE backend.
- `conduit.wait_all_async` emits `air.wait_all async [...]`.
- `conduit.acquire_async` / `conduit.release_async` lower to non-blocking
  `aie.use_lock` poll forms on the AIE backend.

**Annotation / fallback approach:**
Without async ops, the lowering script emits synchronous `air.channel.put`
(no `async` keyword) and records a warning that dependency ordering is not
represented.  This is correct for single-threaded host-side dispatch but
loses pipeline overlap for DMA-pipelined programs.

---

## Lowering Notes — Cross-Feature Summary

| Feature | AIE first-class | AIR first-class | Fallback (annotation) |
|---|---|---|---|
| `conduit.objectfifo_link` | `aie.objectfifo.link` | N/A (multi-channel decl) | offset=0 warning |
| `conduit.put_memref` | `aie.objectfifo` dimensionsToStream | `air.channel.put [off][sz][st]` | %c0 placeholder warning |
| `conduit.put_memref_async` | `aie.use_lock` (non-blocking poll) | `air.channel.put async [deps]` | synchronous fallback warning |
| `conduit.acquire_async` | `aie.use_lock` (AcquireGreaterEqual poll) | N/A (AIR uses channel.put/get async) | blocking acquire fallback |
| `conduit.wait_all_async` | lock sequence | `air.wait_all async [...]` | `air.wait_all` (blocking) |

**General guidance:**
- Prefer first-class ops when targeting a single backend; the lowering is
  lossless and metadata-preserving.
- Use annotation fallbacks only for quick porting; always audit the warning
  log for structural information that was silently dropped.
- For programs mixing AIE objectfifo and AIR channel patterns, use
  `conduit.objectfifo_link` with `mode="distribute"` at the boundary and
  AIR channel ops on the host/segment side.
