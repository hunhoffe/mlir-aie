# Conduit Dialect

Conduit is a 16-op MLIR dialect that acts as a portable bridge between
`mlir-aie` ObjectFIFO semantics (buffer-window, lock-guarded) and `mlir-air`
Channel semantics (N-D DMA descriptor, async token).

**Current state (2026-03-12):**
- Dialect compiled and registered in `aie-opt`
- Pass A (`--objectfifo-to-conduit`): 7/7 lit tests passing
- Pass C (`--conduit-to-dma`): 7/7 lit tests passing; correct for depth=1, single-consumer
- No hardware validation yet — Phoenix/Strix hardware run is the next milestone

---

## Quick start

```bash
# Dialect roundtrip (verifies all 16 ops parse and print)
aie-opt test/Dialect/Conduit/roundtrip.mlir

# Pass A: lift objectfifo → conduit
aie-opt --objectfifo-to-conduit \
  test/Dialect/Conduit/objectfifo_to_conduit_depth_one.mlir

# Pass A + C: full pipeline → raw hardware BD/DMA ops
aie-opt --objectfifo-to-conduit --conduit-to-dma \
  test/Dialect/Conduit/conduit_to_dma_depth_one.mlir

# Compare against the existing stateful transform (correctness check)
aie-opt --objectfifo-to-conduit --conduit-to-dma \
  programming_examples/basic/passthrough_dmas/aie2.mlir

aie-opt --aie-objectFifo-stateful-transform \
  programming_examples/basic/passthrough_dmas/aie2.mlir

# Run all 7 lit tests
/proj/rdi/staff/ehunhoff/packages/bin/lit build/test/Dialect/Conduit/ -v
```

---

## Source layout

```
include/aie/Dialect/Conduit/IR/
  Conduit.td           ← TableGen: all 16 ops + !conduit.async.token type
  ConduitDialect.h     ← C++ header (includes generated .h.inc files)

lib/Dialect/Conduit/IR/
  ConduitOps.cpp       ← dialect init + objectfifo_link verifier

include/aie/Dialect/Conduit/Transforms/
  Passes.td            ← Pass A + C TableGen declarations
  ConduitPasses.h      ← factory functions, registerConduitPasses

lib/Dialect/Conduit/Transforms/
  ObjectFifoToConduit.cpp   ← Pass A: aie.objectfifo.* → conduit.*
  ConduitToDMA.cpp          ← Pass C: conduit.* → aie.dma_bd / aie.lock / aie.buffer

test/Dialect/Conduit/
  roundtrip.mlir                         ← all 16 ops
  invalid.mlir                           ← objectfifo_link verifier errors
  objectfifo_to_conduit_depth_one.mlir   ← Pass A: depth-1 single fifo
  objectfifo_to_conduit_distribute.mlir  ← Pass A: 1→3 distribute link
  objectfifo_to_conduit_join.mlir        ← Pass A: N→1 join link
  conduit_to_dma_depth_one.mlir          ← Pass A+C: depth-1 end-to-end
  conduit_to_dma_distribute.mlir         ← Pass A+C: distribute end-to-end
```

---

## The 16-op dialect

### Infrastructure ops

| Op | What it does |
|---|---|
| `conduit.create {name, capacity}` | Declare a named FIFO channel |
| `conduit.annotate {name, key, value}` | Attach a lowering hint (tile, element type, depth) |
| `conduit.objectfifo_link {srcs, dsts, mode, memtile, offsets?, lock_id?}` | MemTile relay: distribute or join |
| `conduit.register_external_buffers {name, num_buffers, base_addr}` | Register host-side shim DMA buffers |

### Tier 2 — Buffer-window ops (ObjectFIFO path)

Produced by Pass A. Lowered by Pass C to `aie.use_lock` + `aie.buffer` + `aie.dma_bd`.

| Op | What it does |
|---|---|
| `conduit.acquire {name, count}` | Reserve N buffer slots, blocking |
| `conduit.release {name, count}` | Release M buffer slots, blocking |
| `conduit.subview_access {name, index} : T` | Access element N of the held window |
| `%tok = conduit.acquire_async {name, count} : !conduit.async.token` | Non-blocking acquire; returns a completion token |
| `%tok = conduit.release_async {name, count} : !conduit.async.token` | Non-blocking release; returns a completion token |

### Tier 3 — Memref-DMA ops (AIR Channel path)

Will be produced by Pass B (not yet implemented). Lowered by Pass C to `aie.dma_bd`.

| Op | What it does |
|---|---|
| `conduit.put_memref {name, num_elems, offsets, sizes, strides}` | N-D DMA send, blocking |
| `conduit.get_memref {name, num_elems, offsets, sizes, strides}` | N-D DMA receive, blocking |
| `%tok = conduit.put_memref_async {...} : !conduit.async.token` | Non-blocking DMA send |
| `%tok = conduit.get_memref_async {...} : !conduit.async.token` | Non-blocking DMA receive |
| `conduit.wait %tok` | Block until one token completes |
| `conduit.wait_all %tok0, %tok1, ...` | Block until all tokens complete |
| `%tok = conduit.wait_all_async %tok0, ... : (...) -> !conduit.async.token` | Merge tokens, non-blocking |

---

## The key design insight

`conduit.acquire_async` is the bridge between the two tiers. It returns
a `!conduit.async.token` representing *pending permission to access a buffer
window* — the same type returned by `conduit.put_memref_async`, which
represents a *pending DMA transfer*. Passing both to `conduit.wait_all`
lets the hardware satisfy them in parallel:

```mlir
// DMA fills the input buffer from DDR (Tier 3, non-blocking)
%dma_tok = conduit.put_memref_async {name = "input", num_elems = 9 : i64,
               offsets = array<i64: 0, 0>, sizes = array<i64: 3, 3>,
               strides = array<i64: 16, 1>} : !conduit.async.token

// Core requests an output window (Tier 2, non-blocking)
%acq_tok = conduit.acquire_async {name = "output", count = 1 : i64}
               : !conduit.async.token

// Hardware satisfies the DMA fill and the lock grant in parallel
conduit.wait_all %dma_tok, %acq_tok

// Both conditions met — proceed
%in  = conduit.subview_access {name = "input",  index = 0 : i64} : memref<3x3xi16>
%out = conduit.subview_access {name = "output", index = 0 : i64} : memref<9xi32>
// ... stencil compute ...
conduit.release {name = "input",  count = 1 : i64}
conduit.release {name = "output", count = 1 : i64}
```

The blocking forms are sugar: `conduit.acquire` = `conduit.acquire_async` +
`conduit.wait`. Use the async form when overlapping with DMA.

---

## Pass A: ObjectFIFO → Conduit

`--objectfifo-to-conduit` in `lib/Dialect/Conduit/Transforms/ObjectFifoToConduit.cpp`

| ObjectFIFO op | Conduit op |
|---|---|
| `aie.objectfifo @name(prod, {cons}, depth)` | `conduit.create` + `conduit.annotate` × 4 |
| `aie.objectfifo.acquire(Produce/Consume, N)` | `conduit.acquire {name, count=N}` |
| `aie.objectfifo.release(Produce/Consume, M)` | `conduit.release {name, count=M}` |
| `aie.objectfifo.subview.access %sv[i]` | `conduit.subview_access {name, index=i} : T` |
| `aie.objectfifo.link [@srcs] -> [@dsts] (...)` | `conduit.objectfifo_link {srcs, dsts, mode, memtile, offsets}` |

**Known gaps in Pass A:**

| Gap | Detail |
|---|---|
| `subview_access` result type | Current Pass A emits `memref.alloc` as a placeholder — the result is not wired to the DMA-filled buffer. Fix: preserve the subview SSA value instead of erasing the acquire op immediately. |
| Memtile heuristic | Uses the consumer tile of the first source fifo as the memtile coordinate. Should query the device model for actual MemTile tiles. |
| Link mode detection | Assumes 1-src → N-dst is distribute, N-src → 1-dst is join. Ambiguous N→N defaults to distribute. |

## Pass C: Conduit → BD/DMA

`--conduit-to-dma` in `lib/Dialect/Conduit/Transforms/ConduitToDMA.cpp`

| Conduit op | Hardware op |
|---|---|
| `conduit.create` + `conduit.annotate` | `aie.buffer` + `aie.lock` per tile |
| `conduit.acquire` | `aie.use_lock(consLock, AcquireGreaterEqual, N)` |
| `conduit.release` | `aie.use_lock(prodLock, Release, M)` |
| `conduit.subview_access` | erased (placeholder; not yet wired to DMA buffer) |
| `conduit.objectfifo_link` | `aie.objectfifo.link` re-emitted as hardware relay |

**Known gaps in Pass C:**

| Gap | Detail |
|---|---|
| Depth-2 BD chain (top priority) | Only generates a depth-1 BD loop. Double-buffering requires N descriptor blocks. |
| `acquire_async` lowering | `conduit.acquire_async` exists in the dialect but Pass C does not lower it — only the blocking `conduit.acquire` is handled. |
| Multi-consumer broadcast | Only allocates buffers/locks for `consumerTiles[0]`. N consumers need N sets. |
| Buffer type | Uses `memref<capacity/depth x i32>` as a placeholder. Should parse the `element_type` annotation. |
| Lock ID collision | Lock IDs are hardcoded to 0 and 1. Multiple conduits on the same tile will collide. |
| Shim DMA symbol | `aie.shim_dma_allocation` requires a symbol matching an `aie.objectfifo`. Since Pass A erases objectfifos, the symbol must be re-registered. |
| MM2S channel chain | `dma_start` ops for channels > 0 are emitted inside the wrong block. Each must be a separate terminator chain. |

---

## Standard annotate keys

Pass A emits these `conduit.annotate` ops for each channel. Pass C reads them:

| Key | Format | Example |
|---|---|---|
| `producer_tile` | `"tile(col,row)"` | `"tile(0,0)"` |
| `consumer_tile_N` | `"tile(col,row)"` | `"tile(0,2)"` |
| `element_type` | MLIR type string | `"memref<64xi32>"` |
| `depth` | integer string | `"2"` |

---

## What is not in this dialect (and why)

| Removed op | Reason |
|---|---|
| `conduit.put` / `conduit.get` / `conduit.peek` / `conduit.advance` / `conduit.prefill` | Tier 1 scalar DSL ops — belong only in `tools/conduit_interpreter.py`, no hardware lowering |
| `conduit.put_async` / `conduit.get_async` | Scalar async variants; removed with the scalar model |
| `conduit.merge` / `conduit.fork` | No hardware lowering planned; use `conduit.objectfifo_link` with `mode="join"/"distribute"` |
| `conduit.chain` | Broken SSA semantics: cannot mutate an existing token's dependency set. Use `conduit.wait_all_async` |
| `conduit.status` | No lowering target |

---

## Next steps

1. **Fix Pass C depth-2 BD chain** — the highest-priority gap for hardware validation.
2. **Run `compare_lowering_resources.py`** — verify Conduit path produces equivalent
   output to `--aie-objectFifo-stateful-transform` on real benchmarks.
3. **Validate on Phoenix hardware** — `aiesimulator` or on-device run.
4. **Implement Pass B** (`--air-channel-to-conduit`) after A+C validate on hardware.

See `../../../RESEARCH_PLAN.md` for the full three-track, 2-year research plan.
