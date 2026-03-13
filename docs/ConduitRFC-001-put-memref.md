# RFC 001 — Promote `conduit.put_memref` / `conduit.get_memref`

**Status:** Draft
**Feature:** `conduit.put_memref`, `conduit.get_memref`
**Target backend:** mlir-air (`air.channel.put` / `air.channel.get`)
**Evidence:** `artifacts/roundtrip_expanded_results.json`

---

## Summary

Promote `conduit.put_memref` and `conduit.get_memref` from placeholder ops to
verified first-class operations carrying full N-D DMA descriptor payloads
(offsets, sizes, strides) compatible with both `air.channel.put` and
`aie.objectfifo` `dimensionsToStream` / `dimensionsFromStream`.

---

## Motivation

The dominant production form of AIR channel transfers is not a scalar token
exchange but an N-D strided DMA tile transfer:

```mlir
air.channel.put async [%dep] @channel_0[%i, %j]
    (%buf[%o0, %o1] [%c32, %c32] [%c64, %c1]) {id = 1 : i32}
    : (memref<64x64xi32>)
```

The corpus analysis found **1,314 `air.channel.put` occurrences** and
**1,152 `air.channel.get` occurrences** using this descriptor form across
mlir-air. The current scalar `PUT`/`GET` ops cannot represent any of them.

Similarly, `aie.objectfifo.create` uses `dimensionsToStream` / `dimensionsFromStream`
to express identical strided tiling:

```mlir
aie.objectfifo @of(%tile dimensionsToStream [<size = 32, stride = 64>], ...)
```

`conduit.put_memref` unifies both representations under a single descriptor model.

---

## Proposed Conduit Syntax

```
// Blocking memref put (AIR sync form):
conduit.put_memref %chan, %buf {
    offsets=[%row_off, 0],
    sizes=[32, 32],
    strides=[64, 1],
    num_elems=1024,
    chan_indices=[%i, %j]
}

// Async form returning a token:
%tok = conduit.put_memref_async %chan, %buf {
    offsets=[%row_off, 0],
    sizes=[32, 32],
    strides=[64, 1],
    num_elems=1024,
    chan_indices=[%i, %j],
    deps=[%dep0]
}

// Matching get side:
conduit.get_memref %chan, %dst {
    offsets=[], sizes=[], strides=[],
    num_elems=1024,
    chan_indices=[%i, %j]
}
```

**Descriptor payload shape:**

| Field | Type | Semantics |
|-------|------|-----------|
| `offsets` | list of index | Starting indices into source memref per dimension |
| `sizes` | list of index | Element count per dimension (may be symbolic SSA values) |
| `strides` | list of index | Stride in elements between successive transfers per dimension |
| `num_elems` | int or `?` | Flat element count; `?` when sizes are symbolic |
| `chan_indices` | list of index | Channel port indices (e.g. `[%i, %j]` for a `[2,2]` channel) |
| `deps` | optional list of tokens | Async dependency tokens (RFC 002) |
| `memref_type` | optional string | Source/dest memref type, e.g. `memref<64x64xi32>` |

Symbolic values (SSA names, `%cN_new` variants) are preserved verbatim; the
lowering does not attempt numeric evaluation.

---

## Sample Lowering

### conduit.put_memref → air.channel.put

```
conduit.put_memref_async %ch, %A {
    offsets=[%r, 0], sizes=[%c32, %c32], strides=[%c64, %c1],
    num_elems=?, chan_indices=[%i, %j], deps=[%d0]
}
```
→
```mlir
%tok = air.channel.put async [%d0] @ch[%i, %j]
    (%A[%r, %c0] [%c32, %c32] [%c64, %c1]) {id = 1 : i32}
    : (memref<128x64xi32>)
```

### conduit.put_memref → aie.objectfifo (dimensionsToStream)

```
conduit.put_memref %of_chan, %buf {
    offsets=[0, 0], sizes=[32, 32], strides=[64, 1], num_elems=1024
}
```
→
```mlir
aie.objectfifo @of_chan(%src_tile dimensionsToStream
    [<size = 32, stride = 64>, <size = 32, stride = 1>],
    {%dst_tile}, 4 : i32) : !aie.objectfifo<memref<1024xi32>>
```

### Empty descriptor (full-buffer transfer)

```
conduit.get_memref %chan, %dst { offsets=[], sizes=[], strides=[], num_elems=1024 }
```
→
```mlir
air.channel.get @chan[%i, %j] (%dst[] [] []) : (memref<1024xi32, 2>)
```

---

## Evidence

| Artifact | Path |
|----------|------|
| Roundtrip results (AIR memref category) | `artifacts/roundtrip_expanded_results.json` |
| Warning triage (symbolic_size_expression) | `artifacts/triage_warnings.json` |
| BD delta | `artifacts/bd_count_delta.json` |
| Evidence bundle | `artifacts/evidence_bundle.tar` |

Top candidate files exercising this feature (from `artifacts/candidate_pipelines.json`):

1. `fuse_channels.mlir` — 360 `air.channel.put/get` occurrences, 213 warnings (primarily CHECK-line; filtered by parser patch)
2. `isolate_async_dma_loop_nest.mlir` — 167 occurrences, 77 warnings
3. `specialize-channel-wrap-and-stride.mlir` — 76 occurrences, 16 warnings

---

## Tests to Add

Under `test/roundtrip/`:
- `put_memref_basic.mlir` — verify offsets/sizes/strides round-trip for constant values
- `put_memref_symbolic.mlir` — verify `?` num_elems preserved when sizes are SSA
- `put_memref_async.mlir` — verify async form with dep tokens

---

## Acceptance Gates

| Gate | Requirement | Current |
|------|-------------|---------|
| Roundtrip pass rate (AIR category) | ≥ 90% | 99% (50/50 AIR files) |
| Symbolic value preservation | No spurious warnings | Verified (patch applied) |
| offsets/sizes/strides in roundtrip | Present verbatim | Verified by integrity check |
| CI smoke test | Pass | Pass (20/20) |

> **Evidence caveat:** All "pass" results above reflect structural text-transform
> roundtrip correctness (the Python script correctly identifies and reformats
> descriptor fields).  They do **not** mean the generated Conduit IR is valid
> MLIR or that any lowering is semantically correct.  No code has been compiled
> or run on hardware.  These gates will need to be re-evaluated once the Conduit
> dialect is registered and `aie-opt`/`air-opt` can parse the output files.
