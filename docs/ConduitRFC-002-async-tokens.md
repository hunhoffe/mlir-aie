# RFC 002 — Promote Async Token Handles

**Status:** Draft
**Features:** `conduit.put_async`, `conduit.wait`, `conduit.wait_all`,
             `conduit.wait_all_async`, `conduit.chain`
**Target backends:** mlir-air (`air.async.token`), mlir-aie (lock acquire/release)
**Evidence:** `artifacts/roundtrip_expanded_results.json`

---

## Summary

Promote the async token handle family to first-class Conduit IR operations,
enabling non-blocking DMA enqueues with explicit dependency edges. This unifies
`air.async.token` chaining (AIR backend) and lock-sequence ordering (AIE backend)
under a single Conduit abstraction.

---

## Motivation

Every production AIR program uses the async form of channel.put/get:

```mlir
%tok = air.channel.put async [%dep] @chan[%i, %j] (...) : (memref<...>)
air.wait_all [%tok]
```

The async token list `[%dep, %tok0, %tok1]` encodes the DMA dependency graph —
which BDs may run concurrently, which must sequence. Without a Conduit async
model, the roundtrip scripts must degrade all transfers to synchronous form,
losing all pipeline overlap information.

On the AIE backend, the equivalent is lock-based BD ordering:
```mlir
aie.use_lock(%lock, Release, 1)   // producer BD completes
aie.use_lock(%lock, Acquire, 1)   // consumer BD waits
```

The Conduit async ops bridge both representations.

**Corpus evidence:** the triage analysis found 1 explicit `async_channel_parsing`
warning category and 194 `filecheck_wildcard` warnings — the latter almost
entirely from AIR async test files where `{{.*}}` wildcards appear in CHECK
lines for async token SSA names.

---

## Proposed Conduit Syntax

```
// Non-blocking scalar put, returns completion token:
%tok0 = conduit.put_async %chan, %val

// Non-blocking get:
%tok1 = conduit.get_async %chan

// Non-blocking memref put (integrates with RFC 001):
%tok2 = conduit.put_memref_async %chan, %buf {
    offsets=[%o0, %o1], sizes=[%s0, %s1], strides=[%r0, %r1], num_elems=?
}

// Blocking wait on a single token:
conduit.wait %tok0

// Blocking wait on multiple tokens:
conduit.wait_all [%tok0, %tok1, %tok2]

// Non-blocking merge — returns new token signaled when all inputs complete:
%tok_all = conduit.wait_all_async [%tok0, %tok1]

// Explicit dependency edge (tok_b cannot start until tok_a completes):
conduit.chain %tok_a -> %tok_b

// Multi-source chain:
conduit.chain [%tok0, %tok1] -> %tok2
```

**Token type:** `!conduit.async.token`

---

## Mapping to AIR async events

```
%tok2 = conduit.put_memref_async %ch, %A {
    offsets=[%r, 0], sizes=[%c32, %c32], strides=[%c64, %c1],
    num_elems=?, deps=[%tok0, %tok1]
}
conduit.wait %tok2
```
→
```mlir
%tok2 = air.channel.put async [%tok0, %tok1] @ch[%i, %j]
    (%A[%r, %c0] [%c32, %c32] [%c64, %c1]) {id = 3 : i32}
    : (memref<128x64xi32>)
air.wait_all [%tok2]
```

`conduit.chain %ta -> %tb` maps to: `%tb` includes `%ta` in its async dep list.
`conduit.wait_all_async [%t0, %t1]` maps to `air.wait_all async [%t0, %t1]`.

---

## Mapping to AIE lock sequences

```
conduit.chain %tok_put -> %tok_get
```
→
```mlir
// End of put BD:
aie.use_lock(%lock0, Release, 1)
// Start of get BD:
aie.use_lock(%lock0, Acquire, 1)
```

For `conduit.wait_all [%t0, %t1]` on the AIE backend: emit `aie.use_lock` Acquire
for each dependency lock before the waiting operation's BD begins.

---

## Unit Tests to Add

Under `test/roundtrip/`:

| Test file | What it verifies |
|-----------|-----------------|
| `async_put_basic.mlir` | Single `conduit.put_async` → `air.channel.put async []` roundtrip |
| `async_chain.mlir` | `conduit.chain %t0 -> %t1` → async dep list `[%t0]` in AIR |
| `wait_all_async.mlir` | `conduit.wait_all_async` → `air.wait_all async` roundtrip |
| `async_aie_lock.mlir` | `conduit.chain` → `aie.use_lock Release/Acquire` pair |
| `async_symbolic_tokens.mlir` | SSA token names with `%c0_new` suffixes preserved |

---

## Evidence

| Artifact | Path |
|----------|------|
| Roundtrip results (async AIR files) | `artifacts/roundtrip_expanded_results.json` |
| Warning triage (`async_channel_parsing`) | `artifacts/triage_warnings.json` |
| FileCheck wildcard fix | `artifacts/parser_validation.json` |
| Evidence bundle | `artifacts/evidence_bundle.tar` |

Representative files passing with async content:
- `mlir-air/mlir/test/Transform/AIRDmaToChannel/dma_to_channel_async.mlir`
- `mlir-air/mlir/test/Transform/AIRDependency/air_channel.mlir`
- `mlir-air/test/airhost/48_air_mmult_2x2_channel/air.mlir`

---

## Acceptance Gates

| Gate | Requirement | Current |
|------|-------------|---------|
| Roundtrip pass rate (AIR async files) | ≥ 90% | 99% overall |
| async dep lists preserved in roundtrip | Present in output | Verified |
| FileCheck wildcard lines skipped | No false parse failures | Verified (parser patch) |
| AIE lock sequence emitted for chain | Correct acquire/release order | Specified in extensions.md |
| CI smoke test | Pass | Pass (20/20) |

> **Evidence caveat:** All "pass" results above reflect structural text-transform
> roundtrip correctness (the Python script correctly identifies and reformats
> async dependency lists).  They do **not** mean the generated Conduit IR is
> valid MLIR or that token semantics are correct end-to-end.  No code has been
> compiled or run on hardware.  The AIE lock sequence gate is specified but not
> yet verified by any test.  These gates will need to be re-evaluated once the
> Conduit dialect is registered and `aie-opt`/`air-opt` can parse the output.

---

## Fallback Behavior

If the target backend does not support async tokens (e.g., synchronous objectfifo
lowering path), all `conduit.put_async` / `conduit.get_async` ops degrade to
their synchronous equivalents with a lowering warning. `conduit.chain` ops are
dropped and a comment is emitted noting the lost dependency edge.
