//===- Conduit.td - Conduit dialect TableGen definition ----*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Conduit is a portable bridge dialect that unifies mlir-aie ObjectFIFO
// (buffer-window semantics) and mlir-air Channel (N-D DMA descriptor + async
// token semantics) under a single intermediate representation.
//
// ## Unified window + token model
//
// The central design insight:
//
//   A `!conduit.window.token` from `conduit.acquire_async` IS pending
//   permission to access a buffer window.  A `!conduit.dma.token` from
//   `conduit.put_memref_async` IS a pending DMA transfer.
//   `conduit.wait_all` over both lets hardware satisfy them in parallel.
//
// Blocking forms are syntactic sugar:
//
//   conduit.acquire {name, count}
//   ≡  %tok = conduit.acquire_async {name, count}
//      conduit.wait %tok
//
//   conduit.put_memref {...}
//   ≡  %tok = conduit.put_memref_async {...}
//      conduit.wait %tok
//
// ## Cross-tier unified pattern
//
//   // Shim DMA producer (Tier 3): non-blocking N-D DMA fill from DDR
//   %dma_tok = conduit.put_memref_async {name = "input", num_elems = 9 : i64,
//                  offsets = array<i64: 0, 0>, sizes = array<i64: 3, 3>,
//                  strides = array<i64: 16, 1>} : !conduit.dma.token
//
//   // Compute core consumer (Tier 2, bridge): non-blocking window request
//   %acq_tok = conduit.acquire_async {name = "output", count = 1 : i64}
//                  : !conduit.window.token
//
//   // Hardware satisfies DMA fill and lock grant in parallel; wait for both
//   conduit.wait_all %dma_tok, %acq_tok
//
//   // Both ready: access the filled input buffer and the granted output window
//   %in  = conduit.subview_access {name = "input",  index = 0 : i64} : memref<3x3xi16>
//   %out = conduit.subview_access {name = "output", index = 0 : i64} : memref<9xi32>
//   // ... stencil compute ...
//   conduit.release {name = "input",  count = 1 : i64}
//   conduit.release {name = "output", count = 1 : i64}
//
//   This is valid: put_memref_async (Tier 3) on the producer endpoint and
//   acquire_async (Tier 2) on the consumer endpoint lower to the same
//   hardware lock+BD protocol via conduit.wait_all.
//
// ## Two-tier design
//
// ### Tier 2 — Buffer-window ops
//
//   conduit.acquire / conduit.acquire_async / conduit.release /
//   conduit.release_async / conduit.subview_access
//
//   Model: a shared buffer with lock-guarded windows.  The producer and
//   consumer share physical memory; locks enforce ordering.  The consumer
//   acquires a window of N buffer slots, reads/writes them in-place, and
//   releases M slots (M <= N to express sliding windows).
//
//   Used by: programs originating from mlir-aie ObjectFIFO (Pass A output).
//
//   Lowering target: aie.use_lock + aie.buffer + aie.dma_bd chains (Pass C).
//
//   Bridge to Tier 3: conduit.acquire_async returns a !conduit.window.token
//   that can be combined in conduit.wait_all with tokens from
//   conduit.put_memref_async (Tier 3 DMA operations), allowing the hardware
//   to satisfy buffer-window grants and DMA transfers in parallel.
//
// ### Tier 3 — Memref-DMA ops
//
//   conduit.put_memref / conduit.get_memref / conduit.put_memref_async /
//   conduit.get_memref_async
//
//   Model: a DMA transfer of a strided N-D memref tile.  Data is copied from
//   one memory space to another autonomously by the DMA engine.  No core body
//   involvement during the transfer; the software submits the descriptor and
//   optionally waits on an async token.
//
//   Used by: programs originating from mlir-air Channel (Pass B output).
//
//   Lowering target: aie.dma_bd with BDDimLayout wrap/stride descriptors.
//
//   Cross-tier: conduit.put_memref (Tier 3) on the producer endpoint and
//   conduit.acquire (Tier 2) on the consumer endpoint is the canonical
//   ObjectFIFO shim-to-core pattern; both lower to the same hardware
//   lock+BD protocol.
//
// ## What was deliberately excluded
//
// The following ops are NOT in this dialect.  They belong only in the
// Python DSL interpreter (tools/conduit_interpreter.py) or have no valid
// hardware lowering path:
//
//   conduit.put / conduit.get / conduit.peek / conduit.advance /
//   conduit.prefill  — Tier 1 scalar DSL ops; software simulation only.
//   conduit.put_async / conduit.get_async  — scalar async variants; removed
//                                            with the scalar model.
//   conduit.merge / conduit.fork  — no hardware lowering, no plans.
//   conduit.chain  — semantically broken in SSA: SSA values are immutable;
//                    cannot add deps to an existing token.  Use wait_all_async.
//   conduit.status — no lowering target; removed for minimality.
//
//===----------------------------------------------------------------------===//

#ifndef CONDUIT_DIALECT
#define CONDUIT_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// ---------------------------------------------------------------------------
// Dialect definition
// ---------------------------------------------------------------------------

def Conduit_Dialect : Dialect {
  let name = "conduit";
  let cppNamespace = "::xilinx::conduit";
  let summary = "Conduit canonical dataflow IR";
  let description = [{
    Conduit is a portable bridge dialect that unifies mlir-aie ObjectFIFO
    and mlir-air Channel semantics under a single intermediate representation.

    The dialect is organized into two tiers plus infrastructure:

    - Tier 2 (buffer-window): conduit.acquire/acquire_async/release/
      release_async/subview_access — ObjectFIFO path, produced by Pass A
      (--objectfifo-to-conduit)
    - Tier 3 (memref-DMA): conduit.put_memref/get_memref and their async
      variants — AIR Channel path, produced by Pass B
      (--air-channel-to-conduit, not yet implemented)

    The unifying mechanism is split token types: a !conduit.window.token from
    conduit.acquire_async (pending window grant) can be combined in
    conduit.wait_all with a token from conduit.put_memref_async (pending DMA
    transfer), allowing the hardware to satisfy both in parallel.

    Tiers 2 and 3 MAY coexist on the same conduit name at DIFFERENT endpoints:
    conduit.put_memref on the shim DMA producer endpoint and conduit.acquire
    on the compute core consumer endpoint is valid and common.

    Hardware lowering targets:
      Tier 2: aie.use_lock + aie.buffer + aie.dma_bd chains (Pass C)
      Tier 3: aie.dma_bd with BDDimLayout wrap/stride descriptors (Pass C)
  }];
  // Enable generated type printer/parser for !conduit.dma.token / !conduit.window.token.
  let useDefaultTypePrinterParser = 1;
  // Enable generated attribute printer/parser for #conduit.port<...>.
  let useDefaultAttributePrinterParser = 1;
}

// Base class for all Conduit ops.
class Conduit_Op<string mnemonic, list<Trait> traits = []>
    : Op<Conduit_Dialect, mnemonic, traits>;

// ---------------------------------------------------------------------------
// Async token types — !conduit.dma.token and !conduit.window.token
//
// Two separate types (simpler and more idiomatic than a parameterized type):
//   !conduit.dma.token    — produced by put_memref_async / get_memref_async /
//                           wait_all_async.  Represents a pending DMA transfer.
//   !conduit.window.token — produced by acquire_async / release_async.
//                           Represents pending permission to access a buffer
//                           window (a lock grant in flight).
//
// Op assignment:
//   conduit.put_memref_async / get_memref_async → !conduit.dma.token
//   conduit.acquire_async / release_async        → !conduit.window.token
//   conduit.wait                                 → accepts !conduit.dma.token
//   conduit.wait_window                          → accepts !conduit.window.token
//   conduit.wait_all / wait_all_async            → variadic Conduit_AnyTokenType (mixed)
//   conduit.wait_all_async result                → !conduit.dma.token
//
// The cross-tier pattern (hardware satisfies DMA fill and lock grant in
// parallel) uses conduit.wait_all with a mixed list:
//   %d = conduit.put_memref_async ... : !conduit.dma.token
//   %w = conduit.acquire_async    ... : !conduit.window.token
//   conduit.wait_all %d, %w           // AnyType variadic — accepts both
// ---------------------------------------------------------------------------

def Conduit_DMATokenType : TypeDef<Conduit_Dialect, "DMAToken"> {
  let mnemonic = "dma.token";
  let summary = "Completion token for an asynchronous DMA transfer";
  let description = [{
    An opaque handle representing the in-flight completion of an asynchronous
    DMA transfer.  Produced by conduit.put_memref_async, conduit.get_memref_async,
    and conduit.wait_all_async.  Consumed by conduit.wait (blocking) or passed
    to conduit.wait_all / conduit.wait_all_async (fan-in).

    Maps to:
    - mlir-air: !air.async.token (DMA variant)
    - mlir-aie: DMA BD completion; no explicit token type in AIE
  }];
  let parameters = (ins);
  let assemblyFormat = "";
}

def Conduit_WindowTokenType : TypeDef<Conduit_Dialect, "WindowToken"> {
  let mnemonic = "window.token";
  let summary = "Completion token for an asynchronous buffer-window acquisition";
  let description = [{
    An opaque handle representing the in-flight completion of a buffer-window
    lock request.  Produced by conduit.acquire_async and conduit.release_async.
    Consumed by conduit.wait_window (which resolves the token into a
    !conduit.window<T> value).  May also be passed to conduit.wait_all or
    conduit.wait_all_async (mixed variadic fan-in with DMA tokens).

    Maps to:
    - mlir-aie: lock acquire/release pair; no explicit token type in AIE
    - No direct AIR equivalent; AIR uses channel.put/get async for the DMA path
  }];
  let parameters = (ins);
  let assemblyFormat = "";
}

// Type constraint accepting any conduit token type for mixed-token variadics.
// Used by wait_all / wait_all_async which accept a mix of DMA and window tokens.
def Conduit_AnyTokenType : AnyTypeOf<[Conduit_DMATokenType,
                                       Conduit_WindowTokenType],
                                      "conduit token type">;

// ---------------------------------------------------------------------------
// Port enum — #conduit.port<Produce> / #conduit.port<Consume>
//
// Type-safe replacement for the old StrAttr port = "Produce"/"Consume".
// Eliminates an entire class of port-typo bugs; gives compile-time
// exhaustiveness on switch.
// ---------------------------------------------------------------------------

def Conduit_PortProduce : I32EnumAttrCase<"Produce", 0, "Produce">;
def Conduit_PortConsume : I32EnumAttrCase<"Consume", 1, "Consume">;
def Conduit_Port : I32EnumAttr<"Port", "conduit port direction",
    [Conduit_PortProduce, Conduit_PortConsume]> {
  let cppNamespace = "::xilinx::conduit";
  // Suppress I32EnumAttr-generated PortAttr (IntegerAttr subclass) —
  // the EnumAttr<> below generates the real PortAttr as a proper dialect attr.
  let genSpecializedAttr = 0;
}
def Conduit_PortAttr : EnumAttr<Conduit_Dialect, Conduit_Port, "port"> {
  let assemblyFormat = "`<` $value `>`";
}

// ---------------------------------------------------------------------------
// Window type — !conduit.window<T>
// ---------------------------------------------------------------------------

def Conduit_WindowType : TypeDef<Conduit_Dialect, "Window"> {
  let mnemonic = "window";
  let summary = "A typed grant of exclusive access to a DMA-filled buffer window";
  let description = [{
    Represents a buffer window that a compute core has been granted exclusive
    access to after a conduit.acquire completes.  Carries the element memref
    type T so that conduit.subview_access can return a properly-typed memref.

    T must be a memref type.  In Pass C lowering, a !conduit.window<T> value
    maps to the allocated aie.buffer SSA value of type T.

    Example:
      %win = conduit.acquire {name = "x", count = 1 : i64, port = "Consume"}
                 : !conduit.window<memref<10xi32>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<10xi32>> -> memref<10xi32>
  }];
  let parameters = (ins "mlir::MemRefType":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

// ---------------------------------------------------------------------------
// Infrastructure ops (no tier constraint — used by all tiers)
// ---------------------------------------------------------------------------

def Conduit_Create : Conduit_Op<"create"> {
  let summary = "declare a named conduit channel with a given element capacity";
  let description = [{
    Declares a named FIFO channel with a given element capacity.
    Must appear before any Tier 2 or Tier 3 ops that reference the same name.

    Name-based linkage is used (string attribute, not SSA value) because
    channels are referenced across IsolatedFromAbove region boundaries.

    The meaning of 'capacity' is tier-specific:
    - Tier 2: total number of buffer element slots (depth * elemCount)
    - Tier 3: not directly enforced; channel capacity is implicitly the
              memref allocation size on the consumer tile

    Typed lowering attributes (emitted by Pass A, consumed by Pass C).
    These replace the old conduit.annotate string-keyed approach:
    - producer_tile        : I64 array [col, row] of the producer tile
    - consumer_tiles       : flat I64 array [col0, row0, col1, row1, ...] for
                             each non-shim consumer tile in order (row > 0)
    - shim_consumer_tiles  : flat I64 array [col0, row0, ...] for each shim
                             (row == 0) consumer tile in order.  Shim tiles
                             are DMA endpoints with no local memory; they
                             receive aie.shim_dma_allocation + aie.flow
                             rather than aie.buffer + aie.lock.
    - element_type    : TypeAttr — actual element memref type (e.g. memref<10xi32>)
                        Pass C uses this for aie.buffer allocation; never defaults to i32.
    - depth           : I64Attr — double-buffering depth
    - link_mode       : optional StringAttr "distribute", "join", or "forward"
    - routing_mode    : optional StringAttr "circuit" (default) or "packet".
                        When "packet", Pass C emits aie.packet_flow instead of
                        aie.flow for the circuit-switched routing of this conduit.
                        Packet-switched flows carry a packet ID and allow
                        hardware-level switching based on packet headers.
    - producer_rates  : optional DenseI64Array — CSDF producer token-count per
                        phase.  P = (p0, p1, ..., p_{q-1}), phase period q=len(P).
                        When present, consumer_rates must also be present and the
                        Lee-Messerschmitt CSDF balance equation must hold:
                          sum(P) * len(C) == sum(C) * len(P)
                        This is a NECESSARY condition for a periodic schedule
                        (it guarantees equal average production and consumption
                        rates).  The SUFFICIENT condition (deadlock freedom: buffer
                        never underflows or overflows during one hyper-period) is
                        also statically computable from these rates + the capacity
                        and depth attributes via hyper-period simulation — see TODO
                        M7 in the work queue.
    - consumer_rates  : optional DenseI64Array — CSDF consumer token-count per
                        phase.  C = (c0, c1, ..., c_{r-1}), phase period r=len(C).
                        When present, producer_rates must also be present.  The
                        verifier enforces the CSDF balance equation:
                          sum(producer_rates) * len(consumer_rates)
                            == sum(consumer_rates) * len(producer_rates)
                        Note: the weaker condition sum(P)==sum(C) only handles
                        equal-period actors (q==r) and is NOT sufficient in general.
                        Legacy note: access_pattern is a backward-compatible alias
                        for per-phase consumer window sizes and does not carry an
                        explicit producer rate.  New code should use producer_rates
                        + consumer_rates for full CSDF semantics.

    Example:
      conduit.create {name = "c1", capacity = 10 : i64,
                      producer_tile = array<i64: 0, 2>,
                      shim_consumer_tiles = array<i64: 0, 0>,
                      element_type = memref<10xi32>,
                      depth = 1 : i64}

    Cyclostatic (CSDF) example — consumer alternates between acquiring 1 then 2:
      conduit.create {name = "csdf_fifo", capacity = 4 : i64,
                      producer_tile = array<i64: 2, 2>,
                      consumer_tiles = array<i64: 2, 3>,
                      element_type = memref<i32>,
                      depth = 4 : i64,
                      access_pattern = array<i64: 1, 2, 1>}

    The access_pattern encodes the cyclostatic consume sequence: the consumer
    acquires access_pattern[0] elements on the first iteration, access_pattern[1]
    on the second, and so on, cycling back to index 0.  Pass C uses this to
    generate lock acquire/release sequences with the correct varying counts
    rather than the uniform count used in the SDF case.
    When access_pattern is absent the conduit is treated as uniform SDF with
    count = capacity / depth per iteration.

    Full CSDF example with explicit producer and consumer rates:
      conduit.create {name = "csdf_full", capacity = 6 : i64,
                      producer_tile = array<i64: 0, 2>,
                      consumer_tiles = array<i64: 0, 3>,
                      element_type = memref<i32>,
                      depth = 6 : i64,
                      producer_rates = array<i64: 1, 2>,
                      consumer_rates = array<i64: 1, 2>}
      // Balance check: sum(P)*len(C) = 3*2 = 6 == sum(C)*len(P) = 3*2 = 6 ✓
      // (same-period case q=r=2, average rate = 1.5 tokens/firing for both)

    Different-period balanced example (producer bursts 3 then 1; consumer uniform 2):
      conduit.create {name = "csdf_diff_period", capacity = 4 : i64, ...,
                      producer_rates = array<i64: 3, 1>,
                      consumer_rates = array<i64: 2>}
      // P=[3,1] sum=4 q=2; C=[2] sum=2 r=1
      // Balance: sum(P)*len(C) = 4*1 = 4 == sum(C)*len(P) = 2*2 = 4 ✓
      // Average rate: 4/2 = 2 tokens/phase for both actors.

    Sum-equal but CSDF-imbalanced (rejected):
      // P=[3] sum=3 q=1; C=[1,2] sum=3 r=2
      // Balance: sum(P)*len(C) = 3*2 = 6 != sum(C)*len(P) = 3*1 = 3 ✗
      // Producer avg rate = 3/1 = 3; consumer avg rate = 3/2 = 1.5 — mismatch.
  }];
  let arguments = (ins StrAttr:$name,
                       I64Attr:$capacity,
                       OptionalAttr<DenseI64ArrayAttr>:$producer_tile,
                       OptionalAttr<DenseI64ArrayAttr>:$consumer_tiles,
                       OptionalAttr<DenseI64ArrayAttr>:$shim_consumer_tiles,
                       OptionalAttr<TypeAttr>:$element_type,
                       OptionalAttr<I64Attr>:$depth,
                       OptionalAttr<StrAttr>:$link_mode,
                       OptionalAttr<DenseI64ArrayAttr>:$access_pattern,
                       OptionalAttr<StrAttr>:$routing_mode,
                       OptionalAttr<DenseI64ArrayAttr>:$producer_rates,
                       OptionalAttr<DenseI64ArrayAttr>:$consumer_rates,
                       // alloc_tile: optional MemTile coordinate [col, row] for
                       // hardware relay allocation.  No verifier checks tile validity.
                       // Pass A guarantees a valid MemTile coordinate; Pass C will
                       // fail at lookupTileByCoord if wrong.
                       OptionalAttr<DenseI64ArrayAttr>:$alloc_tile,
                       // repeat_count: optional DMA repeat count.  When present,
                       // Pass C must set DMAStartOp repeat_count to this value
                       // instead of hardcoded 0.  Each BD fires repeat_count times
                       // before advancing to the next BD in the chain.
                       // Pass A propagates this from aie.objectfifo.create's
                       // repeat_count attribute.  P1-F rejects repeat_count > 1
                       // until full Pass C implementation is done.
                       OptionalAttr<I64Attr>:$repeat_count);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Conduit_RegisterExternalBuffers : Conduit_Op<"register_external_buffers"> {
  let summary = "register host-side external buffers with a shim-DMA conduit";
  let description = [{
    Registers external memory buffers (typically host-side DDR allocations) with
    a shim DMA conduit endpoint.  The shim DMA will use these buffers instead of
    on-chip tile memory for the data transfer.

    This covers the aie.objectfifo.register_external_buffers pattern where
    external buffers (from aie.external_buffer) are bound to a shim tile's DMA
    for host ↔ device data movement.

    Attributes:
    - name       : conduit channel name (must match a conduit.create)
    - tile_coord : [col, row] of the shim tile these buffers are bound to

    Operands:
    - external_buffers : one or more memref SSA values from aie.external_buffer

    Example:
      %buf0 = aie.external_buffer : memref<512xi16>
      %buf1 = aie.external_buffer : memref<512xi16>
      conduit.register_external_buffers(%buf0, %buf1)
          {name = "shim_chan", tile_coord = array<i64: 0, 0>}
          : (memref<512xi16>, memref<512xi16>)
  }];
  let arguments = (ins StrAttr:$name,
                       DenseI64ArrayAttr:$tile_coord,
                       Variadic<AnyMemRef>:$external_buffers);
  let assemblyFormat = "`(` $external_buffers `)` attr-dict `:` `(` type($external_buffers) `)`";
}

// ---------------------------------------------------------------------------
// Tier 2 — Buffer-window ops
// (ObjectFIFO path; produced by Pass A; lowered by Pass C)
// Tier 2 ops MAY coexist with Tier 3 ops at different endpoints of the same
// conduit name.
// ---------------------------------------------------------------------------

def Conduit_Acquire : Conduit_Op<"acquire"> {
  let summary = "[Tier 2] blocking window acquire — reserve N buffer slots for in-place access";
  let description = [{
    Tier 2 buffer-window op.  Reserves N contiguous buffer slots from the
    conduit for in-place read/write access.  Blocks until N slots are
    available from the producer.

    The 'port' attribute is required and indicates which lock to acquire:
    - "Produce": acquires the producer lock (used by the producer core)
    - "Consume": acquires the consumer lock (used by the consumer core)

    Pass C branches on this attribute to select prodLock vs consLock,
    preventing the deadlock that occurs when a producer core acquires the
    consumer lock or vice versa.

    Returns a !conduit.window<T> SSA value that carries the element memref
    type T.  The window value must be passed to conduit.subview_access to
    obtain the addressable memref.  Pass C maps this window value to the
    allocated aie.buffer SSA value.

    Semantics note: the 'count' is a *ceiling* on the total currently held
    window, not an increment.  If the caller already holds K slots (from a
    prior acquire without a matching full release), only max(0, N - K) new
    slots are acquired from the producer.  This enables sliding-window
    patterns:

      %w = conduit.acquire {name = "x", count = 3 : i64, port = "Consume"}
               : !conduit.window<memref<64xi32>>
      // holds slots [0,1,2]
      conduit.release {name = "x", count = 1 : i64, port = "Consume"}
      %w2 = conduit.acquire {name = "x", count = 3 : i64, port = "Consume"}
                : !conduit.window<memref<64xi32>>
      // fetches 1 new slot; window is now [1,2,3]

    Equivalent to: conduit.acquire_async + conduit.wait (the async form
    is preferred when overlapping with DMA or other async operations).

    Maps to: aie.objectfifo.acquire (Pass A source); aie.use_lock (Pass C output).

    Cross-tier: a conduit where the producer endpoint uses conduit.put_memref
    (Tier 3, shim DMA fill) and the consumer endpoint uses conduit.acquire
    (Tier 2, compute core window) is valid and common.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$count, Conduit_PortAttr:$port);
  let results   = (outs Conduit_WindowType:$window);
  let assemblyFormat = "attr-dict `:` type($window)";
  let hasVerifier = 1;
}

def Conduit_AcquireAsync : Conduit_Op<"acquire_async"> {
  let summary = "[Tier 2+3 bridge] non-blocking acquire — returns a !conduit.window.token granting access to a buffer window";
  let description = [{
    The bridge op between the buffer-window model (Tier 2) and the async token
    model (Tier 3).

    Submits a request to acquire N buffer slots without blocking.  Returns a
    !conduit.window.token that is complete when the N slots are available and
    the buffer window is ready to access.  After conduit.wait_window on this
    token, conduit.subview_access is valid for this conduit.

    NOTE on 'count': the count is the maximum window size (total slots held
    simultaneously), NOT a per-phase CSDF firing rate.  For CSDF rate
    information see the producer_rates / consumer_rates attrs on conduit.create.

    NOTE on port: acquire_async is always consumer-side (Consume port implicit).
    For producer-side async acquisition, use conduit.acquire with port='Produce'.

    This op is the key to unifying ObjectFIFO and AIR Channel semantics:
    - A !conduit.window.token from acquire_async IS pending permission to
      access a buffer window.
    - A !conduit.dma.token from put_memref_async IS a pending DMA transfer.
    - conduit.wait_all over both tokens (AnyType variadic) lets the hardware
      satisfy them in parallel.

    Example — overlap DMA fill with buffer window acquisition:

      // Non-blocking: DMA fills input from DDR using N-D descriptor (Tier 3)
      %dma_tok = conduit.put_memref_async {name = "input",
                     num_elems = 9 : i64,
                     offsets = array<i64: 0, 0>, sizes = array<i64: 3, 3>,
                     strides = array<i64: 16, 1>} : !conduit.dma.token

      // Non-blocking: request output window (Tier 2 bridge op)
      %acq_tok = conduit.acquire_async {name = "output", count = 1 : i64}
                     : !conduit.window.token

      // Hardware satisfies both in parallel; software waits for both
      conduit.wait_all %dma_tok, %acq_tok

      // Materialize window handle after the lock grant is satisfied
      %win = conduit.wait_window %acq_tok for "output"
                 : !conduit.window.token -> !conduit.window<memref<9xi32>>

    Hardware lowering (Pass C):
      aie.use_lock(consLock, AcquireGreaterEqual, count)  // non-blocking poll
      The token is complete when the lock is granted.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$count);
  let results   = (outs Conduit_WindowTokenType:$token);
  let assemblyFormat = "attr-dict `:` type($token)";
  let hasVerifier = 1;
}

def Conduit_Release : Conduit_Op<"release"> {
  let summary = "[Tier 2] blocking window release — return M buffer slots to the producer";
  let description = [{
    Tier 2 buffer-window op.  Releases M slots back to the producer, given
    the !conduit.window<T> SSA value that was acquired.  M may be less than
    the originally acquired window count (partial release), which keeps the
    remaining slots in the consumer's view for the next acquire call (sliding
    window pattern).

    Taking the window as an SSA operand establishes the def-use chain:
    the verifier can confirm you only release a window you actually acquired
    (the !conduit.window<T> must be defined by a preceding conduit.acquire or
    conduit.wait_window on the same channel).

    The 'count' attribute specifies how many slots to return (partial release
    when count < acquired count).

    The 'port' attribute indicates which lock to release:
    - "Produce": releases the producer lock (producer core use)
    - "Consume": releases the consumer lock (consumer core use)

    Pass C reads the window operand's defining op to recover the conduit name,
    then branches on port to select prodLock vs consLock.

    Sliding window example:
      %win = conduit.acquire {name = "x", count = 3 : i64, port = "Consume"}
                 : !conduit.window<memref<64xi32>>
      // ... use slots 0,1,2 ...
      conduit.release %win {count = 1 : i64, port = "Consume"}
          : !conduit.window<memref<64xi32>>   // slide by 1, keep 2 slots
      %win2 = conduit.acquire {name = "x", count = 3 : i64, port = "Consume"}
                  : !conduit.window<memref<64xi32>>
      conduit.release %win2 {count = 3 : i64, port = "Consume"}
          : !conduit.window<memref<64xi32>>

    Maps to: aie.objectfifo.release (Pass A source); aie.use_lock (Pass C output).
  }];
  let arguments = (ins Conduit_WindowType:$window, I64Attr:$count, Conduit_PortAttr:$port);
  let assemblyFormat = "$window attr-dict `:` type($window)";
}

def Conduit_ReleaseAsync : Conduit_Op<"release_async"> {
  let summary = "[Tier 2+3] non-blocking release — submit return of M buffer slots, return a !conduit.window.token";
  let description = [{
    Non-blocking variant of conduit.release.  Submits the release of M buffer
    slots and returns a !conduit.window.token representing the pending release.
    Downstream ops that depend on the producer being unblocked can chain on
    this token.

    Returns !conduit.window.token (not !conduit.dma.token) because the release
    is a lock operation (Tier 2 window model), not a DMA transfer.

    Hardware lowering (Pass C):
      port="Consume": aie.use_lock(prodLock, Release, count) -- free up producer slots
      port="Produce": aie.use_lock(consLock, Release, count) -- signal data ready
      The token signals when the release is complete.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$count, Conduit_PortAttr:$port);
  let results   = (outs Conduit_WindowTokenType:$token);
  let assemblyFormat = "attr-dict `:` type($token)";
  let hasVerifier = 1;
}

def Conduit_SubviewAccess : Conduit_Op<"subview_access", [Pure]> {
  let summary = "[Tier 2] access element N of a previously acquired buffer window";
  let description = [{
    Tier 2 buffer-window op.  Takes a !conduit.window<T> SSA value from a
    preceding conduit.acquire and returns a memref<T> reference to element
    'index' within the acquired window.

    VALIDITY PRECONDITION: %window must be the result of a conduit.acquire
    that has completed.  The window SSA value establishes the SSA data
    dependency: no memref.alloc placeholder is needed.

    Pass C lowering: the !conduit.window<T> operand maps to the allocated
    aie.buffer SSA value; the result memref<T> is the same buffer value
    passed through.

    The result type is memref<T> where T is extracted from the window type.
    The caller does not specify the result type explicitly — it is inferred
    from the window operand type.

    This op covers the 8,392-use corpus gap of aie.objectfifo.subview.access
    across 1,165 mlir-aie .mlir files.

    Maps to: aie.objectfifo.subview.access (Pass A source).

    Example:
      %win = conduit.acquire {name = "input", count = 1 : i64,
                              port = "Consume"}
                 : !conduit.window<memref<10xi32>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<10xi32>> -> memref<10xi32>
      // ... use %buf ...
      conduit.release {name = "input", count = 1 : i64, port = "Consume"}
  }];
  let arguments = (ins Conduit_WindowType:$window, I64Attr:$index);
  let results   = (outs AnyMemRef:$result);
  let assemblyFormat = "$window attr-dict `:` type($window) `->` type($result)";
  let hasVerifier = 1;
}

// ---------------------------------------------------------------------------
// Hardware structure ops
// ---------------------------------------------------------------------------

def Conduit_ObjectFifoLink : Conduit_Op<"objectfifo_link"> {
  let summary = "link conduits via a MemTile relay for hardware-mediated distribute or join";
  let description = [{
    Represents aie.objectfifo.link semantics: hardware-mediated data relay
    through a MemTile with optional byte-offset splits.

    Three modes:
    - "distribute": 1 source → N destinations, each starting at a byte offset
    - "join":       N sources (each at a byte offset) → 1 destination
    - "forward":    exactly 1 source → exactly 1 destination; direct pass-through
                    relay with no splitting or joining (invariant: srcs.size()==1,
                    dsts.size()==1)

    Invariants (enforced by verifier):
    - distribute mode: when offsets is non-empty, offsets.size() == dsts.size()
    - join mode: when offsets is non-empty, offsets.size() == srcs.size()
    - forward mode: srcs.size() == 1 and dsts.size() == 1

    The 'memtile' attribute is a coordinate hint ("tile(col,row)") indicating
    which MemTile tile performs the relay.  Pass C uses this to locate the
    correct tile SSA value in the device body.

    The 'lock_id', when present, passes through to generated AIE ops to
    preserve synchronization semantics.

    Example:
      conduit.objectfifo_link {srcs = ["of_in"],
                               dsts = ["of_out0", "of_out1"],
                               mode = "distribute",
                               memtile = "tile(0,1)",
                               offsets = array<i64: 0, 1024>}
  }];
  let arguments = (ins StrArrayAttr:$srcs,
                       StrArrayAttr:$dsts,
                       StrAttr:$mode,
                       StrAttr:$memtile,
                       OptionalAttr<DenseI64ArrayAttr>:$offsets,
                       OptionalAttr<I64Attr>:$lock_id);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

// ---------------------------------------------------------------------------
// Tier 3 — Memref-DMA ops (blocking)
// (AIR Channel path; produced by Pass B; lowered by Pass C)
// Tier 3 ops MAY coexist with Tier 2 ops at different endpoints of the same
// conduit name.
// ---------------------------------------------------------------------------

def Conduit_PutMemref : Conduit_Op<"put_memref"> {
  let summary = "[Tier 3] blocking N-D DMA send — transfer a strided memref tile into a conduit";
  let description = [{
    Tier 3 memref-DMA op.  Submits a DMA transfer of a strided N-D memref
    tile into the named conduit.  Blocks until the transfer completes.

    The [offsets][sizes][strides] triple follows the same convention as
    air.channel.put: offsets is the starting element indices in each
    dimension, sizes is the number of elements in each dimension, and
    strides is the distance between consecutive elements in each dimension
    (in units of the element type).

    Maps to: air.channel.put (Pass B source); aie.dma_bd with BDDimLayout
    wrap/stride descriptors (Pass C output).

    Cross-tier: conduit.put_memref on the producer endpoint (shim DMA fill)
    paired with conduit.acquire on the consumer endpoint (compute core window)
    is the canonical ObjectFIFO shim-to-core pattern.

    Example:
      conduit.put_memref {name = "input", num_elems = 64 : i64,
                          offsets = array<i64: 0, 0>,
                          sizes   = array<i64: 8, 8>,
                          strides = array<i64: 16, 1>}
  }];
  let arguments = (ins StrAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides);
  let assemblyFormat = "attr-dict";
}

def Conduit_GetMemref : Conduit_Op<"get_memref"> {
  let summary = "[Tier 3] blocking N-D DMA receive — receive a strided memref tile from a conduit";
  let description = [{
    Tier 3 memref-DMA op.  Receives a DMA transfer of a strided N-D memref
    tile from the named conduit into local memory.  Blocks until data is
    available.

    Maps to: air.channel.get (Pass B source); aie.dma_bd (Pass C output).
  }];
  let arguments = (ins StrAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides);
  let assemblyFormat = "attr-dict";
}

// ---------------------------------------------------------------------------
// Tier 3 — Memref-DMA ops (async)
// ---------------------------------------------------------------------------

def Conduit_PutMemrefAsync : Conduit_Op<"put_memref_async"> {
  let summary = "[Tier 3] non-blocking N-D DMA send — submit a strided memref DMA transfer, return a !conduit.dma.token";
  let description = [{
    Non-blocking variant of conduit.put_memref.  Submits the DMA BD descriptor
    and returns immediately with a !conduit.dma.token.

    The returned DMA token may be passed to conduit.wait (blocking), to
    conduit.wait_all (fan-in alongside window tokens from acquire_async), or
    to conduit.wait_all_async (merge into a new DMA token for further
    pipelining).

    Cross-tier pattern: mix this DMA token with a !conduit.window.token from
    conduit.acquire_async in conduit.wait_all (AnyType variadic), so the
    hardware satisfies the DMA fill and the lock grant in parallel.

    Maps to: air.channel.put async [deps] (Pass B source).
  }];
  let arguments = (ins StrAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides,
                       Variadic<Conduit_AnyTokenType>:$deps);
  let results   = (outs Conduit_DMATokenType:$token);
  let assemblyFormat = "(`[` $deps^ `:` type($deps) `]`)? attr-dict `:` type($token)";
  let hasVerifier = 1;
}

def Conduit_GetMemrefAsync : Conduit_Op<"get_memref_async"> {
  let summary = "[Tier 3] non-blocking N-D DMA receive — submit a DMA receive descriptor, return a !conduit.dma.token";
  let description = [{
    Non-blocking variant of conduit.get_memref.  Returns a !conduit.dma.token
    that the caller must wait on before accessing the received data.

    Maps to: air.channel.get async [deps] (Pass B source).
  }];
  let arguments = (ins StrAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides,
                       Variadic<Conduit_AnyTokenType>:$deps);
  let results   = (outs Conduit_DMATokenType:$token);
  let assemblyFormat = "(`[` $deps^ `:` type($deps) `]`)? attr-dict `:` type($token)";
  let hasVerifier = 1;
}

// ---------------------------------------------------------------------------
// Token synchronization ops
// (combine Tier 2 and Tier 3 tokens in conduit.wait_all for cross-tier use)
// ---------------------------------------------------------------------------

def Conduit_Wait : Conduit_Op<"wait"> {
  let summary = "blocking wait on a single !conduit.dma.token";
  let description = [{
    Blocks the current thread until the DMA transfer represented by %token
    is complete.  Accepts only !conduit.dma.token (DMA transfers from
    put_memref_async / get_memref_async / wait_all_async).

    For !conduit.window.token from acquire_async, use conduit.wait_window
    instead — it resolves the lock grant into a !conduit.window<T> SSA value
    needed by conduit.subview_access.

    Maps to: air.wait_all [%tok] (blocking, no result) on the AIR backend.
  }];
  let arguments = (ins Conduit_DMATokenType:$token);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def Conduit_WaitWindow : Conduit_Op<"wait_window"> {
  let summary = "[Tier 2+3 bridge] blocking wait on a !conduit.window.token, returning the granted !conduit.window<T>";
  let description = [{
    Blocks the current thread until the !conduit.window.token (from
    acquire_async) is complete — i.e., until the hardware lock grants the
    buffer window — then returns the !conduit.window<T> SSA value.

    Accepts only !conduit.window.token (NOT !conduit.dma.token).  For DMA
    tokens use conduit.wait instead.

    Full async acquire pattern:
      %tok = conduit.acquire_async {name = "in", count = 1 : i64}
                 : !conduit.window.token
      %dma = conduit.put_memref_async {name = "in", ...} : !conduit.dma.token
      conduit.wait_all %tok, %dma   // AnyType variadic — accepts both
      %win = conduit.wait_window %tok for "in"
                 : !conduit.window.token -> !conduit.window<memref<10xi32>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<10xi32>> -> memref<10xi32>
      conduit.release %win {count = 1 : i64, port = "Consume"}
                 : !conduit.window<memref<10xi32>>

    Hardware lowering (Pass C):
      The underlying lock acquire was performed by the acquire_async hardware
      path.  conduit.wait_window emits aie.use_lock(consLock, AcquireGreaterEqual)
      and maps the resolved token to the allocated aie.buffer SSA value.
  }];
  let arguments = (ins Conduit_WindowTokenType:$token, StrAttr:$name);
  let results   = (outs Conduit_WindowType:$window);
  let assemblyFormat = "$token `for` $name attr-dict `:` type($token) `->` type($window)";
  let hasVerifier = 1;
}

def Conduit_WaitAll : Conduit_Op<"wait_all"> {
  let summary = "blocking fan-in wait — block until all tokens in the list are complete";
  let description = [{
    Blocks the current thread until all tokens in the list are complete.  All
    specified async operations (DMA transfers, lock grants, releases) must
    have finished before execution continues.

    Accepts zero or more conduit token types so that a mix of
    !conduit.dma.token (from put/get_memref_async) and !conduit.window.token
    (from acquire_async / release_async) can be waited on together in a
    single fan-in.  This is the canonical cross-tier pattern:

      %d = conduit.put_memref_async ... : !conduit.dma.token
      %w = conduit.acquire_async    ... : !conduit.window.token
      conduit.wait_all %d, %w   // hardware satisfies both in parallel

    Maps to: air.wait_all [%tok0, %tok1, ...] (blocking, no result).
  }];
  let arguments = (ins Variadic<Conduit_AnyTokenType>:$tokens);
  let assemblyFormat = "$tokens attr-dict `:` type($tokens)";
  let hasVerifier = 1;
}

def Conduit_WaitAllAsync : Conduit_Op<"wait_all_async"> {
  let summary = "non-blocking fan-in — merge mixed tokens into a new !conduit.dma.token when all inputs complete";
  let description = [{
    Returns a new !conduit.dma.token that is complete when all input tokens
    are complete.  Non-blocking: the returned token represents the pending
    completion of the fan-in.

    Accepts variadic conduit token type inputs (mix of !conduit.dma.token and
    !conduit.window.token) and returns a !conduit.dma.token representing the
    merged completion.

    Example:
      %t0 = conduit.put_memref_async {name = "a", num_elems = 64 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 64>,
                 strides = array<i64: 1>} : !conduit.dma.token
      %t1 = conduit.acquire_async {name = "b", count = 1 : i64}
                 : !conduit.window.token
      %merged = conduit.wait_all_async %t0, %t1 :
          (!conduit.dma.token, !conduit.window.token) -> !conduit.dma.token

    Maps to: air.wait_all async [...] (Pass B source).
  }];
  let arguments = (ins Variadic<Conduit_AnyTokenType>:$tokens);
  let results   = (outs Conduit_DMATokenType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

#endif // CONDUIT_DIALECT
