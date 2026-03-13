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
//   A `!conduit.async.token` from `conduit.acquire_async` IS pending
//   permission to access a buffer window.  A `!conduit.async.token` from
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
//                  strides = array<i64: 16, 1>} : !conduit.async.token
//
//   // Compute core consumer (Tier 2, bridge): non-blocking window request
//   %acq_tok = conduit.acquire_async {name = "output", count = 1 : i64}
//                  : !conduit.async.token
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
//   Bridge to Tier 3: conduit.acquire_async returns a !conduit.async.token
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

    The unifying mechanism is the !conduit.async.token type: a token from
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
  // Enable generated type printer/parser for !conduit.async.token.
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 0;
}

// Base class for all Conduit ops.
class Conduit_Op<string mnemonic, list<Trait> traits = []>
    : Op<Conduit_Dialect, mnemonic, traits>;

// ---------------------------------------------------------------------------
// Async token type — !conduit.async.token
// ---------------------------------------------------------------------------

def Conduit_AsyncTokenType : TypeDef<Conduit_Dialect, "AsyncToken"> {
  let mnemonic = "async.token";
  let summary = "Conduit async operation completion token";
  let description = [{
    An opaque handle representing the in-flight completion of an asynchronous
    Conduit operation.  Produced by: conduit.acquire_async, conduit.release_async,
    conduit.put_memref_async, conduit.get_memref_async, or conduit.wait_all_async.
    Consumed by: conduit.wait, conduit.wait_all, conduit.wait_all_async.

    The key semantic: a !conduit.async.token from conduit.acquire_async IS
    pending permission to access a buffer window; a !conduit.async.token from
    conduit.put_memref_async IS a pending DMA transfer.  conduit.wait_all over
    both tokens lets the hardware satisfy them in parallel.

    Maps to:
    - mlir-air  : !air.async.token
    - mlir-aie  : lock acquire/release pair (no explicit token type in AIE)
  }];
  let parameters = (ins);
  let assemblyFormat = "";
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
    - link_mode       : optional StringAttr "distribute" or "join"

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
  }];
  let arguments = (ins StrAttr:$name,
                       I64Attr:$capacity,
                       OptionalAttr<DenseI64ArrayAttr>:$producer_tile,
                       OptionalAttr<DenseI64ArrayAttr>:$consumer_tiles,
                       OptionalAttr<DenseI64ArrayAttr>:$shim_consumer_tiles,
                       OptionalAttr<TypeAttr>:$element_type,
                       OptionalAttr<I64Attr>:$depth,
                       OptionalAttr<StrAttr>:$link_mode,
                       OptionalAttr<DenseI64ArrayAttr>:$access_pattern);
  let assemblyFormat = "attr-dict";
}

def Conduit_RegisterExternalBuffers : Conduit_Op<"register_external_buffers"> {
  let summary = "register host-side external buffers with a shim-DMA conduit";
  let description = [{
    Covers the pattern of aie.objectfifo_register_external_buffers / shimDMA
    registration for host ↔ shim data transfer.  The 'base_addr = 0' form
    indicates runtime-determined addresses (XRT handles buffer mapping).

    Attributes:
    - name        : conduit channel name (typically a shim DMA channel)
    - num_buffers : number of external host buffers to register
    - base_addr   : base address of the first buffer (0 = runtime-determined)

    Example:
      conduit.register_external_buffers {name = "shim_chan",
                                         num_buffers = 2 : i64,
                                         base_addr = 0 : i64}
  }];
  let arguments = (ins StrAttr:$name,
                       I64Attr:$num_buffers,
                       I64Attr:$base_addr);
  let assemblyFormat = "attr-dict";
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
  let arguments = (ins StrAttr:$name, I64Attr:$count, StrAttr:$port);
  let results   = (outs Conduit_WindowType:$window);
  let assemblyFormat = "attr-dict `:` type($window)";
}

def Conduit_AcquireAsync : Conduit_Op<"acquire_async"> {
  let summary = "[Tier 2+3 bridge] non-blocking acquire — returns a token granting access to a buffer window";
  let description = [{
    The bridge op between the buffer-window model (Tier 2) and the async token
    model (Tier 3).

    Submits a request to acquire N buffer slots without blocking.  Returns a
    !conduit.async.token that is complete when the N slots are available and
    the buffer window is ready to access.  After conduit.wait or conduit.wait_all
    on this token, conduit.subview_access is valid for this conduit.

    This op is the key to unifying ObjectFIFO and AIR Channel semantics:
    - A token from conduit.acquire_async IS pending permission to access a
      buffer window.
    - A token from conduit.put_memref_async IS a pending DMA transfer.
    - conduit.wait_all over both tokens lets the hardware satisfy them in
      parallel.

    This allows the core to overlap buffer acquisition (lock wait) with DMA
    submission without software polling.

    Relationship to conduit.acquire (blocking):
      conduit.acquire {name, count}
      ≡  %tok = conduit.acquire_async {name, count}
         conduit.wait %tok

    Example — overlap DMA fill with buffer window acquisition:

      // Non-blocking: DMA fills input from DDR using N-D descriptor (Tier 3)
      %dma_tok = conduit.put_memref_async {name = "input",
                     num_elems = 9 : i64,
                     offsets = array<i64: 0, 0>, sizes = array<i64: 3, 3>,
                     strides = array<i64: 16, 1>} : !conduit.async.token

      // Non-blocking: request output window (Tier 2 bridge op)
      %acq_tok = conduit.acquire_async {name = "output", count = 1 : i64}
                     : !conduit.async.token

      // Hardware satisfies both in parallel; software waits for both
      conduit.wait_all %dma_tok, %acq_tok

      // Both conditions met: input buffer filled, output window granted
      %in  = conduit.subview_access {name = "input",  index = 0 : i64} : memref<3x3xi16>
      %out = conduit.subview_access {name = "output", index = 0 : i64} : memref<9xi32>
      // ... compute stencil ...
      conduit.release {name = "input",  count = 1 : i64}
      conduit.release {name = "output", count = 1 : i64}

    Hardware lowering (Pass C):
      aie.use_lock(consLock, AcquireGreaterEqual, count)  // non-blocking poll
      The token is complete when the lock is granted.  On AIE, both DMA BDs
      and lock operations run in hardware concurrently; software waits on
      whichever completes last.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$count);
  let results   = (outs Conduit_AsyncTokenType:$token);
  let assemblyFormat = "attr-dict `:` type($token)";
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
  let arguments = (ins Conduit_WindowType:$window, I64Attr:$count, StrAttr:$port);
  let assemblyFormat = "$window attr-dict `:` type($window)";
}

def Conduit_ReleaseAsync : Conduit_Op<"release_async"> {
  let summary = "[Tier 2+3] non-blocking release — submit return of M buffer slots, return a completion token";
  let description = [{
    Non-blocking variant of conduit.release.  Submits the release of M buffer
    slots and returns a token representing the pending release.  Downstream
    ops that depend on the producer being unblocked can chain on this token.

    Use case: a pipeline stage that releases its output window and immediately
    allows the next pipeline stage to begin, without waiting for the release
    to propagate through the lock hardware.

    Equivalent to: conduit.release (blocking); use the async form when the
    release must be expressed as an SSA dependency for a subsequent async
    operation.

    Hardware lowering (Pass C):
      aie.use_lock(prodLock, Release, count)
      The token signals when the release is complete.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$count);
  let results   = (outs Conduit_AsyncTokenType:$token);
  let assemblyFormat = "attr-dict `:` type($token)";
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

    Two modes:
    - "distribute": 1 source → N destinations, each starting at a byte offset
    - "join":       N sources (each at a byte offset) → 1 destination

    Invariants (enforced by verifier):
    - distribute mode: when offsets is non-empty, offsets.size() == dsts.size()
    - join mode: when offsets is non-empty, offsets.size() == srcs.size()

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
  let summary = "[Tier 3] non-blocking N-D DMA send — submit a strided memref DMA transfer, return a completion token";
  let description = [{
    Non-blocking variant of conduit.put_memref.  Submits the DMA BD descriptor
    and returns immediately with a completion token.

    The returned token may be passed to conduit.wait (to block until complete),
    to conduit.wait_all (to block alongside other tokens), or to
    conduit.wait_all_async (to merge into a new token for further pipelining).

    Cross-tier note: this token may be combined in conduit.wait_all with a
    token from conduit.acquire_async (Tier 2).  This is the canonical
    cross-tier pattern that allows hardware to satisfy a DMA fill and a lock
    grant in parallel.

    Maps to: air.channel.put async [deps] (Pass B source).
  }];
  let arguments = (ins StrAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides);
  let results   = (outs Conduit_AsyncTokenType:$token);
  let assemblyFormat = "attr-dict `:` type($token)";
}

def Conduit_GetMemrefAsync : Conduit_Op<"get_memref_async"> {
  let summary = "[Tier 3] non-blocking N-D DMA receive — submit a DMA receive descriptor, return a completion token";
  let description = [{
    Non-blocking variant of conduit.get_memref.  Returns a completion token
    that the caller must wait on before accessing the received data.

    Maps to: air.channel.get async [deps] (Pass B source).
  }];
  let arguments = (ins StrAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides);
  let results   = (outs Conduit_AsyncTokenType:$token);
  let assemblyFormat = "attr-dict `:` type($token)";
}

// ---------------------------------------------------------------------------
// Token synchronization ops
// (combine Tier 2 and Tier 3 tokens in conduit.wait_all for cross-tier use)
// ---------------------------------------------------------------------------

def Conduit_Wait : Conduit_Op<"wait"> {
  let summary = "blocking wait on a single async token";
  let description = [{
    Blocks the current thread until the operation represented by %token is
    complete.  Equivalent to conduit.wait_all with a single token.

    Used for DMA tokens (from conduit.put_memref_async / conduit.get_memref_async).
    For acquire_async tokens, use conduit.wait_window instead — it returns
    the !conduit.window<T> SSA value needed by conduit.subview_access.

    Maps to: air.wait_all [%tok] (blocking, no result) on the AIR backend;
    a lock acquire on the AIE backend.
  }];
  let arguments = (ins Conduit_AsyncTokenType:$token);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def Conduit_WaitWindow : Conduit_Op<"wait_window"> {
  let summary = "[Tier 2+3 bridge] blocking wait on an acquire_async token, returning the granted window";
  let description = [{
    Blocks the current thread until the acquire_async token is complete
    (i.e., until the hardware lock grants the buffer window), then returns
    the !conduit.window<T> SSA value representing that window.

    This is the 17th op in the Conduit dialect.  It bridges the async token
    model (Tier 3) and the buffer-window model (Tier 2), completing the
    Option B async acquire pattern.

    The 'name' attribute identifies the conduit channel.  The result type
    must be !conduit.window<memref<N x element_type>> for a window of N
    elements with the given element type.

    Full async acquire pattern:
      %tok = conduit.acquire_async {name = "in", count = 1 : i64}
                 : !conduit.async.token
      %dma = conduit.put_memref_async {name = "in", ...} : !conduit.async.token
      conduit.wait_all %tok, %dma   // hardware satisfies both in parallel
      %win = conduit.wait_window %tok for "in"
                 : !conduit.async.token -> !conduit.window<memref<10xi32>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<10xi32>> -> memref<10xi32>
      conduit.release %win : !conduit.window<memref<10xi32>>

    For the simple blocking case, prefer conduit.acquire directly:
      %win = conduit.acquire {name = "in", count = 1 : i64, port = "Consume"}
                 : !conduit.window<memref<10xi32>>

    Hardware lowering (Pass C):
      The underlying lock acquire was performed by the acquire_async hardware
      path.  conduit.wait_window itself emits no new hardware ops — it maps
      the resolved token to the allocated aie.buffer SSA value.
  }];
  let arguments = (ins Conduit_AsyncTokenType:$token, StrAttr:$name);
  let results   = (outs Conduit_WindowType:$window);
  let assemblyFormat = "$token `for` $name attr-dict `:` type($token) `->` type($window)";
}

def Conduit_WaitAll : Conduit_Op<"wait_all"> {
  let summary = "blocking fan-in wait — block until all tokens in the list are complete";
  let description = [{
    Blocks the current thread until all tokens in %tokens are complete.  All
    specified async operations (DMA transfers, lock grants, releases) must
    have finished before execution continues.

    Accepts zero or more tokens (variadic).

    The canonical cross-tier use: conduit.wait_all %dma_tok, %acq_tok blocks
    until both the DMA fill (Tier 3 token) and the buffer window grant (Tier 2
    token) are complete, allowing the hardware to satisfy both in parallel.

    Maps to: air.wait_all [%tok0, %tok1, ...] (blocking, no result).
  }];
  let arguments = (ins Variadic<Conduit_AsyncTokenType>:$tokens);
  let assemblyFormat = "$tokens attr-dict";
}

def Conduit_WaitAllAsync : Conduit_Op<"wait_all_async"> {
  let summary = "non-blocking fan-in — merge multiple tokens into a new token, complete when all inputs complete";
  let description = [{
    Returns a new !conduit.async.token that is complete when all input tokens
    are complete.  Non-blocking: the returned token represents the pending
    completion of the fan-in.

    Use case: construct a merged dependency token from several parallel
    operations, then pass the merged token to a subsequent async op.

    Example:
      %t0 = conduit.put_memref_async {name = "a", num_elems = 64 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 64>,
                 strides = array<i64: 1>} : !conduit.async.token
      %t1 = conduit.acquire_async {name = "b", count = 1 : i64}
                 : !conduit.async.token
      %merged = conduit.wait_all_async %t0, %t1 :
          (!conduit.async.token, !conduit.async.token) -> !conduit.async.token

    Maps to: air.wait_all async [...] (Pass B source).
  }];
  let arguments = (ins Variadic<Conduit_AsyncTokenType>:$tokens);
  let results   = (outs Conduit_AsyncTokenType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

#endif // CONDUIT_DIALECT
