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
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Custom attribute constraints for AIE BD dimension layout attrs.
// Conduit depends on AIE at C++ level (ConduitDialect.h includes AIEDialect.h),
// so the C++ types are available.  We define CPred-based constraints here
// instead of including AIEAttrs.td to avoid TableGen re-generating AIE enums
// in Conduit's .inc files (which causes redefinition errors).
def AIE_BDDimLayoutArrayAttr : Attr<
    CPred<"::llvm::isa<::xilinx::AIE::BDDimLayoutArrayAttr>($_self)">,
    "AIE::BDDimLayoutArrayAttr"> {
  let storageType = [{ ::mlir::Attribute }];
  let returnType = [{ ::mlir::Attribute }];
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

def AIE_BDDimLayoutArrayArrayAttr : Attr<
    CPred<"::llvm::isa<::xilinx::AIE::BDDimLayoutArrayArrayAttr>($_self)">,
    "AIE::BDDimLayoutArrayArrayAttr"> {
  let storageType = [{ ::mlir::Attribute }];
  let returnType = [{ ::mlir::Attribute }];
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
}

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
      (--air-channel-to-conduit)

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
// Dialect-scoped typed array attribute for flat symbol references.
//
// FlatSymbolRefArrayAttr already exists in mlir/IR/CommonAttrConstraints.td,
// but we define a dialect-scoped alias here to (1) make the origin explicit
// in Conduit.td, and (2) allow future per-element constraints without
// touching the generic MLIR definition.
//
// Used by: conduit.scatter ($dsts), conduit.gather ($srcs),
//          conduit.transpose ($srcs, $dsts).
// ---------------------------------------------------------------------------
def Conduit_FlatSymbolRefArrayAttr
    : TypedArrayAttrBase<FlatSymbolRefAttr,
                         "flat symbol reference array attribute">;

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
// RoutingMode enum — #conduit.routing_mode<circuit> / <packet> / etc.
//
// Replaces OptionalAttr<StrAttr> routing_mode on conduit.create.
// Absent attribute means "unresolved" (previously spelled "any").
// ---------------------------------------------------------------------------

def Conduit_RoutingMode : I32EnumAttr<"RoutingMode", "Conduit routing mode", [
  I32EnumAttrCase<"Circuit",      0, "circuit">,
  I32EnumAttrCase<"Packet",       1, "packet">,
  I32EnumAttrCase<"Cascade",      2, "cascade">,
  I32EnumAttrCase<"Stream",       3, "stream">,
  I32EnumAttrCase<"SharedMemory", 4, "shared_memory">,
  I32EnumAttrCase<"DMA",          5, "dma">
]> {
  let cppNamespace = "::xilinx::conduit";
  let genSpecializedAttr = 0;
}
def Conduit_RoutingModeAttr : EnumAttr<Conduit_Dialect, Conduit_RoutingMode, "routing_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

// ---------------------------------------------------------------------------
// SyncMode enum — synchronization mode for relay ops
//
// barrier (default): all endpoints synchronize on a single lock protocol
// independent: each endpoint has its own lock pair (N-way ack protocol)
// ---------------------------------------------------------------------------

def Conduit_SyncMode : I32EnumAttr<"SyncMode", "Conduit synchronization mode", [
  I32EnumAttrCase<"Barrier",     0, "barrier">,
  I32EnumAttrCase<"Independent", 1, "independent">,
  I32EnumAttrCase<"None",        2, "none">
]> {
  let cppNamespace = "::xilinx::conduit";
  let genSpecializedAttr = 0;
}
def Conduit_SyncModeAttr : EnumAttr<Conduit_Dialect, Conduit_SyncMode, "sync_mode"> {
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

def Conduit_Create : Conduit_Op<"create", [HasParent<"::xilinx::AIE::DeviceOp">, Symbol]> {
  let summary = "declare a named conduit channel with a given slot_elems count";
  let description = [{
    Declares a named FIFO channel (slot_elems elements per slot).
    Must appear before any Tier 2 or Tier 3 ops that reference the same name.

    Name-based linkage is used (string attribute, not SSA value) because
    channels are referenced across IsolatedFromAbove region boundaries.

    The meaning of 'slot_elems' is tier-specific:
    - Tier 2: total number of buffer element slots (depth * elemCount)
    - Tier 3: not directly enforced; channel slot_elems is implicitly the
              memref allocation size on the consumer tile

    Attributes:
    - element_type    : TypeAttr (required) — element memref type (e.g. memref<10xi32>).
                        Buffer capacity = depth * product(element_type.shape).
                        Pass C uses this for aie.buffer allocation.
    - depth           : I64Attr (optional) — buffer slot count. 0 = unresolved sentinel
                        (set by Pass A/B, resolved by --conduit-depth-promote).
    - routing_mode    : optional enum — absent = heuristic (Pass D resolves).
                        shared_memory: force shared mem; error if not adjacent.
                        dma: force DMA; infer-modes picks circuit or packet.
                        circuit: pin circuit-switched DMA; skips shared mem.
                        packet: pin packet-switched DMA.
                        cascade: cascade stream.
                        stream: AXI stream direct (no DMA BD).
    - sync_mode       : optional enum — absent = barrier (default).
                        none: suppress all lock allocation/emission (keeps DMA).
                        independent: per-endpoint lock pair.
    - producer_rates  : optional DenseI64Array — CSDF producer token-count per phase.
    - consumer_rates  : optional DenseI64Array — CSDF consumer token-count per phase.
                        Both must appear together. Verifier checks Lee-Messerschmitt
                        balance: sum(P)*len(C) == sum(C)*len(P).
    - fusion_group    : optional string — IRON hint pairing this channel with one other
                        for spatial fusion via --conduit-fuse-operators.
    - bd_repeat       : lowering annotation — BD unroll factor (Pass A → Pass C).
    - dma_repeat      : lowering annotation — number of times the lowered
                        shim/memtile BD chain replays per dispatch
                        (Pass A → Pass C).  Meaningful only for routing
                        modes that lower to DMA: circuit, packet, dma,
                        shared_memory, or absent (default DMA inference).
                        Stream and cascade routing emit no shim DMA chain,
                        so dma_repeat has no effect there and SHOULD NOT
                        be set on stream/cascade-routed channels.
    - producer_dimensions : optional BDDimLayoutArray — default MM2S DMA layout transform.
                        Per-op attrs on put_memref/put_memref_async override this default.
    - consumer_dimensions : optional BDDimLayoutArrayArray — default S2MM DMA layout transforms.
                        Per-op attrs on get_memref/get_memref_async override this default.

    Example:
      conduit.create @c1 {element_type = memref<10xi32>, depth = 1 : i64}

    CSDF example:
      conduit.create @csdf {element_type = memref<i32>, depth = 6 : i64,
                            producer_rates = array<i64: 1, 2>,
                            consumer_rates = array<i64: 1, 2>}
      // Balance: sum(P)*len(C) = 3*2 = 6 == sum(C)*len(P) = 3*2 = 6 ✓

    Fusion hint example:
      conduit.create @a_out {element_type = memref<128xbf16>, depth = 2 : i64,
                             fusion_group = "a_to_b"}
      conduit.create @b_in  {element_type = memref<128xbf16>, depth = 2 : i64,
                             fusion_group = "a_to_b"}
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       // Semantic core (all required except routing_mode/sync_mode/rates)
                       TypeAttr:$element_type,
                       OptionalAttr<I64Attr>:$depth,
                       // Routing and synchronization (optional — absent = heuristic)
                       // routing_mode values:
                       //   shared_memory — force shared mem; error if not adjacent
                       //   dma          — force DMA; infer-modes picks circuit or packet
                       //   circuit      — pin circuit-switched DMA (skips shared mem)
                       //   packet       — pin packet-switched DMA
                       //   cascade      — cascade stream
                       //   stream       — AXI stream direct (no DMA BD)
                       OptionalAttr<Conduit_RoutingModeAttr>:$routing_mode,
                       // sync_mode values:
                       //   barrier     — single lock protocol (default)
                       //   independent — per-endpoint lock pair
                       //   none        — no lock allocation or emission
                       OptionalAttr<Conduit_SyncModeAttr>:$sync_mode,
                       // CSDF rates (must appear together)
                       OptionalAttr<DenseI64ArrayAttr>:$producer_rates,
                       OptionalAttr<DenseI64ArrayAttr>:$consumer_rates,
                       // Fusion hint: IRON stamps this to pair two conduit.create ops
                       // as candidates for --conduit-fuse-operators (spatial fusion).
                       // The pass reads D_A from producer put_memref descriptors and
                       // D_B from consumer get_memref descriptors, computes D_B ∘ D_A,
                       // and selects: shared_memory (identity+adjacent), direct DMA
                       // (geometric 1:1), or conduit.transpose relay (geometric N:M).
                       OptionalAttr<StrAttr>:$fusion_group,
                       // Lowering annotations (set by Pass A/B, consumed by Pass C only)
                       // bd_repeat: BD-level unroll factor; effectiveBDs = depth * bd_repeat
                       OptionalAttr<I64Attr>:$bd_repeat,
                       // dma_repeat: shim/memtile BD-chain replay count per
                       // dispatch; DMAStartOp.repeat_count = dma_repeat - 1.
                       // Meaningful only for DMA-lowered routing modes
                       // (circuit, packet, dma, shared_memory, or absent).
                       // Has no effect on stream/cascade-routed channels —
                       // do not set it on those.
                       OptionalAttr<I64Attr>:$dma_repeat,
                       // producer_dimensions: default BDDimLayout for MM2S BD chains.
                       // Per-op attrs on put_memref/put_memref_async override this.
                       OptionalAttr<AIE_BDDimLayoutArrayAttr>:$producer_dimensions,
                       // consumer_dimensions: default BDDimLayout for S2MM BD chains.
                       // Per-op attrs on get_memref/get_memref_async override this.
                       OptionalAttr<AIE_BDDimLayoutArrayArrayAttr>:$consumer_dimensions
  );
  // Print as: conduit.create @sym_name { slot_elems = ..., ... }
  // sym_name is consumed by $sym_name so attr-dict omits it automatically.
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    // Backward-compatibility accessors so existing C++ call sites using
    // getName() / getNameAttr() continue to compile without changes.
    // Phase 2 will remove these once all use-sites are updated.
    ::llvm::StringRef getName() { return getSymName(); }
    ::mlir::StringAttr getNameAttr() { return getSymNameAttr(); }
  }];
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
  // Pass C infers the AcquireGreaterEqual lock delta automatically from
  // the live-window state it tracks during lowering.  count=N means
  // "I need a window of N elements total" — no compiler annotation needed.
  let arguments = (ins FlatSymbolRefAttr:$name, I64Attr:$count, Conduit_PortAttr:$port,
                       // window_size: maximum number of buffer slots held simultaneously
                       // by a sliding-window consumer (acquire_count > release_count).
                       // M7-window check: the channel depth must be >= window_size.
                       OptionalAttr<I64Attr>:$window_size);
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

    The 'port' attribute is required and specifies which side of the channel
    is acquiring:
    - "Consume": consumer-side acquisition (most common; wait for data to arrive)
    - "Produce": producer-side acquisition (wait for a free output slot)

    Pass C uses 'port' to select the correct lock (consLock vs prodLock) when
    emitting the AcquireGreaterEqual use_lock op at conduit.wait_window time.

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
      %acq_tok = conduit.acquire_async {name = "output", count = 1 : i64,
                     port = #conduit.port<Consume>}
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
  let arguments = (ins FlatSymbolRefAttr:$name, I64Attr:$count, Conduit_PortAttr:$port,
                       // window_size: maximum number of buffer slots held simultaneously
                       // by a sliding-window consumer (acquire_count > release_count).
                       // M7-window check: the channel depth must be >= window_size.
                       OptionalAttr<I64Attr>:$window_size);
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
  let hasVerifier = 1;
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

    The optional '$window' SSA operand enables type-checker verification that
    the value being released is actually a !conduit.window<T> that was acquired
    for this channel.  When present, the verifier confirms:
      (1) the operand type is !conduit.window<T>
      (2) the channel name of the defining op matches '$name'

    When '$window' is omitted, the op identifies the channel purely by '$name'
    (name-only release path).  This is valid for producer-side standalone
    release ops (e.g. signalling data ready without an explicit prior acquire
    in the same scope).

    Example — type-checked consumer release after conduit.acquire:
      %win = conduit.acquire {name = "fifo", count = 1 : i64,
                              port = #conduit.port<Consume>}
                 : !conduit.window<memref<8xi32>>
      // ... compute ...
      %rel_tok = conduit.release_async %win {name = "fifo", count = 1 : i64,
                     port = #conduit.port<Consume>}
                     : !conduit.window<memref<8xi32>> -> !conduit.window.token

    Example — name-only producer release (no prior acquire):
      %rel_tok = conduit.release_async {name = "fifo", count = 1 : i64,
                     port = #conduit.port<Produce>}
                     : !conduit.window.token

    Hardware lowering (Pass C):
      port="Consume": aie.use_lock(prodLock, Release, count) -- free up producer slots
      port="Produce": aie.use_lock(consLock, Release, count) -- signal data ready
      The token signals when the release is complete.
  }];
  let arguments = (ins FlatSymbolRefAttr:$name, I64Attr:$count, Conduit_PortAttr:$port,
                   Optional<Conduit_WindowType>:$window);
  let results   = (outs Conduit_WindowTokenType:$token);
  // Custom format: optional $window operand before attrs.
  //   With window:    conduit.release_async %win attr-dict : type($window) -> type($token)
  //   Without window: conduit.release_async attr-dict : type($token)
  let assemblyFormat = "(`(` $window^ `:` type($window) `)`)? attr-dict `:` type($token)";
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

def Conduit_ScatterOp : Conduit_Op<"scatter", []> {
  let summary = "1:N fan-out relay — routes one source channel to N destination channels via a MemTile";
  let description = [{
    Routes data from one source channel to N destination channels through a MemTile
    DMA relay. The memtile operand is a FlatSymbolRef to an aie.tile op (row=1).
    offsets: optional per-destination element offsets into the MemTile buffer
    (absent = uniform — all destinations start at offset 0, full buffer length).
    Each relay edge (compute→MemTile or MemTile→compute) may carry a fusion_group
    hint on its conduit.create for --conduit-fuse-operators.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$src,
    Conduit_FlatSymbolRefArrayAttr:$dsts,
    // memtile: "tile(col,row)" string identifying the MemTile relay tile.
    // TODO (post-Sprint 6): migrate to FlatSymbolRefAttr once AIE tile ops
    // carry sym_names. The current StrAttr format is validated by the verifier.
    StrAttr:$memtile,
    OptionalAttr<DenseI64ArrayAttr>:$offsets
  );
  let assemblyFormat = [{
    `{` `src` `=` $src `,` `dsts` `=` $dsts attr-dict `}`
  }];
  let hasVerifier = 1;
}

def Conduit_GatherOp : Conduit_Op<"gather", []> {
  let summary = "N:1 fan-in relay — collects N source channels into one destination channel via a MemTile";
  let description = [{
    Collects data from N source channels into one destination channel through a
    MemTile DMA relay. offsets: optional per-source element offsets (absent = uniform).
  }];
  let arguments = (ins
    Conduit_FlatSymbolRefArrayAttr:$srcs,
    FlatSymbolRefAttr:$dst,
    StrAttr:$memtile,
    OptionalAttr<DenseI64ArrayAttr>:$offsets
  );
  let assemblyFormat = [{
    `{` `srcs` `=` $srcs `,` `dst` `=` $dst attr-dict `}`
  }];
  let hasVerifier = 1;
}

def Conduit_TransposeOp : Conduit_Op<"transpose", []> {
  let summary = "N:M static redistribution relay using time-multiplexed DMA via a MemTile";
  let description = [{
    Redistributes data from N source channels to M destination channels through a
    MemTile using time-multiplexed DMA. offsets (required): N*M element offsets
    encoding the redistribution map — offsets[i*M + j] is the MemTile offset for
    source i to destination j. Packet ID budget: N*M <= 32 (AIE2 packet ID space).
  }];
  let arguments = (ins
    Conduit_FlatSymbolRefArrayAttr:$srcs,
    Conduit_FlatSymbolRefArrayAttr:$dsts,
    StrAttr:$memtile,
    DenseI64ArrayAttr:$offsets
  );
  let assemblyFormat = [{
    `{` `srcs` `=` `[` $srcs `]` `,` `dsts` `=` `[` $dsts `]` attr-dict `}`
  }];
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
  let arguments = (ins FlatSymbolRefAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides,
                       // producer_dimensions: pre-computed BDDimLayout for the MM2S
                       // BD chain. Set by Pass A when the objectfifo carries
                       // dimensionsToStream. The fusion pass reads this as D_A when
                       // computing the composed transformation D_B ∘ D_A.
                       OptionalAttr<AIE_BDDimLayoutArrayAttr>:$producer_dimensions,
                       // arg_index: index of the aie.runtime_sequence block
                       // argument that this op's DMA descriptor binds to.
                       // Set by --dma-task-to-conduit (sourced from the original
                       // aie.dma_bd's BlockArgument index).  Required when the op
                       // is rebuilt back to aiex.dma_configure_task_for in
                       // --conduit-to-dma Step 8g; absence triggers a hard
                       // failure (signals upstream pipeline bug).  Optional in
                       // the dialect to accommodate hand-written verifier tests
                       // and AIR-side variants that never reach Step 8g.
                       OptionalAttr<I64Attr>:$arg_index);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Conduit_GetMemref : Conduit_Op<"get_memref"> {
  let summary = "[Tier 3] blocking N-D DMA receive — receive a strided memref tile from a conduit";
  let description = [{
    Tier 3 memref-DMA op.  Receives a DMA transfer of a strided N-D memref
    tile from the named conduit into local memory.  Blocks until data is
    available.

    Maps to: air.channel.get (Pass B source); aie.dma_bd (Pass C output).
  }];
  let arguments = (ins FlatSymbolRefAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides,
                       // consumer_dimensions: pre-computed BDDimLayout for the S2MM
                       // BD chain. Set by Pass A when the objectfifo carries
                       // dimensionsFromStream. The fusion pass reads this as D_B when
                       // computing the composed transformation D_B ∘ D_A.
                       OptionalAttr<AIE_BDDimLayoutArrayArrayAttr>:$consumer_dimensions,
                       // arg_index: index of the aie.runtime_sequence block
                       // argument that this op's DMA descriptor binds to.
                       // Set by --dma-task-to-conduit (sourced from the original
                       // aie.dma_bd's BlockArgument index).  Required when the op
                       // is rebuilt back to aiex.dma_configure_task_for in
                       // --conduit-to-dma Step 8g; absence triggers a hard
                       // failure (signals upstream pipeline bug).  Optional in
                       // the dialect to accommodate hand-written verifier tests
                       // and AIR-side variants that never reach Step 8g.
                       OptionalAttr<I64Attr>:$arg_index);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
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
  let arguments = (ins FlatSymbolRefAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides,
                       Variadic<Conduit_AnyTokenType>:$deps,
                       OptionalAttr<AIE_BDDimLayoutArrayAttr>:$producer_dimensions);
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
  let arguments = (ins FlatSymbolRefAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides,
                       Variadic<Conduit_AnyTokenType>:$deps,
                       OptionalAttr<AIE_BDDimLayoutArrayArrayAttr>:$consumer_dimensions);
  let results   = (outs Conduit_DMATokenType:$token);
  let assemblyFormat = "(`[` $deps^ `:` type($deps) `]`)? attr-dict `:` type($token)";
  let hasVerifier = 1;
}

// ---------------------------------------------------------------------------
// Token synchronization ops
// (combine Tier 2 and Tier 3 tokens in conduit.wait_all for cross-tier use)
// ---------------------------------------------------------------------------


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
  let arguments = (ins Conduit_WindowTokenType:$token, FlatSymbolRefAttr:$name);
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
