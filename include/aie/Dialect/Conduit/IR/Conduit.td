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
// Conduit is a portable bridge dialect between mlir-aie objectfifo semantics
// and mlir-air channel semantics.  It models typed FIFO channels with token
// and buffer-window access patterns that lower to either backend.
//
//===----------------------------------------------------------------------===//

#ifndef CONDUIT_DIALECT
#define CONDUIT_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// ---------------------------------------------------------------------------
// Dialect definition
// ---------------------------------------------------------------------------

def Conduit_Dialect : Dialect {
  let name = "conduit";
  let cppNamespace = "::xilinx::conduit";
  let summary = "Conduit canonical dataflow IR";
  let description = [{
    Conduit is a portable bridge dialect between mlir-aie objectfifo semantics
    and mlir-air channel semantics. It models typed FIFO channels with token
    and buffer-window access patterns that lower to either backend.
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
    Conduit operation (put_async, get_async, put_memref_async, get_memref_async,
    or wait_all_async).  Tokens may be passed to conduit.wait, conduit.wait_all,
    conduit.wait_all_async, or listed as deps on subsequent async ops.

    Maps to:
    - mlir-air  : !air.async.token
    - mlir-aie  : lock acquire/release pair (no explicit token type in AIE)
  }];
  let parameters = (ins);
  let assemblyFormat = "";
}

// ---------------------------------------------------------------------------
// Core ops
// ---------------------------------------------------------------------------

def Conduit_Create : Conduit_Op<"create"> {
  let summary = "allocate a conduit FIFO channel";
  let description = [{
    Declares a named FIFO channel with a given token capacity.
    Must appear before any put/get/peek/acquire on the same name.

    Example:
      conduit.create {name = "c1", capacity = 32 : i64}
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$capacity);
  let assemblyFormat = "attr-dict";
}

def Conduit_Put : Conduit_Op<"put"> {
  let summary = "push one integer token into a conduit";
  let description = [{
    Enqueues a single integer token onto the named conduit.
    Corresponds to aie.objectfifo.acquire/subview.access/release or
    air.channel.put on the target backends.

    Example:
      conduit.put {name = "c1", value = 42 : i64}
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$value);
  let assemblyFormat = "attr-dict";
}

def Conduit_Get : Conduit_Op<"get", [Pure]> {
  let summary = "pop and return the front token from a conduit";
  let description = [{
    Dequeues the front token and returns its value.

    Example:
      %v = conduit.get {name = "c1"} : i64
  }];
  let arguments = (ins StrAttr:$name);
  let results   = (outs I64:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Conduit_Peek : Conduit_Op<"peek", [Pure]> {
  let summary = "inspect a token at offset without consuming it";
  let description = [{
    Returns the token at the given 0-based offset from the front of the FIFO
    without advancing the read pointer.  offset=0 is the front token.

    Example:
      %v = conduit.peek {name = "c1", offset = 2 : i64} : i64
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$offset);
  let results   = (outs I64:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def Conduit_Advance : Conduit_Op<"advance"> {
  let summary = "discard N tokens from the front of a conduit";
  let arguments = (ins StrAttr:$name, I64Attr:$count);
  let assemblyFormat = "attr-dict";
}

def Conduit_Prefill : Conduit_Op<"prefill"> {
  let summary = "pre-load a list of integer tokens into a conduit";
  let arguments = (ins StrAttr:$name, DenseI64ArrayAttr:$values);
  let assemblyFormat = "attr-dict";
}

def Conduit_Acquire : Conduit_Op<"acquire"> {
  let summary = "reserve a window of N tokens for in-place processing";
  let description = [{
    Maps to aie.objectfifo.acquire on the AIE backend.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$count);
  let assemblyFormat = "attr-dict";
}

def Conduit_Release : Conduit_Op<"release"> {
  let summary = "release a previously acquired token window";
  let description = [{
    Maps to aie.objectfifo.release on the AIE backend.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$count);
  let assemblyFormat = "attr-dict";
}

def Conduit_Merge : Conduit_Op<"merge"> {
  let summary = "merge tokens from multiple source conduits into one";
  let description = [{
    High-level fan-in sugar; distinct from conduit.objectfifo_link.
  }];
  let arguments = (ins StrArrayAttr:$srcs, StrAttr:$dst);
  let assemblyFormat = "attr-dict";
}

def Conduit_Fork : Conduit_Op<"fork"> {
  let summary = "broadcast one token from a source conduit to multiple sinks";
  let description = [{
    High-level fan-out sugar; distinct from conduit.objectfifo_link mode="distribute".
  }];
  let arguments = (ins StrAttr:$src, StrArrayAttr:$dsts);
  let assemblyFormat = "attr-dict";
}

def Conduit_Annotate : Conduit_Op<"annotate"> {
  let summary = "attach lowering hint attributes to a conduit";
  let description = [{
    Example:
      conduit.annotate {name = "c1", key = "lower_to", value = "objectfifo"}
  }];
  let arguments = (ins StrAttr:$name, StrAttr:$key, StrAttr:$value);
  let assemblyFormat = "attr-dict";
}

def Conduit_Status : Conduit_Op<"status", [Pure]> {
  let summary = "query the current token count of a conduit";
  let arguments = (ins StrAttr:$name);
  let results   = (outs I64:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

// ---------------------------------------------------------------------------
// Extension stubs (RFC-000, RFC-001, RFC-002)
// ---------------------------------------------------------------------------

def Conduit_ObjectFifoLink : Conduit_Op<"objectfifo_link"> {
  let summary = "link objectfifo-style src conduits to dst conduits via MemTile";
  let description = [{
    Represents aie.objectfifo.link semantics.

    Attributes:
    - srcs    : list of source conduit names (always a list)
    - dsts    : list of destination conduit names
    - mode    : "distribute" or "join"
    - memtile : MemTile coordinate string hint
    - offsets : optional byte offsets
    - lock_id : optional integer

    Example:
      conduit.objectfifo_link {srcs = ["of_in"], dsts = ["of_out0", "of_out1"],
                               mode = "distribute", memtile = "tile(0,1)",
                               offsets = [0 : i64, 1024 : i64]}
  }];
  let arguments = (ins StrArrayAttr:$srcs,
                       StrArrayAttr:$dsts,
                       StrAttr:$mode,
                       StrAttr:$memtile,
                       OptionalAttr<DenseI64ArrayAttr>:$offsets,
                       OptionalAttr<I64Attr>:$lock_id);
  let assemblyFormat = "attr-dict";
}

def Conduit_PutMemref : Conduit_Op<"put_memref"> {
  let summary = "transfer a strided memref tile into a conduit (blocking)";
  let arguments = (ins StrAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides);
  let assemblyFormat = "attr-dict";
}

def Conduit_GetMemref : Conduit_Op<"get_memref"> {
  let summary = "receive a strided memref tile from a conduit (blocking)";
  let arguments = (ins StrAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides);
  let assemblyFormat = "attr-dict";
}

def Conduit_PutAsync : Conduit_Op<"put_async"> {
  let summary = "non-blocking scalar put; returns a !conduit.async.token";
  let arguments = (ins StrAttr:$name, I64Attr:$value);
  let results   = (outs Conduit_AsyncTokenType:$token);
  let assemblyFormat = "attr-dict `:` type($token)";
}

def Conduit_GetAsync : Conduit_Op<"get_async"> {
  let summary = "non-blocking scalar get; returns a !conduit.async.token";
  let arguments = (ins StrAttr:$name);
  let results   = (outs Conduit_AsyncTokenType:$token);
  let assemblyFormat = "attr-dict `:` type($token)";
}

def Conduit_PutMemrefAsync : Conduit_Op<"put_memref_async"> {
  let summary = "non-blocking memref put; returns a !conduit.async.token";
  let arguments = (ins StrAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides);
  let results   = (outs Conduit_AsyncTokenType:$token);
  let assemblyFormat = "attr-dict `:` type($token)";
}

def Conduit_GetMemrefAsync : Conduit_Op<"get_memref_async"> {
  let summary = "non-blocking memref get; returns a !conduit.async.token";
  let arguments = (ins StrAttr:$name,
                       I64Attr:$num_elems,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides);
  let results   = (outs Conduit_AsyncTokenType:$token);
  let assemblyFormat = "attr-dict `:` type($token)";
}

def Conduit_Wait : Conduit_Op<"wait"> {
  let summary = "blocking wait for a single async token to complete";
  let arguments = (ins Conduit_AsyncTokenType:$token);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def Conduit_WaitAll : Conduit_Op<"wait_all"> {
  let summary = "blocking wait for all tokens in a list to complete";
  let arguments = (ins Variadic<Conduit_AsyncTokenType>:$tokens);
  let assemblyFormat = "operands attr-dict (`:` type(operands)^)?";
}

def Conduit_WaitAllAsync : Conduit_Op<"wait_all_async"> {
  let summary = "non-blocking token merge; returns a new async token";
  let arguments = (ins Variadic<Conduit_AsyncTokenType>:$tokens);
  let results   = (outs Conduit_AsyncTokenType:$result);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def Conduit_Chain : Conduit_Op<"chain"> {
  let summary = "declare a dependency edge from producer token(s) to a consumer token";
  let description = [{
    Void dependency-annotation op; has no result.
    Use conduit.wait_all_async to combine tokens into a new merged token.
  }];
  let arguments = (ins Variadic<Conduit_AsyncTokenType>:$producers,
                       Conduit_AsyncTokenType:$consumer);
  let assemblyFormat = "`[` $producers `]` `->` $consumer attr-dict "
                       "`:` functional-type($producers, type($consumer))";
}

def Conduit_SubviewAccess : Conduit_Op<"subview_access", [Pure]> {
  let summary = "access element N of a previously acquired objectfifo subview";
  let description = [{
    Covers the 8,392-use gap of aie.objectfifo.subview.access.
  }];
  let arguments = (ins StrAttr:$name, I64Attr:$index);
  let results   = (outs I64:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

#endif // CONDUIT_DIALECT
