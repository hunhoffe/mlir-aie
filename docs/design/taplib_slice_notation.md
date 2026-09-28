# Slice notation in taplib and mlir-aie: options

Status: options A and B are implemented on branch
`claude/taplib-slice-notation` (`Layout.reshape`/`transpose`/`T`/
`broadcast_to`; `RuntimeData` views that `fill()`/`drain()` take directly).
C, D and E remain open.

This note asks how far NumPy-style slice notation can be pushed as the
primary way to describe DMA data movement, both in the IRON Python API and in
the MLIR dialects, and what each step would cost.

## 1. What a slice can and cannot say

A buffer descriptor is an offset plus outermost-first `(size, stride)` pairs
(plus, on some DMAs, a repeat count and zero padding). A `Layout` is exactly
that, so the question is which NumPy operations produce which BD fields:

| NumPy operation                     | BD effect                                   | taplib today          |
|-------------------------------------|---------------------------------------------|-----------------------|
| `a[i0:i1:s, ...]`                   | offset, sizes, strides of existing dims     | `Layout.slice` / `[]` |
| `a[3]` (integer index)              | offset, drops a dim                         | `Layout.slice` / `[]` |
| `a[None]`                           | size-1 dim                                  | `Layout.slice` / `[]` |
| `a.reshape(...)`                    | split / merge dims                          | `split`, `merge`, `coalesce` |
| `a.transpose(...)`, `a.T`           | reorder dims (order of the walk)            | `permute`             |
| `np.broadcast_to(a, (n, *a.shape))` | stride-0 outer dim (repeat)                 | `repeat`              |
| `np.pad(a, ...)`                    | zero padding around each dim                | `pad` (memtile only)  |

Everything a BD can do is one of these seven. Slicing alone covers only the
first three. The important consequence, checked in the experiment below, is
that slicing chooses **which** elements but never **in what order** they are
walked; order comes from the axis order, which needs `reshape` and
`transpose`. NumPy users already know this, which is the argument for
following NumPy exactly rather than inventing a slice-only dialect.

### Experiment (runs against this branch)

```python
from aie.helpers.taplib import Layout
M, K, m, k = 256, 128, 32, 16
grid = Layout.full((M, K)).tile((m, k))
v = Layout.full((M, K)).split(1, k).split(0, m)      # NumPy: reshape(M//m, m, K//k, k)
assert v[3, :, 5, :].stream_dims() == grid.at(3, 5).stream_dims()   # one tile: same
v[3, :, :, :].stream_dims()             # [(32,128), (8,16), (16,1)]  row-major within the tile row
grid.group((1, K // k))[3].stream_dims()  # [(8,16), (32,128), (16,1)] tile by tile
v.permute((0, 2, 1, 3))[3, :, :, :].stream_dims()   # tile by tile again
```

So `reshape` + indexing selects a tile; `reshape` + `transpose` + indexing
selects a *group* of tiles in tile order. `tile()`/`group()`/`order()` are
sugar over those three, and they can stay.

## 2. What already exists (verified in the tree)

Python, runtime side:

- `Layout.__getitem__` is NumPy basic indexing: ints, slices, `Ellipsis`,
  `None`, negative indices, and staged (`DispatchTime`) starts, stops and
  integer indices. `TileGrid.__getitem__` selects tiles.
- `RuntimeData.__getitem__` slices the host tensor:
  `weights[:n]` in `programming_examples/ml/resnet/layers_conv2_x/resnet.py`
  is already `fill(weightsFromL3, weights[:totalWeights_init])`.
- `fill`/`drain` take a `Layout` or `TensorAccessPattern` positionally, and
  `ObjectFifo`/`cons`/`forward`/`split`/`join` take a `Layout` for
  `dims_to_stream`/`dims_from_stream`.

Python, core side:

- `Buffer[...]` and an acquired object index through the eudsl `MemRefValue`:
  all-integer indices emit `memref.load`/`memref.store`, anything with a slice
  emits `memref.subview` (with rank reduction on request).

MLIR:

- `aie.dma_bd` and `aiex.npu.dma_memcpy_nd` carry sizes/strides as a mixed
  static/dynamic index list, the same shape `memref.subview` uses.
- `AIEX::traceSubviewToBlockArgument` (lib/Dialect/AIEX/Utils/AIEUtils.cpp)
  lets a BD's buffer operand be a chain of `memref.subview`/`view`/`cast`/
  `reinterpret_cast` rooted at a runtime-sequence argument, but only when
  the subview is static, unit-stride and row-major contiguous. The offset
  is folded into the address patch; the subview's shape is otherwise ignored.
- ObjectFifo core lowering already creates `memref.subview` for a join or
  distribute endpoint's segment (`ObjectFifoCoreEndpointOp::getAccessType`).

## 3. The gaps

1. `Layout` has no `reshape`, `transpose`/`T` or `broadcast_to`, so the
   NumPy spelling of a tile needs `split(1, k).split(0, m)`, which nobody
   will write. This is the only thing between taplib and full NumPy parity.
2. A slice of a runtime tensor is not bound to the tensor: `fill(A, A[...])`
   names `A` twice, and there is no `A.reshape(...)`.
3. In IR, a strided or dynamic `memref.subview` cannot be a BD's buffer. A
   frontend that wants to express "this BD walks this view" must compute the
   sizes/strides itself and pass them as BD dims, which is what IRON does.
4. Slicing an acquired object on the core produces a strided memref that no
   kernel signature accepts, so core-side slicing is useful only for
   elementwise Python code, not for handing a sub-tile to a C++ kernel.

## 4. Options, Python side

### A. NumPy parity on `Layout` (small, no compiler change)

Add `reshape(*shape)`, `transpose(*axes)` / `T`, `broadcast_to(shape)` and
keep `pad`. `reshape` is a sequence of `split`/`merge` steps derived by
matching the old and new shapes left to right, with `-1` allowed; it raises
where a merge is not contiguous, exactly as NumPy would have to copy.
`TileGrid` becomes sugar (it already is):

```python
A = Layout.full((M, K))
tile   = A.reshape(M // m, m, K // k, k)[i, :, j, :]              # == A.tile((m, k)).at(i, j)
row    = A.reshape(M // m, m, K // k, k).transpose(0, 2, 1, 3)[i]  # == A.tile((m, k)).group((1, K//k))[i]
colB   = Layout.full((K, N)).reshape(K // k, k, N // n, n).transpose(2, 0, 1, 3)[j]
twice  = tile.broadcast_to((2, *tile.sizes))                       # == tile.repeat(2)
```

Cost: about 80 lines in `layout.py`, tests in `test/python/taplib`. Staged
sizes work wherever `split` works today (concrete divisor). Risk: a reader
must know that axis order is walk order, which the docs must say once.

### B. Bound views on runtime tensors (small, no compiler change)

Make `RuntimeData.__getitem__`, `.reshape`, `.transpose`, `.T` and
`.broadcast_to` return a `View` that carries both the tensor and a `Layout`,
and let `fill`/`drain` take a `View` as the single positional argument:

```python
def sequence(A, B, C, a_prod, b_prod, c_cons):
    A4 = A.reshape(M // m, m, K // k, k).transpose(0, 2, 1, 3)
    B4 = B.reshape(K // k, k, N // n, n).transpose(2, 0, 1, 3)
    for i in range(M // m):
        tg = TaskGroup()
        a_prod.fill(A4[i], group=tg)                      # one argument
        b_prod.fill(B4.reshape(-1), group=tg)                 # all of B, column of tiles first
        c_cons.drain(C.reshape(M // m, m, N // n, n).transpose(0, 2, 1, 3)[i], group=tg, wait=True)
```

Dispatch-time shapes fall out: `A4[iv]` with a `range_` induction variable is
a staged offset, and `A[iv * m:(iv + 1) * m]` already works today. A `View`
would also expose `.tap()` and `.stream_dims()` so `fill(A, A[...])` keeps
working unchanged.

Cost: a `View` class (about 60 lines), one branch in `emit_shim_transfer`,
and `RuntimeData` growing the four methods. No IR change: the view lowers to
the same `dma_memcpy_nd` as today.

### C. Core-side slicing (already there, limited)

`obj[0:8, :]` on an acquired object emits `memref.subview`. Making that
usable for kernels means either accepting strided memrefs in kernel
declarations (the C++ kernel then takes a pointer plus strides, which our
kernels do not) or restricting slices to contiguous ones and lowering to a
pointer offset. The contiguous case is what join/distribute already does.
Worth documenting; not worth extending until a kernel needs a strided view.

## 5. Options, MLIR side

### D. `memref.subview` chains as the IR form of a Layout (medium)

Today `traceSubviewToBlockArgument` folds only the offset of a contiguous
subview. Extend the shim lowering so that a BD whose buffer operand is a
chain of `memref.subview`, `memref.transpose`, `memref.expand_shape`,
`memref.collapse_shape` and `memref.cast` rooted at a runtime argument, and
that gives **no sizes/strides of its own**, derives its offset, sizes and
strides from the result type's strided layout. Static and dynamic offsets
and sizes are both fine, since `dma_bd`/`dma_memcpy_nd` already take mixed
lists.

```mlir
%v  = memref.subview %A[%i, 0] [32, 128] [1, 1] : memref<256x128xi16> to memref<32x128xi16, strided<[128, 1], offset: ?>>
%e  = memref.expand_shape %v [[0], [1, 2]] output_shape [32, 8, 16] : ... into memref<32x8x16xi16, strided<[128, 16, 1], offset: ?>>
%t  = memref.transpose %e (d0, d1, d2) -> (d1, d0, d2) : ... to memref<8x32x16xi16, strided<[16, 128, 1], offset: ?>>
aiex.npu.dma_memcpy_nd(%t) { metadata = @of_a, id = 0 } : memref<8x32x16xi16, strided<[16, 128, 1], offset: ?>>
```

lowers as if written with `[0, 8, 32, 16][0, 16, 128, 1]`. What this buys:

- the IR says what is walked in upstream vocabulary, readable and
  canonicalizable (subview-of-subview folds upstream);
- other frontends (AIR, IREE plugins, hand-written MLIR) get taplib's
  expressiveness without a Python dependency;
- a dynamic slice is a dynamic subview, so dispatch-time sequences look the
  same as static ones.

What it cannot say: repeat (a stride-0 subview is not something upstream
memref passes expect, even where the verifier lets it through) and padding. Both stay BD attributes (`repeat`/iteration and `pad_dimensions`),
which is consistent with them being DMA-engine features rather than views.

Cost: one lowering helper that walks the chain and accumulates a strided
layout (mixed static/dynamic), used by `AIEDMATasksToNPU.cpp` and
`AIEDmaToNpu.cpp` in place of the contiguity check; verifier updates so a BD
with a strided buffer operand and explicit dims is rejected as ambiguous;
lit tests for each op in the chain, static and dynamic. Roughly 300 lines of
C++ plus tests. IRON would not have to change to benefit, but option B
could emit the chain instead of computing dims once D exists.

### E. Layout attribute on `aie.objectfifo` (low value, skip)

`dimensionsToStream`/`dimensionsFromStream` could be a strided-layout
attribute instead of a `BDDimLayoutArray`. It reads better but changes an
attribute every downstream tool parses, for no new capability. Not
recommended now.

## 6. Recommendation

1. Do A and B together. They are Python-only, small, backward compatible, and
   they make the NumPy spelling the obvious one in examples: reshape,
   transpose, slice. `tile`/`group`/`order` remain as the tiled sugar and the
   docs present them as such.
2. Do D as the compiler-side companion, scoped to the shim path (runtime
   arguments) first. It is the piece that makes "slice notation" mean
   something in the IR rather than only in Python, and it is independent of
   A and B.
3. Leave C and E.

Open questions to settle before A/B: whether `reshape` on a staged size is
allowed when the divisor is concrete (proposed yes, matching `split`),
whether negative indices on a staged dimension stay an error (proposed yes),
and whether `View` should be the type `fill`/`drain` document first with
`tap=` kept as the escape hatch (proposed yes).
