# data.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

from __future__ import annotations

from typing import Any, Iterator, Sequence, get_origin

import numpy as np

from ...extras.dialects.memref import (  # pyright: ignore[reportMissingImports]
    MemRefValue,
)
from ...helpers.npdtypes import (
    NpuDType,
    np_ndarray_type_get_dtype,
    np_ndarray_type_get_shape,
)
from ...helpers.taplib import Layout, TensorAccessPattern, TileGrid


class RuntimeData:
    """A handle to I/O data in the Runtime.

    Indexing, ``reshape``, ``transpose``/``T`` and ``broadcast_to`` on a handle
    return a :class:`View`: a :class:`~aie.helpers.taplib.Layout` bound to this
    buffer, which ``fill()``/``drain()`` take as their single positional
    argument. ``A.reshape(M // m, m, K // k, k).transpose(0, 2, 1, 3)[i]`` is
    tile row ``i`` of ``A``, walked tile by tile; a ``range_`` induction
    variable as ``i`` makes the offset dispatch-time arithmetic.
    """

    def __init__(self, arr_type: type[np.ndarray]):
        """Construct a handle to a Runtime buffer.

        Args:
            arr_type (type[np.ndarray]): The type of the I/O data.
        """
        self._arr_type = arr_type
        self._op = None

    @property
    def shape(self) -> Sequence[int]:
        """Return the shape of the buffer."""
        return np_ndarray_type_get_shape(self._arr_type)

    @property
    def ndim(self) -> int:
        """Number of dimensions of the buffer."""
        return len(self.shape)

    @property
    def dtype(self) -> type[NpuDType]:
        """Return the per-element datatype of the buffer."""
        return np_ndarray_type_get_dtype(self._arr_type)

    @property
    def arr_type(self) -> type[np.ndarray]:
        """The tensor type of the buffer."""
        return self._arr_type

    @property
    def is_scalar(self) -> bool:
        """Whether this runtime argument is a scalar (no shape) rather than a tensor.

        Scalar runtime args (e.g. a runtime ``M``/``K``/``N``) are passed
        to the sequence body as their live SSA value, since they are used in
        arithmetic and ``range_``/``if_`` bounds, not as fill/drain buffers.
        """
        if get_origin(self._arr_type) is not np.ndarray:
            # Not an np.ndarray[...] generic alias at all (e.g. bare np.int32).
            return True
        return len(np_ndarray_type_get_shape(self._arr_type)) == 0

    def default_tap(self) -> TensorAccessPattern:
        """Return a default access pattern for a linear transfer of the buffer."""
        return Layout.full(self.shape).tap()

    # ------------------------------------------------------------------ views

    @property
    def layout(self) -> Layout:
        """The row-major walk over the whole buffer, unbound."""
        return Layout.full(self.shape)

    @property
    def view(self) -> View:
        """The whole buffer as a :class:`View`."""
        return View(self, self.layout)

    def __getitem__(self, key) -> View:
        """Return the :class:`View` a numpy-style slice of this buffer selects.

        ``flow.fill(a[0::2, 1::2, ...])`` transfers that region. This is
        metadata (offset, sizes and strides), not a numpy view or runtime
        values; indexing cannot read the buffer's contents.
        """
        return self.view[key]

    def reshape(self, *shape: Any) -> View:
        """Return the buffer regrouped to ``shape``; see :meth:`Layout.reshape`."""
        return self.view.reshape(*shape)

    def transpose(self, *axes: Any) -> View:
        """Return the buffer with its dimensions reordered; see :meth:`Layout.transpose`."""
        return self.view.transpose(*axes)

    @property
    def T(self) -> View:  # noqa: N802  (NumPy name)
        """``transpose()``: the dimensions reversed."""
        return self.view.T

    def broadcast_to(self, shape: Sequence[Any]) -> View:
        """Return the buffer walked again along new dimensions; see :meth:`Layout.broadcast_to`."""
        return self.view.broadcast_to(shape)

    def repeat(self, count: Any) -> View:
        """Return the whole buffer walked ``count`` times; see :meth:`Layout.repeat`."""
        return self.view.repeat(count)

    def tile(self, tile_dims: Sequence[Any]) -> BoundGrid:
        """Return the buffer as a grid of tiles; see :meth:`Layout.tile`."""
        return self.view.tile(tile_dims)

    def partition(self, parts: Any, dim: int = -1) -> BoundGrid:
        """Return the buffer cut into equal chunks; see :meth:`Layout.partition`."""
        return self.view.partition(parts, dim)

    @property
    def op(self) -> MemRefValue:
        if self._op is None:
            raise ValueError("Cannot get operation for RuntimeData before it is set.")
        return self._op

    @op.setter
    def op(self, op: MemRefValue):
        if self._op:
            raise ValueError("Cannot set operation for RuntimeData more than once.")
        self._op = op


def _rebind(data: RuntimeData, result: Any) -> Any:
    """Bind a layout-algebra result to ``data``; other results pass through."""
    if isinstance(result, Layout):
        return View(data, result)
    if isinstance(result, TileGrid):
        return BoundGrid(data, result)
    return result


class View:
    """A :class:`~aie.helpers.taplib.Layout` bound to the :class:`RuntimeData` it walks.

    Every ``Layout`` operation is available and returns a ``View`` of the same
    buffer (a ``TileGrid`` result becomes a :class:`BoundGrid`), so NumPy
    spellings compose: ``A.reshape(M // m, m, K // k, k).transpose(0, 2, 1,
    3)[i]``, ``B.T``, ``C[iv * m:(iv + 1) * m]``. ``fill()``/``drain()`` take
    a ``View`` as the buffer argument and read the access pattern from it;
    ``.layout`` is the unbound ``Layout``, ``.tap()`` and ``.stream_dims()``
    the conversions.
    """

    __slots__ = ("_data", "_layout")

    def __init__(self, data: RuntimeData, layout: Layout):
        if not isinstance(data, RuntimeData):
            raise TypeError(f"a View binds a RuntimeData, got {data!r}")
        if not isinstance(layout, Layout):
            raise TypeError(f"a View binds a Layout, got {layout!r}")
        self._data = data
        self._layout = layout

    @property
    def data(self) -> RuntimeData:
        """The buffer this view walks."""
        return self._data

    @property
    def layout(self) -> Layout:
        """The walk, unbound."""
        return self._layout

    # NumPy-shaped access, spelled out for readers and tooling.

    @property
    def shape(self) -> tuple[Any, ...]:
        return self._layout.shape

    @property
    def ndim(self) -> int:
        return self._layout.ndim

    @property
    def sizes(self) -> list[Any]:
        return self._layout.sizes

    @property
    def strides(self) -> list[Any]:
        return self._layout.strides

    @property
    def offset(self) -> Any:
        return self._layout.offset

    @property
    def tensor_dims(self) -> list[Any]:
        return self._layout.tensor_dims

    @property
    def dtype(self) -> type[NpuDType]:
        return self._data.dtype

    def __getitem__(self, key: Any) -> View:
        return View(self._data, self._layout[key])

    def reshape(self, *shape: Any) -> View:
        return View(self._data, self._layout.reshape(*shape))

    def transpose(self, *axes: Any) -> View:
        return View(self._data, self._layout.transpose(*axes))

    @property
    def T(self) -> View:  # noqa: N802  (NumPy name)
        return View(self._data, self._layout.T)

    def broadcast_to(self, shape: Sequence[Any]) -> View:
        return View(self._data, self._layout.broadcast_to(shape))

    def repeat(self, count: Any) -> View:
        return View(self._data, self._layout.repeat(count))

    def tile(self, tile_dims: Sequence[Any]) -> BoundGrid:
        return BoundGrid(self._data, self._layout.tile(tile_dims))

    def partition(self, parts: Any, dim: int = -1) -> BoundGrid:
        return BoundGrid(self._data, self._layout.partition(parts, dim))

    def tap(self, ndims: int | None = 4) -> TensorAccessPattern:
        """Return the walk as a :class:`~aie.helpers.taplib.TensorAccessPattern`."""
        return self._layout.tap(ndims)

    def stream_dims(self) -> list[tuple[Any, Any]]:
        """``[(size, stride), ...]`` of the walk."""
        return self._layout.stream_dims()

    def __getattr__(self, name: str) -> Any:
        # Anything else the layout algebra offers (permute, split, merge,
        # coalesce, numel, is_symbolic, ...), with Layout results re-bound.
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._layout, name)
        if callable(attr):
            data = self._data

            def bound(*args: Any, **kwargs: Any) -> Any:
                return _rebind(data, attr(*args, **kwargs))

            bound.__name__ = name
            bound.__doc__ = attr.__doc__
            return bound
        return _rebind(self._data, attr)

    def __repr__(self) -> str:
        return f"View({self._data!r}, {self._layout!r})"


class BoundGrid:
    """A :class:`~aie.helpers.taplib.TileGrid` bound to a :class:`RuntimeData`.

    Indexing and iteration yield :class:`View` objects; ``group``, ``order``,
    ``permute_tile`` and ``repeat`` return a bound grid; ``inverse()`` a
    ``View``.
    """

    __slots__ = ("_data", "_grid")

    def __init__(self, data: RuntimeData, grid: TileGrid):
        self._data = data
        self._grid = grid

    @property
    def data(self) -> RuntimeData:
        return self._data

    @property
    def grid(self) -> TileGrid:
        """The tiling, unbound."""
        return self._grid

    def __getitem__(self, key: Any) -> View:
        return View(self._data, self._grid[key])

    def __iter__(self) -> Iterator[View]:
        for tile in self._grid:
            yield View(self._data, tile)

    def __len__(self) -> int:
        return len(self._grid)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._grid, name)
        if callable(attr):
            data = self._data

            def bound(*args: Any, **kwargs: Any) -> Any:
                return _rebind(data, attr(*args, **kwargs))

            bound.__name__ = name
            bound.__doc__ = attr.__doc__
            return bound
        return _rebind(self._data, attr)

    def __repr__(self) -> str:
        return f"BoundGrid({self._data!r}, {self._grid!r})"
