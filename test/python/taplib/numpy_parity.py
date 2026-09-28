# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""NumPy parity of the layout algebra.

``reshape``, ``transpose``/``T`` and ``broadcast_to`` on a ``Layout`` walk
the elements NumPy's operations of the same name walk on a real array, and
refuse what NumPy could only do with a copy. With those three, NumPy indexing
spells every tiling the ``TileGrid`` sugar builds, which is checked by
equality against ``tile``/``group``/``inverse``. The staged path is driven
with an expression-tree stand-in for a runtime scalar and evaluated against
the integer path.
"""

import itertools

import numpy as np
from aie.helpers.taplib import Layout
from numpy.lib.stride_tricks import as_strided
from util import construct_test

# RUN: %python %s | FileCheck %s


def visited(layout: Layout) -> np.ndarray:
    """Element indices a view visits, in order, from an as_strided oracle."""
    base = np.arange(int(np.prod(layout.tensor_dims)), dtype=np.int64)
    view = as_strided(
        base[layout.offset :],
        shape=tuple(layout.sizes),
        strides=tuple(s * base.itemsize for s in layout.strides),
        writeable=False,
    )
    return view.ravel()


def same_walk(layout: Layout, arr: np.ndarray) -> bool:
    return layout.sizes == list(arr.shape) and (visited(layout) == arr.ravel()).all()


# CHECK-LABEL: reshape_matches_numpy
@construct_test
def reshape_matches_numpy():
    M, K = 64, 32
    A = np.arange(M * K).reshape(M, K)
    L = Layout.full((M, K))
    shapes = [
        (M * K,),
        (-1,),
        (M // 8, 8, K),
        (M, K // 4, 4),
        (M // 8, 8, K // 4, 4),
        (2, -1, 4),
        (1, M, 1, K, 1),
        (M * K // 16, 16),
    ]
    for shape in shapes:
        assert same_walk(L.reshape(*shape), A.reshape(*shape)), shape
        assert same_walk(L.reshape(shape), A.reshape(shape)), shape
    # Reshape of a view that is already strided: merge where contiguous, split
    # anywhere, and size-1 dimensions never get in the way.
    S = L[8:40, ::2]  # sizes (32, 16), strides (32, 2)
    a = A[8:40, ::2]
    for shape in [(4, 8, 16), (32, 4, 4), (4, 8, 4, 4), (1, 32, 16), (-1,)]:
        assert same_walk(S.reshape(*shape), a.reshape(*shape)), shape
    # Across a repeat (stride 0) the outer dimension may be regrouped alone.
    R = L.repeat(3)
    assert same_walk(R.reshape(3, M * K), np.broadcast_to(A, (3, M, K)).reshape(3, -1))
    assert same_walk(
        R.reshape(3, M // 8, 8 * K),
        np.broadcast_to(A, (3, M, K)).reshape(3, M // 8, -1),
    )
    print("reshape agrees with numpy on", len(shapes) + 5 + 2, "shapes")
    # CHECK: reshape agrees with numpy on 15 shapes


# CHECK-LABEL: reshape_refuses_copies
@construct_test
def reshape_refuses_copies():
    L = Layout.full((64, 32))
    refused = 0
    for bad in [
        lambda: L.T.reshape(-1),  # column-major walk is not contiguous
        lambda: L[::2, :].reshape(-1),  # every other row: rows are not adjacent
        lambda: L.repeat(2).reshape(-1),  # repeat is not a merge
        lambda: L.reshape(7, -1),  # 2048 % 7 != 0
        lambda: L.reshape(-1, -1),
        lambda: L.reshape(64, 33),
        lambda: L.reshape(),
        lambda: L.reshape(0, -1),
    ]:
        try:
            bad()
        except ValueError:
            refused += 1
    print("refused", refused)
    # CHECK: refused 8


# CHECK-LABEL: transpose_and_broadcast_match_numpy
@construct_test
def transpose_and_broadcast_match_numpy():
    A = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    L = Layout.full((2, 3, 4, 5))
    assert same_walk(L.T, A.T)
    assert same_walk(L.transpose(), A.transpose())
    n = 0
    for axes in itertools.permutations(range(4)):
        assert same_walk(L.transpose(*axes), A.transpose(*axes)), axes
        assert same_walk(L.transpose(axes), A.transpose(axes)), axes
        n += 1
    B = A[:, 0, :, :]  # (2, 4, 5)
    S = L[:, 0, :, :]
    assert same_walk(S.broadcast_to((3, 2, 4, 5)), np.broadcast_to(B, (3, 2, 4, 5)))
    assert same_walk(
        S[:, None].broadcast_to((2, 6, 4, 5)), np.broadcast_to(B[:, None], (2, 6, 4, 5))
    )
    assert same_walk(S.broadcast_to((2, 4, 5)), B)
    assert S.broadcast_to((7, 2, 4, 5)) == S.repeat(7)
    for bad in [(4, 5), (2, 4, 6), (2, 3, 5)]:
        try:
            S.broadcast_to(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    print("transposes checked", n)
    # CHECK: transposes checked 24


# CHECK-LABEL: numpy_spells_the_tile_grid
@construct_test
def numpy_spells_the_tile_grid():
    M, K, m, k = 96, 64, 16, 8
    L = Layout.full((M, K))
    four = L.reshape(M // m, m, K // k, k)
    grid = L.tile((m, k))
    for i, j in itertools.product(range(M // m), range(K // k)):
        assert four[i, :, j, :] == grid.at(i, j)
    tiles = four.transpose(0, 2, 1, 3)  # (M/m, K/k, m, k): tile by tile
    for i in range(M // m):
        assert tiles[i] == grid.group((1, K // k))[i], i
    # The two grid axes are not adjacent in memory, so merging them is a copy.
    try:
        tiles.reshape(-1, m, k)
        raise AssertionError("merged non-contiguous grid axes")
    except ValueError:
        pass
    # Column order over the grid is the same transpose on the grid axes.
    col = four.transpose(2, 0, 1, 3)  # (K/k, M/m, m, k)
    for j in range(K // k):
        assert col[j] == grid.group((M // m, 1), col_major=True).order("col")[j], j
    # Reading a tile-blocked buffer back in row-major order: the inverse walk.
    blocked = Layout.full((M // m * (K // k), m, k))  # tiles stored one after another
    inv = blocked.reshape(M // m, K // k, m, k).transpose(0, 2, 1, 3)
    stored = np.arange(M * K).reshape(M // m, K // k, m, k)
    assert same_walk(inv, stored.transpose(0, 2, 1, 3))
    assert inv.coalesce().stream_dims() == grid.inverse().coalesce().stream_dims()
    print("numpy spellings equal the grid sugar")
    # CHECK: numpy spellings equal the grid sugar


# ----------------------------------------------------------------- staged path

REQUIRED: list = []


class Sym:
    """A recorded integer expression over named runtime scalars."""

    __aie_symbolic__ = True
    __slots__ = ("op", "args")

    def __init__(self, op, *args):
        self.op = op
        self.args = args

    @staticmethod
    def var(name):
        return Sym("var", name)

    @staticmethod
    def _lift(v):
        return v if isinstance(v, Sym) else Sym("const", int(v))

    def _bin(self, op, other, rev=False):
        other = self._lift(other)
        return Sym(op, other, self) if rev else Sym(op, self, other)

    def __add__(self, o):
        return self._bin("add", o)

    def __radd__(self, o):
        return self._bin("add", o, rev=True)

    def __sub__(self, o):
        return self._bin("sub", o)

    def __rsub__(self, o):
        return self._bin("sub", o, rev=True)

    def __mul__(self, o):
        return self._bin("mul", o)

    def __rmul__(self, o):
        return self._bin("mul", o, rev=True)

    def __floordiv__(self, o):
        return self._bin("div", o)

    def __rfloordiv__(self, o):
        return self._bin("div", o, rev=True)

    def __mod__(self, o):
        return self._bin("rem", o)

    def __rmod__(self, o):
        return self._bin("rem", o, rev=True)

    def __lt__(self, o):
        return self._bin("lt", o)

    def __le__(self, o):
        return self._bin("le", o)

    def __gt__(self, o):
        return self._bin("gt", o)

    def __ge__(self, o):
        return self._bin("ge", o)

    def __eq__(self, o):
        return self._bin("eq", o)

    def __ne__(self, o):
        return self._bin("ne", o)

    __hash__ = object.__hash__

    def _select(self, a, b):
        return Sym("select", self, self._lift(a), self._lift(b))

    def _require(self, message):
        REQUIRED.append((self, message))

    def __bool__(self):
        raise TypeError("a Sym has no truth value at generation time")

    __index__ = __int__ = __bool__

    def eval(self, env):
        if self.op == "var":
            return env[self.args[0]]
        if self.op == "const":
            return self.args[0]
        if self.op == "select":
            c, a, b = (x.eval(env) for x in self.args)
            return a if c else b
        a, b = (x.eval(env) for x in self.args)
        return {
            "add": lambda: a + b,
            "sub": lambda: a - b,
            "mul": lambda: a * b,
            "div": lambda: a // b,
            "rem": lambda: a % b,
            "lt": lambda: a < b,
            "le": lambda: a <= b,
            "gt": lambda: a > b,
            "ge": lambda: a >= b,
            "eq": lambda: a == b,
            "ne": lambda: a != b,
        }[self.op]()


def ev(v, env):
    return v.eval(env) if isinstance(v, Sym) else int(v)


def concrete(layout, env) -> Layout:
    return Layout(
        [ev(d, env) for d in layout.tensor_dims],
        ev(layout.offset, env),
        [ev(s, env) for s in layout.sizes],
        [ev(s, env) for s in layout.strides],
    )


# CHECK-LABEL: staged_reshape_matches_integer_path
@construct_test
def staged_reshape_matches_integer_path():
    Mv, K, m, k = Sym.var("M"), 64, 16, 8
    checked = 0
    # The staged (outermost) dimension is split; the concrete one matched.
    for shape in [
        (Mv // m, m, K // k, k),
        (-1, m, K // k, k),
        (-1, m, K),
        (-1, K),
    ]:
        del REQUIRED[:]
        staged = Layout.full((Mv, K)).reshape(*shape)
        for M in (16, 64, 160):
            env = {"M": M}
            want = Layout.full((M, K)).reshape(
                *[-1 if isinstance(x, int) and x == -1 else ev(x, env) for x in shape]
            )
            assert concrete(staged, env) == want, (shape, M)
            assert all(c.eval(env) for c, _ in REQUIRED), shape
            checked += 1
        # M = 20 is not a multiple of m: the split's guard must fail.
        if any(isinstance(x, int) and x == m for x in shape):
            assert not all(c.eval({"M": 20}) for c, _ in REQUIRED), shape
    # Then a NumPy tile selection with a staged row index, against the grid.
    iv = Sym.var("i")
    tiles = Layout.full((Mv, K)).reshape(-1, m, K // k, k).transpose(0, 2, 1, 3)
    for M, i in itertools.product((32, 64), (0, 1)):
        env = {"M": M, "i": i}
        want = Layout.full((M, K)).tile((m, k)).group((1, K // k))[i]
        assert concrete(tiles[iv], env) == want, (M, i)
        checked += 1
    # A runtime inner dimension stages the outer strides, not the outer size:
    # the outer dimension still splits, and never merges across the staged
    # stride.
    Kv = Sym.var("K")
    for shape in [(8, 8, -1), (8, 8, Kv), (64, Kv // 8, 8)]:
        del REQUIRED[:]
        staged = Layout.full((64, Kv)).reshape(*shape)
        for K_ in (16, 64):
            env = {"K": K_}
            want = Layout.full((64, K_)).reshape(
                *[-1 if isinstance(x, int) and x == -1 else ev(x, env) for x in shape]
            )
            assert concrete(staged, env) == want, (shape, K_)
            assert all(c.eval(env) for c, _ in REQUIRED), shape
            checked += 1
    # A static buffer split into a staged number of chunks, or by a staged
    # chunk size: the shape entries are staged, the view is concrete.
    nv = Sym.var("n")
    for shape in [(nv, -1), (-1, nv), (nv, 4, -1), (2, nv, -1, 4)]:
        del REQUIRED[:]
        staged = Layout.full((256,)).reshape(*shape)
        for n_ in (2, 8):
            env = {"n": n_}
            want = Layout.full((256,)).reshape(
                *[-1 if isinstance(x, int) and x == -1 else ev(x, env) for x in shape]
            )
            assert concrete(staged, env) == want, (shape, n_)
            assert all(c.eval(env) for c, _ in REQUIRED), shape
            checked += 1
        assert not all(c.eval({"n": 3}) for c, _ in REQUIRED), shape
    # Merging a staged dimension is structural and refused; two staged dims
    # too; the concrete side must still match exactly.
    for bad, err in [
        (lambda: Layout.full((Mv, K)).transpose().reshape(-1), ValueError),
        (lambda: Layout.full((Mv, Sym.var("N"))).reshape(-1), TypeError),
        (lambda: Layout.full((Mv, K)).reshape(-1, 8, 4), ValueError),
        (lambda: Layout.full((64, Kv)).reshape(-1), ValueError),
        (lambda: Layout.full((64, Kv)).reshape(4, -1), ValueError),
        # -1 never stands for a concrete dimension next to a staged one.
        (lambda: Layout.full((Mv, K)).reshape(Mv // m, m, -1), ValueError),
        (lambda: Layout.full((Mv, K)).reshape(Mv, -1), ValueError),
        (lambda: Layout.full((64, Kv)).reshape(-1, 8, Kv), ValueError),
    ]:
        try:
            bad()
        except err:
            continue
        raise AssertionError(f"expected {err.__name__}")
    print("staged reshape checked", checked)
    # CHECK: staged reshape checked 30
