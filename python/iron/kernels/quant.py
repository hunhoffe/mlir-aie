# kernels/quant.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Packed quantization kernel factories and byte-exact host references."""

import math
from functools import partial

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils import bfp
from aie.utils.compile.jit.markers import In, InOut, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    Param,
    Trace,
    _arch_traits,
    _kernel_source,
    _make_extern,
    _on_object,
)
from .linalg import _ZeroInitializedKernel


def _geometry(m_tile, k_tile, group, ct_k, s, t):
    geometry = dict(m_tile=m_tile, k_tile=k_tile, group=group, ct_k=ct_k, s=s, t=t)
    for name, value in geometry.items():
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"q4nx_dequant: {name} must be a positive integer")
    if s != 8 or t != 8:
        raise ValueError("q4nx_dequant: s and t must both be 8")
    if m_tile % 16:
        raise ValueError("q4nx_dequant: m_tile must be a multiple of 16")
    if ct_k % s:
        raise ValueError("q4nx_dequant: ct_k must be a multiple of s")
    if group % s:
        raise ValueError("q4nx_dequant: group must be a multiple of s")
    if k_tile % ct_k:
        raise ValueError("q4nx_dequant: k_tile must be a multiple of ct_k")
    if k_tile % group:
        raise ValueError("q4nx_dequant: k_tile must be a multiple of group")
    geometry = {name: int(value) for name, value in geometry.items()}
    if geometry["m_tile"] * geometry["k_tile"] * 9 // 8 > np.iinfo(np.int32).max:
        raise ValueError("q4nx_dequant: geometry exceeds the kernel's int offsets")
    return geometry


def _bf16_floor(x):
    """Narrow float32 toward minus infinity, not numpy's bf16 nearest-even."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    upper = bits >> 16
    upper += ((bits >> 31 != 0) & (bits & 0xFFFF != 0)).astype(np.uint32)
    return (upper << 16).view(np.float32)


def q4nx_unpack(
    payload, *, m_tile=32, k_tile=256, group=32, ct_k=128, s=8, t=8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack packed q4nx to its codes, scales and minima.

    Input has shape ``(..., m_tile*k_tile//2 + 4*m_tile*k_tile//group)``.
    It contains little-endian bf16 scales, then bf16 minima, both indexed
    ``[k_group, n]``, followed by unsigned nibbles (low nibble first) indexed
    ``[n//16, k, n%16]``. The weight at ``[..., k, n]`` is
    ``mins[..., k // group, n] + scales[..., k // group, n] * codes[..., k, n]``.
    Finite scales and minima are required.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(codes, scales, mins)``,
            with the same leading dimensions as ``payload``. ``codes`` is
            ``uint8`` with shape ``(..., k_tile, m_tile)``; ``scales`` and
            ``mins`` are ``float32`` with shape ``(..., k_tile // group,
            m_tile)``.
    """
    geometry = _geometry(m_tile, k_tile, group, ct_k, s, t)
    m_tile, k_tile, group, ct_k, s, t = geometry.values()
    scales_count = m_tile * k_tile // group
    input_bytes = m_tile * k_tile // 2 + 4 * scales_count
    payload = np.asarray(payload, dtype=np.uint8)
    if payload.ndim == 0 or payload.shape[-1] != input_bytes:
        raise ValueError(f"q4nx_unpack: expected {input_bytes} bytes per tile")
    lead = payload.shape[:-1]
    data = payload.reshape(-1, input_bytes)
    if not len(data):
        codes = np.empty((*lead, k_tile, m_tile), dtype=np.uint8)
        params = np.empty((*lead, k_tile // group, m_tile), dtype=np.float32)
        return codes, params, params.copy()
    params = np.ascontiguousarray(data[:, : 4 * scales_count]).view("<u2")
    params = (params.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(params).all():
        raise ValueError("q4nx_unpack: scales and minima must be finite")
    scales, mins = params.reshape(-1, 2, k_tile // group, m_tile).transpose(1, 0, 2, 3)
    packed = data[:, 4 * scales_count :]
    q = np.empty((len(data), m_tile * k_tile), dtype=np.uint8)
    q[:, 0::2], q[:, 1::2] = packed & 15, packed >> 4
    q = q.reshape(-1, m_tile // 16, k_tile, 16).transpose(0, 2, 1, 3)
    q = q.reshape(*lead, k_tile, m_tile)
    scales = scales.reshape(*lead, k_tile // group, m_tile)
    mins = mins.reshape(*lead, k_tile // group, m_tile)
    return q, scales, mins


def q4nx_dequant_ref(
    payload, *, m_tile=32, k_tile=256, group=32, ct_k=128, s=8, t=8, rounding="floor"
):
    """Dequantize packed q4nx to the kernel's bfp16ebs8 output bytes.

    See ``q4nx_unpack`` for the packed input layout.

    Accumulator results narrow to bf16, then to bfp16ebs8, both in the
    ``rounding`` mode, ``floor`` or ``conv_even``. Output is uint8 with the
    same leading dimensions, indexed ``[k//ct_k, n//8, (k%ct_k)//8, n%8, 9]``:
    each nine-byte block is an exponent followed by eight signed mantissas for
    consecutive k values. Finite dequantized bf16 results are required.
    """
    geometry = _geometry(m_tile, k_tile, group, ct_k, s, t)
    m_tile, k_tile, group, ct_k, s, t = geometry.values()
    _check_rounding(rounding)
    scales_count = m_tile * k_tile // group
    input_bytes = m_tile * k_tile // 2 + 4 * scales_count
    payload = np.asarray(payload, dtype=np.uint8)
    if payload.ndim == 0 or payload.shape[-1] != input_bytes:
        raise ValueError(f"q4nx_dequant_ref: expected {input_bytes} bytes per tile")
    lead = payload.shape[:-1]
    if not math.prod(lead):
        return np.empty((*lead, m_tile * k_tile * 9 // 8), dtype=np.uint8)
    q, scales, mins = q4nx_unpack(payload, **geometry)
    q = q.reshape(-1, k_tile, m_tile)
    scales = scales.reshape(-1, k_tile // group, m_tile)
    mins = mins.reshape(-1, k_tile // group, m_tile)
    # A bf16 scale times a four-bit integer is exact in float32. Float64
    # models the fused addition before its single float32 accumulator rounding.
    with np.errstate(over="ignore", invalid="ignore"):
        values = (
            np.repeat(scales, group, axis=1).astype(np.float64) * q
            + np.repeat(mins, group, axis=1).astype(np.float64)
        ).astype(np.float32)
        if rounding == "floor":
            values = _bf16_floor(values)
        else:
            values = values.astype(bfloat16).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("q4nx_dequant_ref: dequantized bf16 values must be finite")
    values = values.reshape(-1, k_tile // ct_k, ct_k // s, s, m_tile // t, t)
    values = values.transpose(0, 1, 4, 2, 5, 3)
    values = values.reshape(math.prod(lead), -1)
    encoded = (
        bfp.encode(values) if rounding == "floor" else bfp.encode_core_even(values)
    )
    return encoded.reshape(*lead, m_tile * k_tile * 9 // 8)


def _q4nx_sample(rng, calls, *, m_tile, k_tile, group, **_):
    count = m_tile * k_tile // group
    scales = rng.uniform(-2, 2, (calls, count)).astype(bfloat16)
    mins = rng.uniform(-8, 8, (calls, count)).astype(bfloat16)
    params = np.concatenate([scales, mins], axis=1)
    params = params.view(np.uint16).astype("<u2").view(np.uint8)
    packed = rng.integers(0, 256, (calls, m_tile * k_tile // 2), dtype=np.uint8)
    return [np.concatenate([params, packed], axis=1)]


def _check_rounding(rounding):
    if rounding not in ("floor", "conv_even"):
        raise ValueError(
            f"q4nx_dequant: rounding must be 'floor' or 'conv_even', got {rounding!r}"
        )


def q4nx_dequant(
    *, m_tile=32, k_tile=256, group=32, ct_k=128, s=8, t=8, rounding="floor"
) -> ExternalFunction:
    """AIE2P-only q4nx dequantization into GEMM B-operand BFP storage.

    ``m_tile`` counts n rows (a multiple of 16); ``k_tile`` must be divisible
    by both ``group`` and ``ct_k``. The latter two are positive multiples of
    eight; ``s`` and ``t`` must be eight. Groups need not divide k slices.
    See ``q4nx_dequant_ref`` for the packed input and output layouts.

    Both arguments are uint8 byte buffers, including the bfp16ebs8 output,
    so the generic harness compares the complete encoded result byte for
    byte rather than decoding or quantizing it again. The kernel saves,
    selects and restores its rounding mode itself; no setup kernel is needed.
    ``rounding`` is that mode for both narrowings: ``floor``, which matches
    FastFlowLM's packing, or ``conv_even``.
    """
    geometry = _geometry(m_tile, k_tile, group, ct_k, s, t)
    m_tile, k_tile, group, ct_k, s, t = geometry.values()
    _check_rounding(rounding)
    if not _arch_traits().bfp16:
        raise NotImplementedError("q4nx_dequant() is only available on aie2p.")
    input_bytes = m_tile * k_tile // 2 + 4 * m_tile * k_tile // group
    output_bytes = m_tile * k_tile * 9 // 8
    return _make_extern(
        "q4nx_dequant_bfp",
        _kernel_source("quant/q4nx_dequant.cc"),
        [
            np.ndarray[(input_bytes,), np.dtype[np.uint8]],
            np.ndarray[(output_bytes,), np.dtype[np.uint8]],
        ],
        # The inner loop is one long latency chain, so it is allowed five
        # pipeline stages instead of the default three.
        compile_flags=[
            f"-DQ4NX_{name.upper()}={value}" for name, value in geometry.items()
        ]
        + ["-mllvm", "--aie-pipeliner-max-stagecount=5"]
        + (["-DROUND_CONV_EVEN"] if rounding == "conv_even" else []),
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out),
            reference=partial(q4nx_dequant_ref, **geometry, rounding=rounding),
            sample=partial(_q4nx_sample, **geometry),
            tolerance=Tolerance.exact(
                note=f"{rounding} bf16 and bfp16ebs8 narrowings, byte for byte"
            ),
            ops_per_call=2 * m_tile * k_tile,
            acc_dtype=np.float32,
            reduction=1,
        ),
    )


# Columns sharing one scale in a row of ternary weights.
TERNARY_GROUP = 128


def _ternary_geometry(name, m_tile, k_tile) -> int:
    """The bytes of one ``(m_tile, k_tile)`` ternary block."""
    if m_tile <= 0 or m_tile % 16:
        raise ValueError(f"{name}: m_tile must be a positive multiple of 16")
    if k_tile <= 0 or k_tile % TERNARY_GROUP:
        raise ValueError(f"{name}: k_tile must be a positive multiple of 128")
    return m_tile * k_tile // 4 + 2 * m_tile * k_tile // TERNARY_GROUP


def ternary_dequant_ref(w, *, m_tile=32, k_tile=256) -> np.ndarray:
    """Ternary blocks as their float32 weights, ``(..., m_tile, k_tile)``.

    A block is each row's 2-bit codes, 32 bytes per 128 columns, byte ``i``
    holding columns ``i``, ``i + 32``, ``i + 64`` and ``i + 96`` at bits 0,
    2, 4 and 6; then the little-endian bf16 scales, one per row per 128
    columns, row by row. Code 1 is the scale, 2 its negative, 0 and 3 zero.
    This is our own layout, which the host repacks a checkpoint into; it is
    not Bonsai's PTQ1_0.
    """
    nbytes = _ternary_geometry("ternary_dequant_ref", m_tile, k_tile)
    w = np.asarray(w).view(np.uint8)
    if w.ndim == 0 or w.shape[-1] != nbytes:
        raise ValueError(f"ternary_dequant_ref: expected {nbytes} bytes per block")
    lead, groups = w.shape[:-1], k_tile // TERNARY_GROUP
    data = w.reshape(-1, nbytes)
    codes = data[:, : m_tile * k_tile // 4].reshape(-1, m_tile, groups, 1, 32)
    codes = (codes >> np.arange(0, 8, 2, dtype=np.uint8)[:, None]) & 3
    sign = (codes == 1).astype(np.float32) - (codes == 2)
    scales = np.ascontiguousarray(data[:, m_tile * k_tile // 4 :]).view("<u2")
    scales = (scales.astype(np.uint32) << 16).view(np.float32)
    weights = sign * scales.reshape(-1, m_tile, groups, 1, 1)
    return weights.reshape(*lead, m_tile, k_tile)


def ternary_gemv_ref(w, x, *, m_tile=32, k_tile=256):
    """Numpy reference for one ``ternary_gemv_block`` call: the float32
    accumulator, zero before the call, plus ``w`` times slice 0 of ``x``.
    """
    weights = ternary_dequant_ref(w, m_tile=m_tile, k_tile=k_tile).astype(np.float64)
    x = np.asarray(x, np.float32).reshape(-1)[:k_tile].astype(np.float64)
    return (weights @ x).astype(np.float32)


def _ternary_sample(rng, calls, *, dim, m_tile, k_tile):
    # Integer x and power-of-two scales keep every sum exact in float32.
    codes = rng.integers(0, 256, (calls, m_tile * k_tile // 4), dtype=np.uint8)
    count = m_tile * k_tile // TERNARY_GROUP
    scales = rng.choice([-1.0, 1.0], (calls, count)) * 2.0 ** -rng.integers(
        0, 5, (calls, count)
    )
    scales = scales.astype(bfloat16).view(np.uint16).astype("<u2").view(np.uint8)
    x = rng.integers(-4, 5, dim).astype(bfloat16)
    return [np.concatenate([codes, scales], axis=1), x]


class _TernaryGemvKernel(_ZeroInitializedKernel):
    ternary_gemv_zero: ExternalFunction
    ternary_gemv_store: ExternalFunction


def ternary_gemv(
    *, dim: int = 512, m_tile: int = 32, k_tile: int = 256
) -> _TernaryGemvKernel:
    """``y = w x`` for one bf16 row ``x`` of ``dim`` values and ternary
    weights ``w``, from ``quant/ternary.cc``; see ``ternary_dequant_ref`` for a
    block's layout.

    The returned kernel is ``ternary_gemv_block(w, x, y_acc, k)``, which adds
    one block times slice ``k`` of ``x`` into a float32 accumulator: each
    128-column group's signed copies of ``x`` summed in float32, times its
    scale. The other entry points, attributes of the kernel:

    - ``ternary_gemv_zero(y_acc)``: before a tile's k loop.
    - ``ternary_gemv_store(y, y_acc)``: after it, ``y_acc`` narrowed to bf16
      in the core's rounding mode (floor on a fresh core).
    """
    nbytes = _ternary_geometry("ternary_gemv", m_tile, k_tile)
    if dim <= 0 or dim % k_tile:
        raise ValueError("ternary_gemv: dim must be a positive multiple of k_tile")
    y_acc = np.ndarray[(m_tile,), np.dtype[np.float32]]
    fn = _make_extern(
        "ternary_gemv_block",
        _kernel_source("quant/ternary.cc"),
        [
            np.ndarray[(nbytes,), np.dtype[np.uint8]],
            np.ndarray[(dim,), np.dtype[bfloat16]],
            y_acc,
            np.int32,
        ],
        compile_flags=[f"-DTERNARY_M_TILE={m_tile}", f"-DTERNARY_K_TILE={k_tile}"],
        cls=_TernaryGemvKernel,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Param, InOut, Param),
            parameter_bindings=((3, 0),),
            initializers=((2, lambda fn: fn.ternary_gemv_zero),),
            reference=partial(ternary_gemv_ref, m_tile=m_tile, k_tile=k_tile),
            sample=partial(_ternary_sample, dim=dim, m_tile=m_tile, k_tile=k_tile),
            tolerance=Tolerance.exact(
                note="the sample keeps every sum exact in float32"
            ),
            ops_per_call=2 * m_tile * k_tile,
            acc_dtype=np.float32,
            reduction=k_tile,
        ),
    )
    fn.ternary_gemv_zero = _on_object(
        fn,
        "ternary_gemv_zero",
        [y_acc],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(Out,),
            reference=lambda: np.zeros((1, m_tile), np.float32),
            tolerance=Tolerance.exact(note="zero fill"),
            ops_per_call=0,
        ),
    )
    fn.ternary_gemv_store = _on_object(
        fn,
        "ternary_gemv_store",
        [np.ndarray[(m_tile,), np.dtype[bfloat16]], y_acc],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(Out, In),
            reference=lambda y_acc: _bf16_floor(y_acc).astype(bfloat16),
            tolerance=Tolerance.exact(note="the floor rounding a fresh core uses"),
            ops_per_call=0,
        ),
    )
    return fn


def ternary_dequant(*, m_tile: int = 32, k_tile: int = 256) -> ExternalFunction:
    """One ternary block as its bf16 weights, ``(m_tile, k_tile)`` row-major,
    from ``quant/ternary.cc``; each is a signed copy of its scale, exact. See
    ``ternary_dequant_ref`` for the block's layout.
    """
    nbytes = _ternary_geometry("ternary_dequant", m_tile, k_tile)
    return _make_extern(
        "ternary_dequant",
        _kernel_source("quant/ternary.cc"),
        [
            np.ndarray[(nbytes,), np.dtype[np.uint8]],
            np.ndarray[(m_tile * k_tile,), np.dtype[bfloat16]],
        ],
        compile_flags=[f"-DTERNARY_M_TILE={m_tile}", f"-DTERNARY_K_TILE={k_tile}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out),
            reference=lambda w: ternary_dequant_ref(w, m_tile=m_tile, k_tile=k_tile)
            .reshape(len(w), -1)
            .astype(bfloat16),
            sample=lambda rng, calls: _ternary_sample(
                rng, calls, dim=k_tile, m_tile=m_tile, k_tile=k_tile
            )[:1],
            tolerance=Tolerance.exact(note="a signed copy of a bf16 scale"),
            ops_per_call=m_tile * k_tile,
        ),
    )
