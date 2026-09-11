# test_kernels_e2e.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest -m "not extensive and not core_state" %s
# RUN: %run_on_npu2_xrt% %pytest -m "not extensive and not core_state" %s
# RUN: %run_on_npu2_hrx% %pytest -m "not extensive and not core_state" %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings

"""Device tests for the IRON kernel library, driven by ``kernel_cases.CASES``.

``test/python/test_kernel_contracts.py`` proves every contract matches its
factory and lowers to MLIR on the host. This file is the tier that needs a
device: each case runs through ``aie.utils.kernel_harness`` and is judged
under the tolerance its kernel declares. It catches what types cannot -- a
wrong exported symbol, a wrong compile flag, a DMA-alignment bug, a
reference that disagrees with the C++.

Three tiers share one table (``kernel_cases.py``):

* ``test_kernel`` runs the ``smoke`` cases on random data: one representative
  shape per kernel, on every pull request.
* ``test_kernel_extensive`` (marker ``extensive``, deselected by the RUN
  lines above) runs every case under every edge-data case its contract
  admits, for ``--seeds`` random seeds. The nightly benchmark workflow runs
  it as the correctness gate before anything is timed.
* ``test_kernel_core_state`` (marker ``core_state``, also deselected above)
  is the dirty-state sweep: one case per compiled kernel, run with the core's
  rounding and saturation registers preset to every mode as if a previous
  kernel had left them there. The output must be bit-identical to the run
  from the boot state unless the contract names the mode the harness should
  set first, and a probe checks that the kernel ran in the state its
  contract names and left what its ``leaves_*`` claims say. The nightly
  core-state workflow runs it on both NPUs.

Cases whose kernels exist only for one NPU generation carry
``supported_devices`` (see ``conftest.py``), so they skip elsewhere.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.kernels import (
    BOOT_ROUNDING,
    BOOT_SATURATION,
    ROUNDING_MODES,
    SATURATION_MODES,
)
from aie.iron.kernels._common import _detect_arch
from aie.utils import kernel_harness as kh
from aie.utils.kernel_harness import CoreState
from aie.utils.kernel_harness.cases import distinct_kernels, inputs_for
from kernel_cases import CASES
from ml_dtypes import bfloat16


def _param(case):
    marks = [pytest.mark.supported_devices(*case.devices)] if case.devices else []
    return pytest.param(case, marks=marks, id=case.name)


def _run(case, data_case: str, seed: int):
    fn = case.fn()
    inputs = inputs_for(case, data_case, np.random.default_rng(1000 + seed))
    design = kh.design(
        getattr(kernels, case.factory),
        **case.harness_opts(),
        params=kh.param_values(fn, inputs),
        **case.kwargs,
    )
    ref = kh.expected(fn, inputs, scalars=case.scalars)
    out_n = kh.output_size(fn, calls=case.calls, shape=case.shape)
    out_dt = kh.output_dtype(fn, ref.dtype)
    # The output is poisoned so a kernel that writes nothing cannot pass.
    got = kh.run(design, inputs, out_n, out_dt, poison=True, fn=fn)
    verdict = kh.judge(fn, got, ref, calls=case.calls)
    assert verdict, f"{case.name} [{data_case}, seed {seed}]: {verdict.detail}"


@pytest.mark.parametrize("case", [_param(c) for c in CASES if c.smoke])
def test_kernel(case):
    _run(case, "random", 0)


def pytest_generate_tests(metafunc):
    if {"case", "data_case", "seed"} <= set(metafunc.fixturenames):
        seeds = metafunc.config.getoption("--seeds")
        params = [
            pytest.param(
                c,
                dc,
                seed,
                marks=[pytest.mark.supported_devices(*c.devices)] if c.devices else [],
                id=f"{c.name}/{dc}/s{seed}",
            )
            for c in CASES
            for dc in c.data_policy()
            for seed in range(seeds if dc == "random" else 1)
        ]
        metafunc.parametrize("case,data_case,seed", params)


@pytest.mark.extensive
def test_kernel_extensive(case, data_case, seed):
    _run(case, data_case, seed)


def test_case_names_are_unique():
    names = [c.name for c in CASES]
    assert len(names) == len(set(names)), "two cases share a series name"


# ---------------------------------------------------------------------------
# Dirty-state sweep: what a kernel does when the core is not freshly booted.
#
# Both mode registers are sticky per core. A one-Worker design always starts
# from the boot state, so a kernel that silently depends on the register
# passes every test above and still degrades a fused pipeline where another
# kernel ran first. Each compiled kernel is run once from the boot state
# (the baseline, also judged against its reference) and once per other mode
# preset on the core before the contract's own setters run.
# ---------------------------------------------------------------------------

CLEAN_STATE = CoreState(BOOT_ROUNDING, BOOT_SATURATION)
DIRTY_STATES = [
    CoreState(r, BOOT_SATURATION) for r in ROUNDING_MODES[2:] if r != BOOT_ROUNDING
] + [CoreState(BOOT_ROUNDING, s) for s in SATURATION_MODES[2:] if s != BOOT_SATURATION]

_clean_runs: dict[str, np.ndarray] = {}


def _run_in_state(case, preset: CoreState):
    """Build, run and probe ``case`` with the core preset to ``preset``."""
    fn = case.fn()
    inputs = inputs_for(case, "random", np.random.default_rng(1000))
    design = kh.design(
        getattr(kernels, case.factory),
        **case.harness_opts(),
        params=kh.param_values(fn, inputs),
        core_state=preset,
        **case.kwargs,
    )
    ref = kh.expected(fn, inputs, scalars=case.scalars)
    out_n = kh.output_size(fn, calls=case.calls, shape=case.shape)
    out_dt = kh.output_dtype(fn, ref.dtype)
    got, reading = kh.run_probed(design, inputs, out_n, out_dt, poison=True, fn=fn)
    state = kh.judge_state(fn, reading, preset)
    assert state, f"{case.name} [{preset}]: {state.detail}"
    return fn, got, ref


def _clean_output(case) -> np.ndarray:
    """Return the baseline output from the boot state, judged against the reference once."""
    if case.name not in _clean_runs:
        fn, got, ref = _run_in_state(case, CLEAN_STATE)
        verdict = kh.judge(fn, got, ref, calls=case.calls)
        assert verdict, f"{case.name} [{CLEAN_STATE}]: {verdict.detail}"
        _clean_runs[case.name] = got
    return _clean_runs[case.name]


def _state_id(preset: CoreState) -> str:
    return f"r={preset.rounding}/s={preset.saturation}"


@pytest.mark.core_state
@pytest.mark.parametrize("case", [_param(c) for c in distinct_kernels(CASES)])
@pytest.mark.parametrize("preset", DIRTY_STATES, ids=_state_id)
def test_kernel_core_state(case, preset):
    """Bit-identical output under every preset, and the probe agrees with the contract."""
    clean = _clean_output(case)
    fn, got, _ = _run_in_state(case, preset)
    verdict = kh.judge_dirty(fn, clean, got, preset)
    assert verdict, f"{case.name} [{_state_id(preset)}]: {verdict.detail}"


@pytest.mark.core_state
def test_core_boot_state():
    """A fresh core holds the boot state the contracts assume (floor, saturation off).

    Everything above presets the registers explicitly, so this is the one run
    that measures what the design finds without touching them. If it fails,
    the boot-state assumption in ``aie.iron.kernels`` (``BOOT_ROUNDING``,
    ``BOOT_SATURATION``) is wrong for this device, firmware or driver, and
    the trusted-base notes need an entry.
    """
    case = next(
        c for c in CASES if c.factory == "passthrough" and c.calls == 4 and not c.kwargs
    )
    fn = case.fn()
    inputs = inputs_for(case, "random", np.random.default_rng(1000))
    design = kh.design(
        getattr(kernels, case.factory),
        **case.harness_opts(),
        params=kh.param_values(fn, inputs),
        core_state=CoreState(None, None),
        **case.kwargs,
    )
    ref = kh.expected(fn, inputs, scalars=case.scalars)
    out_n = kh.output_size(fn, calls=case.calls, shape=case.shape)
    _, reading = kh.run_probed(
        design, inputs, out_n, kh.output_dtype(fn, ref.dtype), poison=True, fn=fn
    )
    assert reading.marker_ok, "the probe never wrote its tile"
    assert (
        reading.before == CLEAN_STATE
    ), f"a fresh core holds {reading.before}, not the assumed {CLEAN_STATE}"
    assert reading.after == reading.before, "passthrough must not touch the registers"


# ---------------------------------------------------------------------------
# mha: flash-attention toolkit — compile regression only.
#
# mha.cc is not a single kernel but a set of composable symbols (matmul_PV,
# partial_softmax, rescale_O, init_scale_buffer, …) that ``#include`` sibling
# ``softmax.cc`` + ``mm.cc``.  A full attention dataflow needs a bespoke
# multi-core design; that's out of scope here.  What this test pins is that
# mha.cc must COMPILE against mlir-aie's softmax.cc, which relies on
# ``partial_softmax_bf16`` / ``partial_softmax_alias_bf16`` being defined there.
# If those regress, mha.cc stops compiling and this test fails.
# ``init_scale_buffer`` is the simplest symbol to instantiate the translation
# unit.
# ---------------------------------------------------------------------------

# init_scale_buffer's buffer argument is mha's per-row scale buffer, which is
# dim_m elements wide, so the probe's ObjectFifo has to be that wide too.
_MHA_DIM = 64
_MHA_TILE = _MHA_DIM


@iron.jit
def _mha_compile_probe(
    a_in: In, b_out: Out, *, size: iron.CompileTime[int] = _MHA_TILE
):
    buf = np.ndarray[(_MHA_TILE,), np.dtype[bfloat16]]
    # kernels.mha compiles mha.cc once and binds its symbols; init_scale_buffer
    # is the simplest of them to instantiate the translation unit with.
    kern = kernels.mha(dim_m=_MHA_DIM, dim_k=_MHA_DIM, dim_n=_MHA_DIM).init_scale_buffer
    of_in = ObjectFifo(buf, name="mhi")
    of_out = ObjectFifo(buf, name="mho")

    def core(of_in, of_out, k):
        a = of_in.acquire(1)
        c = of_out.acquire(1)
        k(c, _MHA_TILE)  # init_scale_buffer writes its buffer arg
        for i in range_(_MHA_TILE):
            c[i] = a[i]
        of_in.release(1)
        of_out.release(1)

    w = Worker(core, fn_args=[of_in.cons(), of_out.prod(), kern])
    vec = np.ndarray[(size,), np.dtype[bfloat16]]

    def seq(a, b, ih, oh):
        ih.fill(a)
        oh.drain(b, wait=True)

    rt = Runtime(seq, [vec, vec, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def test_mha_compiles_against_mlir_aie_softmax():
    """mha.cc compiles + runs on aie2p (prereq: partial_softmax_* in softmax.cc).

    aie2p-only: mha.cc is an aie2p source and pulls in the aie2p mm/softmax
    micro-kernels.  Skip on any other arch rather than fail.
    """
    if _detect_arch() != "aie2p":
        pytest.skip("mha.cc is aie2p-only")
    a = iron.tensor(np.zeros(_MHA_TILE, dtype=bfloat16), dtype=bfloat16, device="npu")
    b = iron.zeros(_MHA_TILE, dtype=bfloat16, device="npu")
    _mha_compile_probe(a, b, size=_MHA_TILE)
    # Reaching here means aiecc compiled mha.cc (with its softmax.cc/mm.cc
    # includes) and the design ran — the partial_softmax_* symbols resolved.


# ---------------------------------------------------------------------------
# generic/mv.cc:  bf16 matrix-vector multiply (c = A @ b).
#
# No kernels.* factory exposes this yet (kernels.mv resolves aie2/mv.cc, an
# i16->i32 kernel), so the ExternalFunction is hand-built directly against the
# source — same approach as the mha probe above.  The entry is
# matvec_vectorized_bf16_bf16(m, row_offset,
# a, b, c); it needs -DDIM_K and assumes k >= 2*VEC_SIZE (VEC_SIZE=64 -> k>=128).
# Once it gets a factory and a contract, this collapses into CASES above.
# ---------------------------------------------------------------------------

_MV_M = 32
_MV_K = 128


@iron.jit
def _mv_design(a_in: In, b_in: In, c_out: Out):
    from aie.iron.kernel import ExternalFunction
    from aie.iron.kernels._common import _include_dirs, _kernel_source

    # generic/ source is arch-independent; the subdir arg pins it to generic/mv.cc.
    src = _kernel_source("aie2", "generic", "mv.cc")
    a_ty = np.ndarray[(_MV_M * _MV_K,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(_MV_K,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(_MV_M,), np.dtype[bfloat16]]
    kern = ExternalFunction(
        "matvec_vectorized_bf16_bf16",
        source_file=str(src),
        arg_types=[np.int32, np.int32, a_ty, b_ty, c_ty],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DDIM_K={_MV_K}"],
    )
    of_a = ObjectFifo(a_ty, name="mva")
    of_b = ObjectFifo(b_ty, name="mvb")
    of_c = ObjectFifo(c_ty, name="mvc")

    def core(of_a, of_b, of_c, k):
        a = of_a.acquire(1)
        b = of_b.acquire(1)
        c = of_c.acquire(1)
        k(_MV_M, 0, a, b, c)  # (m, row_offset=0, a, b, c)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)

    w = Worker(core, fn_args=[of_a.cons(), of_b.cons(), of_c.prod(), kern])

    def seq(a, b, c, ah, bh, ch):
        ah.fill(a)
        bh.fill(b)
        ch.drain(c, wait=True)

    rt = Runtime(seq, [a_ty, b_ty, c_ty, of_a.prod(), of_b.prod(), of_c.cons()])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


def test_mv_bf16_e2e():
    from aie.utils.verify import Tolerance, compare

    rng = np.random.default_rng(7)
    mat = rng.uniform(-1, 1, size=(_MV_M, _MV_K)).astype(bfloat16)
    vec = rng.uniform(-1, 1, size=(_MV_K,)).astype(bfloat16)
    at = iron.tensor(mat.reshape(-1), dtype=bfloat16, device="npu")
    bt = iron.tensor(vec, dtype=bfloat16, device="npu")
    ct = iron.zeros(_MV_M, dtype=bfloat16, device="npu")

    _mv_design(at, bt, ct)

    expected = (mat.astype(np.float32) @ vec.astype(np.float32)).astype(bfloat16)
    verdict = compare(
        ct.numpy(),
        expected,
        Tolerance.relative(
            0.03, 0.05, max_mismatch_frac=0.02, note="fp32 accumulate, bf16 round"
        ),
    )
    assert verdict, verdict.detail
