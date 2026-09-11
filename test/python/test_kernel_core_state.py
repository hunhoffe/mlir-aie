# test_kernel_core_state.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""The dirty-state harness mode, without a device.

``test/python/npu/test_kernels_e2e.py -m core_state`` runs every kernel with
the core's rounding and saturation registers preset to every mode and judges
the output and a register probe against the contract. Everything that can be
pinned on a host is pinned here:

* the contract fields (``saturation_mode``, ``leaves_rounding``,
  ``leaves_saturation``) validate, and a register can only be "left" in a
  mode by a source that writes it;
* the probe's C++ code table matches the Python mode tables it indexes;
* ``expected_state`` / ``judge_state`` / ``judge_dirty`` say the right thing
  for every combination of contract claim and preset, with messages that
  name the fix;
* a design built with ``core_state`` presets, then applies the contract's
  setters, then probes, then runs, then probes again, in that order, and
  takes the extra state tensor; a design built without it is unchanged;
* one case per compiled kernel is swept, the smallest;
* every register-writing source is declared, and nothing else is.
"""

from __future__ import annotations

import re
import types
from pathlib import Path

import numpy as np
import pytest
from aie.iron import kernels
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernels import (
    BOOT_ROUNDING,
    BOOT_SATURATION,
    ROUNDING_MODES,
    SATURATION_MODES,
    KernelContract,
)
from aie.iron.kernels.core import CORE_STATE_MARKER, CORE_STATE_WORDS
from aie.utils import kernel_harness as kh
from aie.utils.hostruntime import set_current_device
from aie.utils.kernel_harness import CoreState, CoreStateReading
from aie.utils.kernel_harness.cases import distinct_kernels, kernel_build_key
from ml_dtypes import bfloat16

REPO = Path(__file__).resolve().parents[2]
PROBE_SOURCE = REPO / "aie_kernels" / "generic" / "read_core_state.cc"

ROUNDING = ROUNDING_MODES[2:]
SATURATION = SATURATION_MODES[2:]


@pytest.fixture(autouse=True)
def _aie2p_device():
    set_current_device(NPU2Col1())
    yield
    set_current_device(None)


# ---------------------------------------------------------------------------
# Contract fields
# ---------------------------------------------------------------------------


def test_contract_defaults_claim_independence_and_preservation():
    c = KernelContract(roles=("in", "out"))
    assert c.saturation_mode == "unspecified"
    assert c.leaves_rounding == "preserves"
    assert c.leaves_saturation == "preserves"
    assert c.needs_saturation_mode is None


def test_contract_validates_saturation_and_leaves():
    with pytest.raises(ValueError, match="saturation_mode must be"):
        KernelContract(roles=("in", "out"), saturation_mode="clamp")
    with pytest.raises(ValueError, match="leaves_rounding must be"):
        KernelContract(roles=("in", "out"), leaves_rounding="unspecified")
    with pytest.raises(ValueError, match="leaves_saturation must be"):
        KernelContract(roles=("in", "out"), leaves_saturation="sets_own")
    # Only a source that writes the register can leave it in a named mode.
    with pytest.raises(ValueError, match="needs rounding_mode='sets_own'"):
        KernelContract(
            roles=("in", "out"), rounding_mode="conv_even", leaves_rounding="conv_even"
        )
    with pytest.raises(ValueError, match="needs saturation_mode='sets_own'"):
        KernelContract(roles=("in", "out"), leaves_saturation="saturate")
    c = KernelContract(
        roles=("in", "out"),
        rounding_mode="sets_own",
        saturation_mode="sets_own",
        leaves_rounding="positive_inf",
        leaves_saturation="saturate",
    )
    assert (c.needs_rounding_mode, c.needs_saturation_mode) == (None, None)
    assert (
        KernelContract(
            roles=("in", "out"), saturation_mode="saturate"
        ).needs_saturation_mode
        == "saturate"
    )


def test_mode_tables_name_the_boot_state():
    assert BOOT_ROUNDING in ROUNDING and BOOT_SATURATION in SATURATION
    assert set(SATURATION) == {"none", "saturate", "symmetric"}


# ---------------------------------------------------------------------------
# The core-state kernels and the probe's code table
# ---------------------------------------------------------------------------


def test_set_saturation_factory():
    for mode in SATURATION:
        ef = kernels.set_saturation(mode)
        assert ef.name.endswith(f"set_saturation_{mode}")
        assert f"-DSATURATION_MODE={mode}" in ef.compile_flags
        assert kh._arg_types(ef) == []
    for bad in ("unspecified", "sets_own", "clamp"):
        with pytest.raises(ValueError, match="aie::saturation_mode name"):
            kernels.set_saturation(bad)
    assert "set_saturation" in kernels.__all__ and "read_core_state" in kernels.__all__


def test_probe_factory_takes_one_state_tile():
    ef = kernels.read_core_state()
    assert ef.name.endswith("read_core_state")
    (ty,) = kh._arg_types(ef)
    assert kh._shape_dtype(ty) == ((CORE_STATE_WORDS,), np.int32)
    assert getattr(ef, "contract", None) is None
    assert CORE_STATE_WORDS == kh._STATE_WORDS
    assert CORE_STATE_MARKER == kh._STATE_MARKER


def _code_table(source: str, enum: str) -> dict[str, int]:
    return {
        name: int(code)
        for name, code in re.findall(
            rf"case aie::{enum}::(\w+):\s*return (\d+);", source
        )
    }


def test_probe_codes_match_the_python_tables():
    """read_core_state.cc numbers the modes by their position in the Python tables."""
    src = PROBE_SOURCE.read_text()
    assert _code_table(src, "rounding_mode") == {m: i for i, m in enumerate(ROUNDING)}
    assert _code_table(src, "saturation_mode") == {
        m: i for i, m in enumerate(SATURATION)
    }
    marker = re.search(r"out\[2\] = (0x[0-9A-Fa-f]+);", src)
    assert marker and int(marker.group(1), 16) == CORE_STATE_MARKER
    assert "default:\n    return -1;" in src  # an unknown enumerator decodes to None


def _words(r: int, s: int, marker: int = CORE_STATE_MARKER) -> list[int]:
    return [r, s, marker, 0]


def test_decode_core_state():
    words = _words(ROUNDING.index("ceil"), SATURATION.index("saturate")) + _words(
        ROUNDING.index("conv_even"), SATURATION.index("none")
    )
    reading = kh.decode_core_state(np.array(words, dtype=np.int32))
    assert reading.before == CoreState("ceil", "saturate")
    assert reading.after == CoreState("conv_even", "none")
    assert reading.ok
    # The poison the harness fills the tile with decodes to nothing at all.
    poison = np.full(2 * CORE_STATE_WORDS * 4, 0x55, dtype=np.uint8).view(np.int32)
    reading = kh.decode_core_state(poison)
    assert reading.before == CoreState(None, None)
    assert not reading.marker_ok and not reading.ok
    # -1 (an enumerator the probe does not know) and a missing marker.
    reading = kh.decode_core_state(np.array(_words(-1, 0) + _words(0, 0, 0)))
    assert reading.before.rounding is None and reading.before.saturation == "none"
    assert not reading.marker_ok


# ---------------------------------------------------------------------------
# expected_state / judge_state / judge_dirty
# ---------------------------------------------------------------------------


def _fn(**contract) -> types.SimpleNamespace:
    """Return a stand-in kernel; the judges only read ``.name`` and ``.contract``."""
    return types.SimpleNamespace(
        name="k", contract=KernelContract(roles=("in", "out"), **contract)
    )


UNSPEC = _fn()
NAMED = _fn(rounding_mode="conv_even", saturation_mode="saturate")
SETS_AND_LEAVES = _fn(
    rounding_mode="sets_own",
    saturation_mode="sets_own",
    leaves_rounding="positive_inf",
    leaves_saturation="saturate",
)
SETS_AND_RESTORES = _fn(rounding_mode="sets_own")


@pytest.mark.parametrize(
    "fn, preset, before, after",
    [
        # no claim: the kernel sees the preset and leaves it
        (
            UNSPEC,
            CoreState("ceil", "symmetric"),
            ("ceil", "symmetric"),
            ("ceil", "symmetric"),
        ),
        # no preset: the boot state
        (
            UNSPEC,
            CoreState(None, None),
            (BOOT_ROUNDING, BOOT_SATURATION),
            (BOOT_ROUNDING, BOOT_SATURATION),
        ),
        # a named mode is set after the preset, so the kernel never sees the preset
        (
            NAMED,
            CoreState("ceil", "none"),
            ("conv_even", "saturate"),
            ("conv_even", "saturate"),
        ),
        (
            NAMED,
            CoreState(None, None),
            ("conv_even", "saturate"),
            ("conv_even", "saturate"),
        ),
        # sets_own: the preset reaches the kernel; what it leaves is declared
        (
            SETS_AND_LEAVES,
            CoreState("ceil", "none"),
            ("ceil", "none"),
            ("positive_inf", "saturate"),
        ),
        # sets_own and restores: leaves what it found
        (
            SETS_AND_RESTORES,
            CoreState("conv_odd", "symmetric"),
            ("conv_odd", "symmetric"),
            ("conv_odd", "symmetric"),
        ),
    ],
)
def test_expected_state(fn, preset, before, after):
    assert kh.expected_state(fn, preset) == (CoreState(*before), CoreState(*after))


def _reading(before, after, marker_ok=True) -> CoreStateReading:
    return CoreStateReading(CoreState(*before), CoreState(*after), marker_ok)


def test_judge_state_passes_a_reading_that_matches_the_contract():
    preset = CoreState("ceil", "none")
    assert kh.judge_state(
        SETS_AND_LEAVES,
        _reading(("ceil", "none"), ("positive_inf", "saturate")),
        preset,
    )
    assert kh.judge_state(
        NAMED, _reading(("conv_even", "saturate"), ("conv_even", "saturate")), preset
    )


def test_judge_state_reports_a_preset_or_setter_that_did_not_take():
    v = kh.judge_state(
        UNSPEC,
        _reading(("floor", "none"), ("floor", "none")),
        CoreState("ceil", "none"),
    )
    assert not v and "before the first call" in v.detail and "preset" in v.detail
    v = kh.judge_state(
        NAMED, _reading(("ceil", "none"), ("ceil", "none")), CoreState("ceil", "none")
    )
    assert not v and "rounding_mode='conv_even'" in v.detail


def test_judge_state_reports_an_undeclared_leftover():
    v = kh.judge_state(
        UNSPEC,
        _reading(("ceil", "none"), ("conv_even", "none")),
        CoreState("ceil", "none"),
    )
    assert not v
    assert "after the last call" in v.detail
    assert "leaves_rounding='preserves'" in v.detail
    assert "swap_rounding" in v.detail
    v = kh.judge_state(
        SETS_AND_LEAVES,
        _reading(("ceil", "none"), ("positive_inf", "none")),
        CoreState("ceil", "none"),
    )
    assert not v and "leaves_saturation='saturate'" in v.detail


def test_judge_state_reports_a_probe_that_never_ran():
    v = kh.judge_state(
        UNSPEC,
        _reading((None, None), (None, None), marker_ok=False),
        CoreState("ceil", "none"),
    )
    assert not v and "marker missing" in v.detail


def test_judge_dirty_passes_bit_identical_output_including_nans():
    a = np.array([1.0, np.nan, -0.0, np.inf], dtype=np.float32)
    v = kh.judge_dirty(UNSPEC, a, a.copy(), CoreState("ceil", None))
    assert v and v.n_mismatch == 0
    b = a.astype(bfloat16)
    assert kh.judge_dirty(UNSPEC, b, b.copy(), CoreState("ceil", None))


def test_judge_dirty_fails_any_difference_whatever_the_tolerance():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = a.copy()
    b[2] = np.nextafter(np.float32(3.0), np.float32(4.0))  # one ulp
    v = kh.judge_dirty(UNSPEC, a, b, CoreState("ceil", None))
    assert not v
    assert v.n_mismatch == 1 and v.first_bad_index == 2
    assert "1 of 4 elements differ" in v.detail
    # -0.0 and 0.0 are different bit patterns, so they are a difference too.
    z = np.array([0.0], dtype=np.float32)
    assert not kh.judge_dirty(UNSPEC, z, -z, CoreState("ceil", None))
    # Different NaN payloads likewise.
    n1 = np.array([np.nan], dtype=np.float32)
    n2 = n1.view(np.uint32) ^ np.uint32(1)
    assert not kh.judge_dirty(UNSPEC, n1, n2.view(np.float32), CoreState("ceil", None))


def test_judge_dirty_names_the_fix_per_claim():
    a = np.zeros(4, dtype=np.int16)
    b = a.copy()
    b[0] = 1
    v = kh.judge_dirty(UNSPEC, a, b, CoreState("ceil", None))
    assert "rounding_mode='unspecified'" in v.detail
    assert "name the aie::rounding_mode the kernel needs" in v.detail
    assert "saturation_mode" not in v.detail  # saturation was not preset
    v = kh.judge_dirty(UNSPEC, a, b, CoreState(None, "saturate"))
    assert (
        "saturation_mode='unspecified'" in v.detail and "rounding_mode" not in v.detail
    )
    v = kh.judge_dirty(SETS_AND_LEAVES, a, b, CoreState("ceil", "saturate"))
    assert (
        "rounding_mode='sets_own' (the source must set the register itself" in v.detail
    )
    assert (
        "saturation_mode='sets_own' (the source must set the register itself"
        in v.detail
    )
    v = kh.judge_dirty(NAMED, a, b, CoreState("ceil", None))
    assert "harness sets that mode after the preset" in v.detail


def test_judge_dirty_refuses_outputs_of_different_size():
    with pytest.raises(ValueError, match="differ in size"):
        kh.judge_dirty(UNSPEC, np.zeros(4), np.zeros(5), CoreState("ceil", None))


# ---------------------------------------------------------------------------
# The design: preset -> contract setters -> probe -> kernel -> probe
# ---------------------------------------------------------------------------


def _core_calls(mlir: str) -> list[str]:
    """Symbols called from the first aie.core body, in order, prefixes dropped."""
    body = mlir.split("aie.core", 1)[1].split("aie.end", 1)[0]
    return [
        re.sub(r"^[0-9a-f]{8}_", "", sym)
        for sym in re.findall(r"func\.call @(\w+)", body)
    ]


def test_core_state_validates_its_modes():
    with pytest.raises(ValueError, match="CoreState.rounding"):
        kh.design(kernels.add, calls=2, core_state=CoreState("banker", None))
    with pytest.raises(ValueError, match="CoreState.saturation"):
        kh.design(kernels.add, calls=2, core_state=CoreState(None, "clamp"))


def test_design_presets_then_sets_then_probes_then_runs():
    """``add`` names conv_even: the preset comes first, its own mode overrides it."""
    assert kernels.add().contract.needs_rounding_mode == "conv_even"
    mlir = str(
        kh.design(
            kernels.add, calls=2, core_state=CoreState("ceil", "saturate")
        ).as_mlir()
    )
    calls = _core_calls(mlir)
    assert calls[:3] == [
        "set_rounding_ceil",
        "set_saturation_saturate",
        "set_rounding_conv_even",
    ]
    assert calls[3] == "read_core_state"
    assert calls[-1] == "read_core_state"
    assert any(c.startswith("eltwise_add") for c in calls[4:-1]), calls
    assert calls.count("read_core_state") == 2
    assert f"memref<{2 * CORE_STATE_WORDS}xi32>" in mlir  # the extra state tensor


def test_design_presets_only_the_registers_named():
    mlir = str(
        kh.design(kernels.add, calls=2, core_state=CoreState("ceil", None)).as_mlir()
    )
    calls = _core_calls(mlir)
    assert calls[:2] == ["set_rounding_ceil", "set_rounding_conv_even"]
    assert "set_saturation" not in mlir


def test_design_does_not_repeat_a_preset_the_contract_names():
    mlir = str(
        kh.design(
            kernels.add, calls=2, core_state=CoreState("conv_even", None)
        ).as_mlir()
    )
    assert _core_calls(mlir)[:2] == ["set_rounding_conv_even", "read_core_state"]


def test_sets_own_kernel_gets_the_preset_and_no_setter():
    assert kernels.convert_copy().contract.rounding_mode == "sets_own"
    mlir = str(
        kh.design(
            kernels.convert_copy, calls=1, core_state=CoreState("ceil", None)
        ).as_mlir()
    )
    calls = _core_calls(mlir)
    assert calls[:2] == ["set_rounding_ceil", "read_core_state"]
    assert "set_rounding_conv_even" not in mlir


def test_probe_only_design_touches_no_register():
    mlir = str(
        kh.design(
            kernels.passthrough, calls=2, core_state=CoreState(None, None)
        ).as_mlir()
    )
    assert "set_rounding" not in mlir and "set_saturation" not in mlir
    assert _core_calls(mlir).count("read_core_state") == 2


def test_design_without_core_state_is_unchanged():
    mlir = str(kh.design(kernels.add, calls=2).as_mlir())
    assert "read_core_state" not in mlir
    assert f"memref<{2 * CORE_STATE_WORDS}xi32>" not in mlir
    assert "set_rounding_conv_even" in mlir


@pytest.mark.parametrize(
    "factory, opts",
    [
        (
            kernels.mm,
            dict(
                shape=(128, 128, 128),
                dim_m=64,
                dim_k=32,
                dim_n=64,
                input_dtype=bfloat16,
                output_dtype=np.float32,
            ),
        ),
        (kernels.mv, dict(shape=(128, 128), dim_m=32, dim_k=32)),
        (kernels.swiglu, dict(calls=2)),  # three inputs in one fifo
    ],
)
def test_matrix_and_packed_designs_probe_too(factory, opts):
    mlir = str(
        kh.design(factory, core_state=CoreState("ceil", "saturate"), **opts).as_mlir()
    )
    calls = _core_calls(mlir)
    assert calls[0] == "set_rounding_ceil" and calls[1] == "set_saturation_saturate"
    assert calls.count("read_core_state") == 2 and calls[-1] == "read_core_state"
    assert f"memref<{2 * CORE_STATE_WORDS}xi32>" in mlir


# ---------------------------------------------------------------------------
# One case per compiled kernel
# ---------------------------------------------------------------------------


def _cases():
    import importlib.util

    path = REPO / "test" / "python" / "npu" / "kernel_cases.py"
    spec = importlib.util.spec_from_file_location("kernel_cases", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return list(mod.CASES)


def _e2e_module():
    import importlib.util

    path = REPO / "test" / "python" / "npu" / "test_kernels_e2e.py"
    spec = importlib.util.spec_from_file_location("test_kernels_e2e", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.path.insert(0, str(path.parent))  # for `from kernel_cases import CASES`
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def test_trimmed_presets_touch_both_registers():
    """The nightly's presets still exercise both registers; ``full`` is every mode."""
    mod = _e2e_module()
    trimmed, full = mod.DIRTY_STATES["trimmed"], mod.DIRTY_STATES["full"]
    assert mod.CLEAN_STATE == CoreState(BOOT_ROUNDING, BOOT_SATURATION)
    assert mod.CLEAN_STATE not in trimmed and mod.CLEAN_STATE not in full
    assert {s.rounding for s in trimmed} >= {"ceil", "conv_even"}
    assert "saturate" in {s.saturation for s in trimmed}
    assert set(trimmed) <= set(full)
    assert {s.rounding for s in full} == set(ROUNDING)
    assert {s.saturation for s in full} == set(SATURATION)
    assert len(full) == len(ROUNDING) - 1 + len(SATURATION) - 1


def test_distinct_kernels_cover_every_build_once_with_the_smallest_case():
    cases = _cases()
    chosen = distinct_kernels(cases)
    keys = [kernel_build_key(c.fn()) for c in chosen]
    assert len(keys) == len(set(keys)), "a build is swept twice"
    assert set(keys) == {
        kernel_build_key(c.fn()) for c in cases
    }, "a build is not swept"
    by_key = {k: c for k, c in zip(keys, chosen)}
    for case in cases:
        best = by_key[kernel_build_key(case.fn())]
        assert best.kernel_calls() <= case.kernel_calls()
    # Different tile sizes of one factory are one build; different flags are not.
    names = {c.factory for c in chosen}
    assert "passthrough" in names and "mm" in names
    assert sum(c.factory == "passthrough" and not c.kwargs for c in chosen) == 1
    assert sum(c.factory == "mm" for c in chosen) >= 4  # bf16, i16, i8, col-major flags


# ---------------------------------------------------------------------------
# Declarations follow the sources
# ---------------------------------------------------------------------------

_SETS_SATURATION = re.compile(
    r"^\s*(?!//)[^/\n]*\b(set_saturation\s*\(|set_sat\s*\(\s*\))", re.M
)
_SETS_ROUNDING = re.compile(r"^\s*(?!//)[^/\n]*\bset_rounding\s*\(", re.M)
_RESTORES_ROUNDING = re.compile(
    r"swap_rounding\s*\(.*\n(?:.*\n)*?.*set_rounding\s*\(\s*saved"
)

# What each register-writing build leaves behind, per architecture, from
# reading the sources: (leaves_rounding, leaves_saturation). A build not
# listed must preserve both. layer_norm.cc holds three entry points with
# three behaviours, which is why this is a table and not a file-level regex.
LEAVES = {
    "aie2": {
        "conv2dk1": ("positive_inf", "saturate"),
        "conv2dk3": ("positive_inf", "saturate"),
        "conv2dk1_skip": ("positive_inf", "saturate"),
        "conv2dk1_i8": ("symmetric_inf", "saturate"),
        "conv2dk14": ("symmetric_inf", "saturate"),
        "conv2dk1_skip_init": ("positive_inf", "saturate"),
        "filter2d": ("preserves", "saturate"),
        "add_weighted": ("preserves", "saturate"),
        "mv/dim_k=256/input_dtype=bfloat16/output_dtype=bfloat16": (
            "conv_even",
            "preserves",
        ),
    },
}
LEAVES["aie2p"] = dict(
    LEAVES["aie2"],
    softmax=("conv_even", "preserves"),
    layer_norm=("conv_even", "preserves"),
    mha=("conv_even", "preserves"),
)


@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
def test_register_effects_follow_the_sources(arch):
    from aie.utils.compile.remarks import kernel_builds

    set_current_device(NPU1Col1() if arch == "aie2" else NPU2Col1())
    seen = set()
    for name, ef in kernel_builds():
        c = getattr(ef, "contract", None)
        if c is None:
            continue
        src = Path(ef.source_file).read_text() if ef.source_file else ef.source_string
        base = name.split("/")[0]
        # Saturation: a source that writes the register declares sets_own and
        # what it leaves; one that does not cannot claim either.
        if _SETS_SATURATION.search(src):
            assert c.saturation_mode == "sets_own", f"{name}: source sets saturation"
        else:
            assert (
                c.saturation_mode != "sets_own"
            ), f"{name}: source never sets saturation"
            assert c.leaves_saturation == "preserves", name
        # Rounding: a named leftover needs a set_rounding of that mode in the
        # source with no restore of the saved value on that path.
        expected = LEAVES[arch].get(name) or LEAVES[arch].get(
            base, ("preserves", "preserves")
        )
        if name in LEAVES[arch] or base in LEAVES[arch]:
            seen.add(name if name in LEAVES[arch] else base)
        assert (c.leaves_rounding, c.leaves_saturation) == expected, name
        if c.leaves_rounding != "preserves":
            mode = c.leaves_rounding
            sets = re.search(
                rf"set_rounding\s*\(\s*(?:::)?aie::rounding_mode::{mode}\)", src
            )
            macro = re.search(
                rf"#define ROUNDING_MODE aie::rounding_mode::{mode}\b", src
            )
            assert sets or macro, f"{name}: no set_rounding({mode}) in the source"
        elif _SETS_ROUNDING.search(src) and c.rounding_mode == "sets_own":
            assert _RESTORES_ROUNDING.search(
                src
            ), f"{name}: sets rounding but declares preserves"
    missing = {n for n in LEAVES[arch] if n not in seen and n.split("/")[0] not in seen}
    assert not missing, f"{arch}: LEAVES names builds that no longer exist: {missing}"
