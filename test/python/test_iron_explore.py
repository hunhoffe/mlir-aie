# test_iron_explore.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""The optional IRON Design Explorer hook: IRON_EXPLORE, explore=True and
design.explore(). The explorer itself is stood in for, so nothing here needs
it installed, and no NPU or compile is involved."""

import json
import sys
import types
from pathlib import Path

import pytest
from aie.utils.compile.jit import _explore
from aie.utils.compile.jit.compilabledesign import CompilableDesign
from aie.utils.compile.jit.markers import CompileTime, In, Out
from aie.utils.jit import jit


def _gemm(a: In, c: Out, *, N: CompileTime[int]):
    pass


@pytest.fixture
def explorer(monkeypatch):
    """A stand-in iron_explorer that records what it was asked to draw."""
    calls = []

    def explore(build_dir, name, flags):
        calls.append((Path(build_dir), name, flags))
        return types.SimpleNamespace(path=Path(build_dir) / "explore" / f"{name}.html")

    monkeypatch.setitem(
        sys.modules, "iron_explorer", types.SimpleNamespace(explore=explore)
    )
    monkeypatch.setattr(_explore, "_announced", set())
    return calls


@pytest.mark.parametrize(
    "setting, env, expected",
    [
        (None, None, False),
        (None, "1", True),
        (None, "on", True),
        (None, "0", False),
        (None, "false", False),
        (True, None, True),
        (False, "1", False),  # the design's own setting wins
    ],
)
def test_requested_follows_the_design_then_the_environment(
    monkeypatch, setting, env, expected
):
    if env is None:
        monkeypatch.delenv(_explore.ENV, raising=False)
    else:
        monkeypatch.setenv(_explore.ENV, env)
    assert _explore.requested(setting) is expected


def test_explore_is_configuration_not_a_compile_parameter():
    design = jit(explore=True, N=8)(_gemm)
    assert design.compilable._compile_config.explore is True
    assert dict(design.compilable.compile_kwargs) == {"N": 8}
    assert design.compilable.specialize(N=16)._compile_config.explore is True


def test_explore_leaves_the_cache_key_alone():
    plain = CompilableDesign(_gemm, compile_kwargs={"N": 8})
    exploring = CompilableDesign(_gemm, compile_kwargs={"N": 8}, explore=True)
    assert plain._compute_cache_hash() == exploring._compute_cache_hash()


def test_explore_round_trips_through_json():
    design = CompilableDesign(_gemm, compile_kwargs={"N": 8}, explore=False)
    again = CompilableDesign.from_json(design.to_json(), _gemm)
    assert again._compile_config.explore is False
    assert json.loads(CompilableDesign(_gemm).to_json())["explore"] is None


def _compiled(monkeypatch, tmp_path, **config):
    """A design whose compile() finds its build in `tmp_path`, without aiecc."""
    design = CompilableDesign(
        _gemm, compile_kwargs={"N": 8}, aiecc_flags=["--placer=sa_placer"], **config
    )

    def fake_compile(*args):
        design._kernel_dir = tmp_path
        return tmp_path / "final.xclbin", tmp_path / "insts.bin"

    monkeypatch.setattr(design, "_compile", fake_compile)
    return design


def test_compile_writes_the_page_and_prints_it_once(
    monkeypatch, tmp_path, explorer, capsys
):
    design = _compiled(monkeypatch, tmp_path, explore=True)
    assert design.compile() == (tmp_path / "final.xclbin", tmp_path / "insts.bin")
    design.compile()
    assert explorer == [(tmp_path, "_gemm", ["--placer=sa_placer"])] * 2
    assert capsys.readouterr().out.count("explore ") == 1


def test_compile_without_explore_never_imports_the_explorer(
    monkeypatch, tmp_path, explorer
):
    monkeypatch.delenv(_explore.ENV, raising=False)
    _compiled(monkeypatch, tmp_path).compile()
    assert explorer == []


def test_environment_turns_it_on(monkeypatch, tmp_path, explorer):
    monkeypatch.setenv(_explore.ENV, "1")
    _compiled(monkeypatch, tmp_path).compile()
    assert len(explorer) == 1


def test_a_missing_explorer_never_fails_a_compile(monkeypatch, tmp_path, caplog):
    monkeypatch.setitem(sys.modules, "iron_explorer", None)  # import fails
    monkeypatch.setattr(_explore, "_announced", set())
    design = _compiled(monkeypatch, tmp_path, explore=True)
    design.compile()
    design.compile()
    assert caplog.text.count("not installed") == 1
    with pytest.raises(ImportError, match="pip install"):
        _explore.write_page(tmp_path, "_gemm", [], required=True)


def test_a_failing_explorer_never_fails_a_compile(monkeypatch, tmp_path, caplog):
    def explore(*args, **kwargs):
        raise RuntimeError("no stages")

    monkeypatch.setitem(
        sys.modules, "iron_explorer", types.SimpleNamespace(explore=explore)
    )
    _compiled(monkeypatch, tmp_path, explore=True).compile()
    assert "no stages" in caplog.text
    with pytest.raises(RuntimeError):
        _explore.write_page(tmp_path, "_gemm", [], required=True)


def test_explore_draws_an_existing_build(monkeypatch, tmp_path, explorer):
    (tmp_path / "aie.mlir").write_text("module {}")
    (tmp_path / "input_with_addresses.mlir").write_text("module {}")
    design = CompilableDesign(_gemm, compile_kwargs={"N": 8})
    design._kernel_dir = tmp_path
    page = design.explore()
    assert page.path == tmp_path / "explore" / "_gemm.html"
    assert explorer == [(tmp_path, "_gemm", [])]


def test_a_failed_compile_gets_a_page_and_still_fails(monkeypatch, tmp_path, explorer):
    design = CompilableDesign(_gemm, compile_kwargs={"N": 8}, explore=True)

    def failing_compile(*args):
        raise RuntimeError("[aiecc] Compilation failed")

    monkeypatch.setattr(design, "_compile", failing_compile)
    monkeypatch.setattr(design, "_explore_build", lambda **kw: tmp_path)
    with pytest.raises(RuntimeError, match="Compilation failed"):
        design.compile()
    assert explorer == [(tmp_path, "_gemm", [])]
