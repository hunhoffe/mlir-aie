# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Squash a publication branch to one commit against a local bare remote."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "utils/dashboard/squash_gh_pages.sh"
GIT_ID = ["-c", "user.name=t", "-c", "user.email=t@example.com"]


def git(cwd, *args, **kw):
    return subprocess.run(
        ["git", *GIT_ID, *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        **kw,
    ).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    """A bare remote with a three-commit gh-pages, and a clone of it."""
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "-q", "--initial-branch=main", str(remote))
    work = tmp_path / "work"
    git(tmp_path, "clone", "-q", str(remote), str(work))
    git(work, "checkout", "-q", "--orphan", "gh-pages")
    for i in range(3):
        (work / f"file{i}.txt").write_text(f"content {i}\n")
        (work / "index.html").write_text(f"<p>deploy {i}</p>\n")
        git(work, "add", "-A")
        git(work, "commit", "-q", "-m", f"deploy {i}")
    git(work, "push", "-q", "origin", "gh-pages")
    return remote, work


def count(remote):
    return int(git(remote, "rev-list", "--count", "gh-pages"))


def test_squash_keeps_the_tree_and_drops_the_history(repos):
    remote, work = repos
    assert count(remote) == 3
    before = git(remote, "rev-parse", "gh-pages^{tree}")
    out = subprocess.run(
        [str(SCRIPT)], cwd=work, check=True, capture_output=True, text=True
    ).stdout
    assert "one commit, same tree" in out
    assert count(remote) == 1
    assert git(remote, "rev-parse", "gh-pages^{tree}") == before
    assert git(remote, "log", "-1", "--format=%s", "gh-pages").startswith(
        "gh-pages snapshot of "
    )
    # A second run finds nothing to do and touches nothing.
    tip = git(remote, "rev-parse", "gh-pages")
    out = subprocess.run(
        [str(SCRIPT)], cwd=work, check=True, capture_output=True, text=True
    ).stdout
    assert "already a single commit" in out
    assert git(remote, "rev-parse", "gh-pages") == tip


def test_squash_works_from_a_shallow_clone(repos, tmp_path):
    remote, _ = repos
    shallow = tmp_path / "shallow"
    git(
        tmp_path,
        "clone",
        "-q",
        "--depth=1",
        "--branch",
        "gh-pages",
        str(remote),
        str(shallow),
    )
    subprocess.run(
        [str(SCRIPT)], cwd=shallow, check=True, capture_output=True, text=True
    )
    assert count(remote) == 1
    assert (git(remote, "cat-file", "-p", "gh-pages:index.html")) == "<p>deploy 2</p>"


def test_squash_leaves_a_branch_someone_pushed_to_since(repos, tmp_path, monkeypatch):
    """The lease: a push between the fetch and the force-push is not overwritten."""
    remote, work = repos
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", "--branch", "gh-pages", str(remote), str(other))
    # Stand in for `git push` so the branch moves after the script fetched.
    fake = tmp_path / "bin"
    fake.mkdir()
    real_git = subprocess.run(
        ["which", "git"], capture_output=True, text=True, check=True
    ).stdout.strip()
    (fake / "git").write_text(
        "#!/usr/bin/env bash\n"
        f'if [ "$1" = push ] && [ -z "${{RACED:-}}" ]; then\n'
        f'  (cd "{other}" && echo late > late.txt && {real_git} -c user.name=t -c user.email=t@example.com add -A '
        f"&& {real_git} -c user.name=t -c user.email=t@example.com commit -q -m late && RACED=1 {real_git} push -q origin gh-pages)\n"
        "fi\n"
        f'exec {real_git} "$@"\n'
    )
    (fake / "git").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    result = subprocess.run([str(SCRIPT)], cwd=work, capture_output=True, text=True)
    assert result.returncode != 0
    assert "stale info" in result.stderr or "rejected" in result.stderr
    assert count(remote) == 4
    assert git(remote, "cat-file", "-p", "gh-pages:late.txt") == "late"
