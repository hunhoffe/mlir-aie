from __future__ import annotations

from collections import abc
from copy import deepcopy
import os
import subprocess


class Example:
    DEFAULT_MLIR = "aie2.mlir"
    DEFAULT_RUN = "run"
    DEFAULT_IRON_BUILD = "use_alt=1"
    DEFAULT_IRON_EXT_BUILD = ""

    def __init__(
        self,
        name: str,
        dir: str,
        category: str | None = None,
        iron_src: str | None = None,
        iron_ext_src: str | None = None,
        mlir_src: str = DEFAULT_MLIR,
        run_cmd: str = DEFAULT_RUN,
        iron_build_env: str = DEFAULT_IRON_BUILD,
        iron_ext_build_env: str = DEFAULT_IRON_EXT_BUILD,
    ):
        self._name = name
        self._category = category
        self._dir = dir
        self._iron_src = iron_src
        self._iron_ext_src = iron_ext_src
        self._mlir_src = mlir_src
        self._run_cmd = run_cmd
        self._iron_build_env = iron_build_env
        self._iron_ext_build_env = iron_ext_build_env

        # Make sure paths are valid
        self._dir = os.path.abspath(self._dir)
        if not os.path.exists(self._dir):
            raise ValueError(f"Invalid directory: {self._dir}")
        if not os.path.isdir(self._dir):
            raise ValueError(f"dir path exists but is not a directory: {self._dir}")

        # Set default values for source file names
        if self._iron_src is None or self._iron_ext_src is None:
            design_name = self._dir.split(os.path.sep)[-1]
            if self._iron_src is None:
                self._iron_src = f"{design_name}_alt.py"
            if self._iron_ext_src is None:
                self._iron_ext_src = f"{design_name}.py"

        # Check validity of source file path.
        self._iron_src = os.path.join(self._dir, self._iron_src)
        if not os.path.isfile(self._iron_src):
            raise ValueError(f"IRON src is not a file: {self._iron_src}")
        self._iron_ext_src = os.path.join(self._dir, self._iron_ext_src)
        if not os.path.isfile(self._iron_ext_src):
            raise ValueError(f"IRON ext src is not a file: {self._iron_ext_src}")

    def _run(self, build_env: str, verbose: bool = True):
        if verbose:
            print(f"Running {str(self)}...")

        result = subprocess.run(
            "make clean", shell=True, env=os.environ, cwd=self._dir, capture_output=True
        )
        if result.returncode != 0:
            print(f"============= Clean failed with exit code: {result.returncode}")
            print(result.stdout)
            print(result.stderr)
            return False
        elif verbose:
            print(f"\t Cleaned successfully!")

        result = subprocess.run(
            f"env {build_env} make",
            shell=True,
            env=os.environ,
            cwd=self._dir,
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"============= Build failed with exit code: {result.returncode}")
            print(result.stdout)
            print(result.stderr)
            return False
        elif verbose:
            print(f"\t Built successfully!")

        result = subprocess.run(
            f"make {self._run_cmd}",
            shell=True,
            env=os.environ,
            cwd=self._dir,
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"============= Run failed with exit code: {result.returncode}")
            print(result.stdout)
            print(result.stderr)
            return False
        elif verbose:
            print(f"\t Ran successfully!")

        if verbose:
            print(f"Done running {str(self)}!")
        return True

    def run_iron(self) -> bool:
        return self._run(self._iron_build_env)

    def run_iron_ext(self) -> bool:
        return self._run(self._iron_ext_build_env)

    def cmp_srcs(self):
        pass

    def __str__(self):
        if self._category is None:
            name = self._name
        else:
            name = f"{self._category}({self._name})"
        return name


class ExampleCollection(abc.MutableSequence, abc.Iterable):
    def __init__(self):
        self._examples = []

    def __contains__(self, e: Example):
        return e in self._examples

    def __iter__(self):
        return iter(self._examples)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]

    def __setitem__(self, idx: int, e: Example):
        self._examples[idx] = deepcopy(e)

    def __delitem__(self, idx: int):
        del self._examples[idx]

    def insert(self, index: int, value: Example):
        self._examples.insert(index, value)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._examples == other._examples and self._current_step
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def filter_category(self, category) -> ExampleCollection:
        ec = ExampleCollection()
        for e in self._examples:
            if e.category == category:
                ec.add(e)
        return ec

    def run_all(self, use_iron_ext: bool = False) -> None:
        for e in self._examples:
            if use_iron_ext:
                if not e.run_iron_ext():
                    raise Exception("Failed to run example!")
            else:
                if not e.run_iron():
                    raise Exception("Failed to run example!")
