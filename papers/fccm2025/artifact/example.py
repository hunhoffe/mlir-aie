from __future__ import annotations

from collections import abc
from copy import deepcopy
import os
import shutil
import subprocess


class Example:
    DEFAULT_MLIR = "build/aie.mlir"
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
        self._iron_stdout = None
        self._iron_stderr = None
        self._iron_ext_stdout = None
        self._iron_ext_stderr = None

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
        self._iron_ext_src = os.path.join(self._dir, self._iron_ext_src)

        # Can't check mlir source yet if not generated
        self._mlir_src = os.path.join(self._dir, self._mlir_src)

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def mlir_src(self) -> str:
        return self._mlir_src

    @property
    def iron_src(self) -> str:
        return self._iron_src

    @property
    def iron_ext_src(self) -> str:
        return self._iron_ext_src

    def _run(self, build_env: str, is_iron_ext: bool, verbose: bool = True):
        if verbose:
            ext_str = "IRON"
            if is_iron_ext:
                ext_str = "IRON(ext)"
            print(f"Running {str(self)} {ext_str}...")

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

        if os.path.isfile(self._mlir_src):
            print(
                f"============= MLIR source file ({self._mlir_src}) exists even after clean?"
            )
            return False

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

        if not os.path.isfile(self._mlir_src):
            print(
                f"============= MLIR source file ({self._mlir_src}) does not exist even after build?"
            )
            return False

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

        if is_iron_ext:
            self._iron_ext_stdout = result.stdout
            self._iron_ext_stderr = result.stderr
            print(self._iron_ext_stdout)
            print(self._iron_ext_stderr)
        else:
            self._iron_stdout = result.stdout
            self._iron_stderr = result.stderr
            print(self._iron_stdout)
            print(self._iron_stderr)

        if verbose:
            print(f"Done running {str(self)}!")
        return True

    def run_iron(self, verbose: bool = True) -> bool:
        return self._run(self._iron_build_env, is_iron_ext=False, verbose=verbose)

    def run_iron_ext(self, verbose: bool = True) -> bool:
        return self._run(self._iron_ext_build_env, is_iron_ext=True, verbose=verbose)

    def write_results(self, iron_dir: str, iron_ext_dir: str) -> None:
        print(self)
        if self._iron_stderr:
            with open(os.path.join(iron_dir, f"{str(self)}.stderr.txt"), "wb") as f:
                f.write(self._iron_stderr)
        with open(os.path.join(iron_dir, f"{str(self)}.stdout.txt"), "wb") as f:
            f.write(self._iron_stdout)

        if self._iron_ext_stderr:
            with open(os.path.join(iron_ext_dir, f"{str(self)}.stderr.txt"), "wb") as f:
                f.write(self._iron_ext_stderr)
        with open(os.path.join(iron_ext_dir, f"{str(self)}.stdout.txt"), "wb") as f:
            f.write(self._iron_ext_stdout)

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

    def filter(self, term: str) -> ExampleCollection:
        ec = ExampleCollection()
        term = term.lower()
        for e in self._examples:
            if term in e.category or term in e.name:
                ec.add(e)
        return ec

    def run_all(self, use_iron_ext: bool = False, verbose: bool = True) -> None:
        for e in self._examples:
            if use_iron_ext:
                if not e.run_iron_ext(verbose=verbose):
                    raise Exception("Failed to run example!")
            else:
                if not e.run_iron(verbose=verbose):
                    raise Exception("Failed to run example!")

    def collect_mlir_and_results(self, example_dir: str, verbose: bool = True):

        iron_dir = os.path.join(example_dir, "iron")
        assert os.path.exists(iron_dir) and os.path.isdir(iron_dir)
        iron_ext_dir = os.path.join(example_dir, "iron_ext")
        assert os.path.exists(iron_ext_dir) and os.path.isdir(iron_ext_dir)

        # Create one subdir for mlir and results per design type
        iron_mlir_dir = os.path.join(example_dir, "iron_mlir")
        os.makedirs(iron_mlir_dir)
        iron_result_dir = os.path.join(example_dir, "iron_result")
        os.makedirs(iron_result_dir)
        iron_ext_mlir_dir = os.path.join(example_dir, "iron_ext_mlir")
        os.makedirs(iron_ext_mlir_dir)
        iron_ext_result_dir = os.path.join(example_dir, "iron_ext_result")
        os.makedirs(iron_ext_result_dir)

        if verbose:
            print(
                f"Collecting files in {example_dir}/({iron_mlir_dir}|{iron_result_dir}|{iron_ext_mlir_dir}|{iron_result_dir})"
            )

        # Copy local example code into programming examples directories which contain makefiles, etc.
        for e in self._examples:
            shutil.copyfile(os.path.join(iron_dir, f"{str(e)}.py"), e.iron_src)
            shutil.copyfile(os.path.join(iron_ext_dir, f"{str(e)}.py"), e.iron_ext_src)

        # Generate the MLIR and copy files
        self.run_all(use_iron_ext=False, verbose=verbose)
        for e in self._examples:
            shutil.copyfile(e.mlir_src, os.path.join(iron_mlir_dir, f"{str(e)}.mlir"))
        print(f"Collected IRON MLIR in {iron_mlir_dir}")

        # Generate the MLIR and copy files for ext
        self.run_all(use_iron_ext=True, verbose=verbose)
        for e in self._examples:
            shutil.copyfile(
                e.mlir_src, os.path.join(iron_ext_mlir_dir, f"{str(e)}.mlir")
            )
            e.write_results(iron_result_dir, iron_ext_result_dir)
        print(
            f"Collected IRON(ext) src in {iron_ext_dir} and results for IRON and IRON(ext) in {iron_result_dir} and {iron_ext_result_dir} respectively."
        )

        print(f"Done!")
