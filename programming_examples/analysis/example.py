from __future__ import annotations

from collections import abc
from copy import deepcopy
import os


class Example:
    DEFAULT_MLIR = "aie2.mlir"
    DEFAULT_RUN = "run"

    def __init__(
        self,
        name: str,
        dir: str,
        category: str | None = None,
        iron_src: str | None = None,
        iron_ext_src: str | None = None,
        mlir_src: str = DEFAULT_MLIR,
        run_cmd: str = DEFAULT_RUN,
    ):
        self._name = name
        self._category = category
        self._dir = dir
        self._iron_src = iron_src
        self._iron_ext_src = iron_ext_src
        self._mlir_src = mlir_src
        self._run_cmd = run_cmd

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

    def run(self):
        pass

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

    def run_all(self) -> None:
        pass
