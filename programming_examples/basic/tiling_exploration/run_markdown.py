# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run the ``python`` code blocks of Markdown pages, in order, as one script.

    python3 run_markdown.py README.md part2.md

The taplib pages keep their examples as code blocks, with the pictures
drawn from those blocks beside them. This checks that the code still runs
(and that its asserts hold). A ``python skip`` block is shown on the page
but not run. Each page runs in its own directory, so it can import the
design next to it. Plots are drawn off screen and discarded.
"""

import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

FENCE = re.compile(r"^(```+)[ \t]*python[ \t]*\n(.*?)^\1[ \t]*$", re.S | re.M)


def run(page: Path) -> int:
    page = page.resolve()
    ns = {"__name__": "__main__", "__file__": str(page)}
    here = os.getcwd()
    os.chdir(page.parent)
    sys.path.insert(0, str(page.parent))
    try:
        blocks = FENCE.findall(page.read_text())
        for i, (_, code) in enumerate(blocks):
            exec(compile(code, f"{page.name}[block {i + 1}]", "exec"), ns)
    finally:
        sys.path.remove(str(page.parent))
        os.chdir(here)
    return len(blocks)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(f"{arg}: {run(Path(arg))} code blocks ran")
    print("Complete!")
