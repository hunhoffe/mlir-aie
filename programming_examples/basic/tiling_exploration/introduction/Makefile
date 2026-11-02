# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: all run clean

all: run

# Runs the code blocks of both taplib pages, in order.
run: ${srcdir}/README.md ${srcdir}/part2.md
	python3 ${srcdir}/../run_markdown.py ${srcdir}/README.md ${srcdir}/part2.md

clean:
	rm -rf build
