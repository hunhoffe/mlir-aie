<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Tiling Exploration

## Overview

This directory contains resources for the Tensor Access Pattern library (`taplib`) on AIEs with IRON. `taplib` describes DMA data movements as `TensorAccessPattern`s: strided walks over a tensor that are tiled with `TensorAccessPattern.full(dims).tile(tile_dims)`, reordered with `.permute()`, sliced with NumPy-style indexing, repeated with `.repeat()`, and handed to `fill()`/`drain()` or to an `ObjectFifo`'s `to_stream`/`from_stream` directly. See the [taplib reference](../../../docs/api/taplib.md) for the full set of operations.
* [introduction](introduction): two pages with pictures that introduce `taplib`: [part 1](introduction/README.md) builds patterns (tiling, permuting, slicing, repeating), and [part 2](introduction/part2.md) covers hierarchical tiling, padding, what a DMA can walk, and chaining patterns with `gather()`/`scatter()` to follow data from the host to a core
* [per_tile](per_tile): an example design illustrating the order elements are accessed when tiling (`TensorAccessPattern.tile`)
* [tile_group](tile_group): an example design illustrating how to access every tile with a single DMA operation (the whole `TensorAccessPattern.tile` walk)
