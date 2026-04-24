<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins> 2-D Array Transpose using AIE DMAs </ins>

This reference design can be run on a Ryzen™ AI NPU.

In the [design](./dma_transpose_placed.py), a 2-D array in a row-major layout is read from external memory to a compute tile with a transposed layout,
by using an implicit copy via the compute tile's Direct Memory Access (DMA). The data is read from and written to external memory through a shim tile.

This data movement transformation can be visualized as a map which shows the order the data the data is streamed (e.g., in transposed layout):
<p align="center">
  <img
    src="transpose_data.png">
    <h3 align="center"> Visualization of the Transpose Data Transformation for M=64, K=32. 
 </h3> 
</p>

The implicit copy is performed using the `ObjectFifo.forward()` function that specifies how input data arriving via `of_in` should be sent further via `of_out` by specifically leveraging a compute tile's (`AnyComputeTile`'s) DMA.

> This design was previously lowered using packet-switched flows for the underlying data movement (via the now-removed `--packet-sw-objFifos` aiecc flag). Packet routing now lives only on `conduit.create` ops via the `routing_mode = "packet"` attribute; restoring it for this example requires constructing the conduit IR directly via the low-level dialect API. The IRON Python path for setting per-channel `routing_mode` on this design is deferred to a follow-up sprint — until then, this example will lower with circuit routing.

## Design Versions
* [dma_transpose_placed.py](./dma_transpose_placed.py) shows a lower-level version of IRON, where constructors directly correspond to MLIR operations

The `object_fifo_link` operation used explicitly by `dma_transpose._placed.py` is described in more depth in [Section-2b](../../../programming_guide/section-2/section-2b/README.md/#object-fifo-link) of the programming guide.

To compile and run the design `dma_transpose_placed.py` for NPU:
```shell
make
make run
```

To generate a data visualization of the transpose (like that above), run:
```shell
make generate_access_map
```