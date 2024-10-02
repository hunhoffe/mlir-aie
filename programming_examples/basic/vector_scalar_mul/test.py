# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys
from aie.utils.xrt import setup_aie, execute as execute_on_aie
import aie.utils.test as test_utils


def main(opts):
    print("Running...\n")

    data_size = int(opts.size)
    vector_dtype = np.int16
    scalar_dtype = np.int32
    scale_factor = 3

    app = setup_aie(
        opts.xclbin,
        opts.instr,
        data_size,
        vector_dtype,
        1,
        scalar_dtype,
        data_size,
        vector_dtype,
    )
    input_vector = np.arange(1, data_size + 1, dtype=vector_dtype)
    input_factor = np.array([3], dtype=scalar_dtype)
    aie_output = execute_on_aie(app, input_vector, input_factor)
    ref = np.arange(1, data_size + 1, dtype=vector_dtype) * scale_factor

    # Copy output results and verify they are correct
    errors = 0
    if opts.verify:
        if opts.verbosity >= 1:
            print("Verifying results ...")
        e = np.equal(ref, aie_output)
        errors = np.size(e) - np.count_nonzero(e)

    if not errors:
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nError count: ", errors)
        print("\nFailed.\n")
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument("-s", "--size", required=True, dest="size", help="Vector size")
    opts = p.parse_args(sys.argv[1:])
    main(opts)
