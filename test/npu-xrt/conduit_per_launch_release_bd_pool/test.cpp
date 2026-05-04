//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// XRT host harness for the per-launch release BD-pool pattern (Task #15
// design; pattern-matched on test/npu-xrt/dmabd_task_queue/test.cpp).
//
// Inputs:  5120 int32 ramp (input[i] = i, i = 0..5119)
// Outputs: 1024 int32 sum-reduction
// Per output slice [i] (i = 0..3, slice size = 256):
//   output[i*256 + j] = sum_{k=0..4} input[(i*5 + k)*256 + j]
//
// Shared by stateful.lit (PASS expected) and (post-Path-C) the
// conduit_path_b_xfail.lit rewrite.

#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int SLICE = 256;
constexpr int N_OUT_SLICES = 4;
constexpr int N_IN_PER_OUT = 5;
constexpr int INPUT_LEN = N_OUT_SLICES * N_IN_PER_OUT * SLICE; // 5120
constexpr int OUTPUT_LEN = N_OUT_SLICES * SLICE;               // 1024

using DTYPE = int32_t;

constexpr int INPUT_SIZE = INPUT_LEN * sizeof(DTYPE);
constexpr int OUTPUT_SIZE = OUTPUT_LEN * sizeof(DTYPE);

int main(int argc, const char *argv[]) {
  cxxopts::Options options("conduit_per_launch_release_bd_pool");
  test_utils::add_default_options(options);

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  std::string Node = vm["kernel"].as<std::string>();
  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << Node << "\n";

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input =
      xrt::bo(device, INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_output =
      xrt::bo(device, OUTPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  // Deterministic int32 ramp input.
  DTYPE *buf_input = bo_input.map<DTYPE *>();
  std::vector<DTYPE> input_vec(INPUT_LEN);
  for (int i = 0; i < INPUT_LEN; i++) {
    input_vec[i] = i;
  }
  memcpy(buf_input, input_vec.data(), INPUT_SIZE);

  DTYPE *buf_output = bo_output.map<DTYPE *>();
  memset(buf_output, 0, OUTPUT_SIZE);

  void *buf_instr = bo_instr.map<void *>();
  memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = xrt::run(kernel);
  run.set_arg(0, opcode);
  run.set_arg(1, bo_instr);
  run.set_arg(2, instr_v.size());
  run.set_arg(3, bo_input);
  run.set_arg(4, bo_output);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  run.start();
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Reference: per output slice i, sum the 5 input slices that feed it.
  std::vector<DTYPE> ref(OUTPUT_LEN, 0);
  for (int i = 0; i < N_OUT_SLICES; i++) {
    for (int j = 0; j < SLICE; j++) {
      DTYPE acc = 0;
      for (int k = 0; k < N_IN_PER_OUT; k++) {
        acc += input_vec[(i * N_IN_PER_OUT + k) * SLICE + j];
      }
      ref[i * SLICE + j] = acc;
    }
  }

  int errors = 0;
  for (int i = 0; i < OUTPUT_LEN; i++) {
    if (buf_output[i] != ref[i]) {
      if (errors < 16) {
        std::cout << "Mismatch at output[" << i << "]: expected " << ref[i]
                  << " actual " << buf_output[i] << std::endl;
      }
      errors++;
    }
  }

  if (errors == 0) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nError count: " << errors << "\n";
  std::cout << "\nFailed.\n\n";
  return 1;
}
