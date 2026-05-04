//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// XRT host harness for the canon refuse-to-collapse-on-await predicate
// (puts-side variant) — companion to:
//   test/Dialect/Conduit/conduit_canon_no_collapse_on_token_true.mlir
//   test/npu-xrt/conduit_canon_no_collapse_on_link/test.cpp
//   test/npu-xrt/conduit_canon_no_collapse_on_gets_with_await/test.cpp
//
// Inputs:  64 bf16 ramp (input[j] = j, j = 0..63), all bf16-exact.
// Outputs: 256 bf16, four identical replicas of the input slice.
// Reference (computed inline below from spec, NOT lifted from compiler emit):
//   output[i*64 + j] = input[j]  for i in [0,4), j in [0,64)
//
// Pre-fix (canon over-collapses 4 puts whose chain shape is
// `[true, false]`): HW dispatch hangs (XRT timeout — same failure mode
// as the canon-link smoke and as the Llama prefill attn_scores GEMM).
// Post-fix: dispatch completes byte-equivalent to the reference
// computed below.

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

constexpr int N_REPEAT = 4;
constexpr int SLICE = 64;
constexpr int INPUT_LEN = SLICE;             // 64
constexpr int OUTPUT_LEN = N_REPEAT * SLICE; // 256

constexpr int INPUT_SIZE = INPUT_LEN * static_cast<int>(sizeof(uint16_t));
constexpr int OUTPUT_SIZE = OUTPUT_LEN * static_cast<int>(sizeof(uint16_t));

static uint16_t float_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Truncate-toward-zero of the low 16 mantissa bits is exact for
  // float32 values whose low 16 mantissa bits are zero — true for every
  // integer in [0, 256) (their float32 representation has zeros there).
  return static_cast<uint16_t>(bits >> 16);
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("conduit_canon_no_collapse_on_puts_with_await");
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

  // Deterministic bf16 ramp input: input[j] = bf16(j) for j in [0, 64).
  uint16_t *buf_input = bo_input.map<uint16_t *>();
  std::vector<uint16_t> input_vec(INPUT_LEN);
  for (int j = 0; j < INPUT_LEN; j++) {
    input_vec[j] = float_to_bf16(static_cast<float>(j));
  }
  std::memcpy(buf_input, input_vec.data(), INPUT_SIZE);

  uint16_t *buf_output = bo_output.map<uint16_t *>();
  std::memset(buf_output, 0, OUTPUT_SIZE);

  void *buf_instr = bo_instr.map<void *>();
  std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

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

  // Reference (derived from spec — 4 identical replicas of input ramp;
  // NOT lifted from compiler emit).
  std::vector<uint16_t> ref(OUTPUT_LEN, 0);
  for (int i = 0; i < N_REPEAT; i++) {
    for (int j = 0; j < SLICE; j++) {
      ref[i * SLICE + j] = input_vec[j];
    }
  }

  int errors = 0;
  for (int i = 0; i < OUTPUT_LEN; i++) {
    if (buf_output[i] != ref[i]) {
      if (errors < 16) {
        std::cout << "Mismatch at output[" << i << "]: expected 0x" << std::hex
                  << std::setw(4) << std::setfill('0') << ref[i] << " actual 0x"
                  << std::setw(4) << std::setfill('0') << buf_output[i]
                  << std::dec << std::endl;
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
