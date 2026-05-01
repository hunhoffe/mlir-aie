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
// XRT host harness for `--conduit-fuse-operators` convergent K=2 merge HW
// smoke (Sprint N+2 Phase 2; e2e companion to lit-only pin
//   test/Dialect/Conduit/fuse_operators_convergent_basic.mlir).
//
// Inputs:  two 64-element bf16 ramps, gate_in[j] = up_in[j] = (j % 16).
// Outputs: one 64-element bf16 buffer, output[j] = gate_in[j] * up_in[j].
//
// Symmetric inputs are deliberate — the post-fusion runtime sequence's
// surviving args (ext_in_gate, ext_in_up, ext_out_mul) may be reprojected
// to any positional order; symmetric inputs make the byte-equivalence check
// independent of which BO ends up bound to which arg slot.  Max output
// value is 15² = 225 < 256, well within bf16-exact integer range.

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

constexpr int SLICE = 64;
constexpr int IO_LEN = SLICE;
constexpr int IO_SIZE = IO_LEN * static_cast<int>(sizeof(uint16_t));

static uint16_t float_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Truncate-toward-zero is exact for integers in [0, 256).
  return static_cast<uint16_t>(bits >> 16);
}

static float bf16_to_float(uint16_t b) {
  uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("fuse_operators_convergent_npu");
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
  auto bo_gate =
      xrt::bo(device, IO_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_up =
      xrt::bo(device, IO_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out =
      xrt::bo(device, IO_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // Symmetric bf16 ramp: gate[j] = up[j] = (j % 16), all bf16-exact.
  uint16_t *buf_gate = bo_gate.map<uint16_t *>();
  uint16_t *buf_up = bo_up.map<uint16_t *>();
  std::vector<uint16_t> input_vec(IO_LEN);
  for (int j = 0; j < IO_LEN; j++) {
    input_vec[j] = float_to_bf16(static_cast<float>(j % 16));
  }
  std::memcpy(buf_gate, input_vec.data(), IO_SIZE);
  std::memcpy(buf_up, input_vec.data(), IO_SIZE);

  uint16_t *buf_out = bo_out.map<uint16_t *>();
  std::memset(buf_out, 0, IO_SIZE);

  void *buf_instr = bo_instr.map<void *>();
  std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_gate.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_up.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = xrt::run(kernel);
  run.set_arg(0, opcode);
  run.set_arg(1, bo_instr);
  run.set_arg(2, instr_v.size());
  // arg 3 / 4 / 5 = first three runtime-sequence args of the merged device,
  // in source-order: ext_in_gate, ext_in_up, ext_out_mul.  Symmetric inputs
  // mean the verification below holds regardless of which positional order
  // post-fusion arg reprojection actually emits.
  run.set_arg(3, bo_gate);
  run.set_arg(4, bo_up);
  run.set_arg(5, bo_out);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  run.start();
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Reference: elementwise multiply.  All ref values are bf16-exact ints.
  std::vector<uint16_t> ref(IO_LEN, 0);
  for (int j = 0; j < IO_LEN; j++) {
    float a = bf16_to_float(input_vec[j]);
    float b = bf16_to_float(input_vec[j]);
    ref[j] = float_to_bf16(a * b);
  }

  int errors = 0;
  for (int j = 0; j < IO_LEN; j++) {
    if (buf_out[j] != ref[j]) {
      if (errors < 16) {
        std::cout << "Mismatch at output[" << j << "]: expected 0x" << std::hex
                  << std::setw(4) << std::setfill('0') << ref[j] << " actual 0x"
                  << std::setw(4) << std::setfill('0') << buf_out[j] << std::dec
                  << std::endl;
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
