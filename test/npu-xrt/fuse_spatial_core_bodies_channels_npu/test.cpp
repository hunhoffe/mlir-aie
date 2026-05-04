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
// XRT host harness for the FULL HYBRID COMPOSITION `--conduit-fuse-spatial`
// + `--conduit-fuse-core-bodies` + `--conduit-fuse-channels` HW smoke.
// Sibling-pair to single-pass smokes:
//   test/npu-xrt/fuse_operators_basic_npu/test.cpp
//   test/npu-xrt/fuse_core_bodies_npu/test.cpp
//   test/npu-xrt/fuse_channels_npu/test.cpp
// And companion to the two-pass composition fixtures:
//   test/npu-xrt/fuse_channels_after_core_bodies_npu/test.cpp  (compose-A)
//   test/npu-xrt/fuse_spatial_and_channels_npu/test.cpp        (compose-B)
//
// Inputs:  one 64-element bf16 ramp, in[j] = (j % 16).
// Outputs: TWO 64-element bf16 buffers:
//   out_mul[j] = (in[j] + 1.0) * 2.0   (Section A of devMul's core,
//                                       merged into the post-fuse-core-
//                                       bodies single-core body)
//   out_aux[j] = 7.0 (constant)        (Section B of devMul's core,
//                                       merged into the post-fuse-core-
//                                       bodies single-core body)
//
// Multi-invocation: the harness dispatches the merged kernel 4 times so the
// smoke catches the merged-device merged-core ELF deadlock/misroute class —
// Bug A/B-style failures that compile cleanly but fail mid-run on hardware,
// any Pattern E forward-chain mis-fusion across the merged body, and any
// fuse-channels mis-grouping that swaps DMA channels between the two
// producer-side outputs (which would surface as out_mul-destined bytes
// landing in bo_aux or vice versa).  Per-invocation verification (vs.
// final-only verification) localizes when the failure first surfaces.
//
// Max output value across both buffers = max((15+1)*2, 7) = 32 < 256, well
// within bf16-exact integer range, so byte-equivalence to the host-computed
// reference is meaningful even with truncate-toward-zero bf16 conversion
// below.

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
constexpr int NUM_INVOCATIONS = 4;

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
  cxxopts::Options options("fuse_spatial_core_bodies_channels_npu");
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
  auto bo_in =
      xrt::bo(device, IO_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out_mul =
      xrt::bo(device, IO_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out_aux =
      xrt::bo(device, IO_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // Bf16 ramp: in[j] = (j % 16), bf16-exact.
  uint16_t *buf_in = bo_in.map<uint16_t *>();
  std::vector<uint16_t> input_vec(IO_LEN);
  for (int j = 0; j < IO_LEN; j++) {
    input_vec[j] = float_to_bf16(static_cast<float>(j % 16));
  }
  std::memcpy(buf_in, input_vec.data(), IO_SIZE);

  uint16_t *buf_out_mul = bo_out_mul.map<uint16_t *>();
  uint16_t *buf_out_aux = bo_out_aux.map<uint16_t *>();

  void *buf_instr = bo_instr.map<void *>();
  std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = xrt::run(kernel);
  run.set_arg(0, opcode);
  run.set_arg(1, bo_instr);
  run.set_arg(2, instr_v.size());
  // arg 3 / 4 / 5 = surviving runtime-sequence args of the post-aie-combine-
  // device merged sequence, in source-order across devAdd then devMul:
  //   - ext_in_add  (devAdd's @add_seq %ain)
  //   - ext_out_mul (devMul's @mul_seq %mout)
  //   - ext_out_aux (devMul's @mul_seq %maux)
  // After --conduit-fuse-operators pairs and erases the intermediate
  // channel pair (inter_add / consume_add), only the external-facing
  // input/output args survive — same erase pattern as compose-B sibling,
  // which this fixture clones for the bf16 reference compute.
  run.set_arg(3, bo_in);
  run.set_arg(4, bo_out_mul);
  run.set_arg(5, bo_out_aux);

  // Reference for Section A: out_mul[j] = (in[j] + 1.0) * 2.0, bf16-exact.
  std::vector<uint16_t> ref_mul(IO_LEN, 0);
  for (int j = 0; j < IO_LEN; j++) {
    float a = bf16_to_float(input_vec[j]);
    ref_mul[j] = float_to_bf16((a + 1.0f) * 2.0f);
  }
  // Reference for Section B: out_aux[j] = 7.0 for all j, bf16-exact.
  uint16_t ref_aux_val = float_to_bf16(7.0f);

  int total_errors = 0;
  for (int inv = 0; inv < NUM_INVOCATIONS; inv++) {
    if (verbosity >= 1)
      std::cout << "Invocation " << inv << ".\n";
    std::memset(buf_out_mul, 0, IO_SIZE);
    std::memset(buf_out_aux, 0, IO_SIZE);
    bo_out_mul.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out_aux.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    run.start();
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete on invocation " << inv
                << ". Returned status: " << r << "\n";
      return 1;
    }

    bo_out_mul.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out_aux.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int errors = 0;
    for (int j = 0; j < IO_LEN; j++) {
      if (buf_out_mul[j] != ref_mul[j]) {
        if (errors < 16) {
          std::cout << "Invocation " << inv << " out_mul mismatch at [" << j
                    << "]: expected 0x" << std::hex << std::setw(4)
                    << std::setfill('0') << ref_mul[j] << " actual 0x"
                    << std::setw(4) << std::setfill('0') << buf_out_mul[j]
                    << std::dec << std::endl;
        }
        errors++;
      }
      if (buf_out_aux[j] != ref_aux_val) {
        if (errors < 16) {
          std::cout << "Invocation " << inv << " out_aux mismatch at [" << j
                    << "]: expected 0x" << std::hex << std::setw(4)
                    << std::setfill('0') << ref_aux_val << " actual 0x"
                    << std::setw(4) << std::setfill('0') << buf_out_aux[j]
                    << std::dec << std::endl;
        }
        errors++;
      }
    }
    total_errors += errors;
  }

  if (total_errors == 0) {
    std::cout << "\nPASS!\n\n";
    return 0;
  }
  std::cout << "\nError count: " << total_errors << "\n";
  std::cout << "\nFailed.\n\n";
  return 1;
}
