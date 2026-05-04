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
// XRT host harness for the COMPOSITION smoke
// `--conduit-fuse-core-bodies-flag --conduit-fuse-channels-flag`.  Sibling
// to the single-pass harnesses:
//   test/npu-xrt/fuse_core_bodies_npu/test.cpp
//   test/npu-xrt/fuse_channels_npu/test.cpp
//
// After `aie-combine-device same-tile=true` (gated by
// `--conduit-fuse-core-bodies-flag`) collapses devAdd + devMul into ONE
// merged device, the resulting runtime sequence presents a SINGLE pair of
// shim args to the host: (a_in for ext_in_add, o_out for ext_out_mul).
// The intermediate-channel runtime DMA configures (devAdd's @inter_add,
// devMul's @consume_add) are erased by `aie-combine-device` /
// `--conduit-fuse-core-bodies` since the intermediate is no longer
// shim-bound after merge.  So the host harness here matches the
// fuse_channels_npu shape (1 input + 1 output), NOT the fuse_core_bodies
// pre-merge shape.
//
// Inputs:
//   in_a[j] = (j % 16)  — bf16 ramp.
// Output:
//   out[j]  = (in_a[j] + 1.0) * 2.0.
//
// Multi-invocation: the harness dispatches the merged kernel 4 times
// (mirroring the single-pass siblings' NUM_INVOCATIONS=4) so the smoke
// catches multi-block / re-acquire failure modes.  Per-invocation
// verification (vs. final-only verification) localizes when the failure
// first surfaces — composition smokes specifically need this because a
// fuse-channels mis-grouping bug triggered by fuse-core-bodies' output IR
// could surface only on the second or later dispatch when the grouped
// channels re-acquire.
//
// Max output value = (15+1)*2 = 32 < 256, well within bf16-exact integer
// range, so byte-equivalence to the host-computed reference is meaningful
// even with truncate-toward-zero bf16 conversion below.

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
  cxxopts::Options options("fuse_channels_after_core_bodies_npu");
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
  auto bo_a =
      xrt::bo(device, IO_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out =
      xrt::bo(device, IO_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  // Bf16 ramp: in_a[j] = (j % 16), bf16-exact.
  uint16_t *buf_a = bo_a.map<uint16_t *>();
  std::vector<uint16_t> input_a(IO_LEN);
  for (int j = 0; j < IO_LEN; j++) {
    input_a[j] = float_to_bf16(static_cast<float>(j % 16));
  }
  std::memcpy(buf_a, input_a.data(), IO_SIZE);

  uint16_t *buf_out = bo_out.map<uint16_t *>();

  void *buf_instr = bo_instr.map<void *>();
  std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = xrt::run(kernel);
  run.set_arg(0, opcode);
  run.set_arg(1, bo_instr);
  run.set_arg(2, instr_v.size());
  // arg 3 / 4 = runtime-sequence args of the post-aie-combine-device
  // merged sequence, in source-order: a_in (ext_in_add), o_out
  // (ext_out_mul).  The intermediate-channel runtime DMA configures from
  // devAdd's @inter_add and devMul's @consume_add are erased by
  // `aie-combine-device` / `--conduit-fuse-core-bodies` (the intermediate
  // is no longer shim-bound after the two devices merge onto tile(0,2)).
  run.set_arg(3, bo_a);
  run.set_arg(4, bo_out);

  // Reference: out[j] = (in_a[j] + 1.0) * 2.0, all bf16-exact ints.
  std::vector<uint16_t> ref(IO_LEN, 0);
  for (int j = 0; j < IO_LEN; j++) {
    float a = bf16_to_float(input_a[j]);
    ref[j] = float_to_bf16((a + 1.0f) * 2.0f);
  }

  int total_errors = 0;
  for (int inv = 0; inv < NUM_INVOCATIONS; inv++) {
    if (verbosity >= 1)
      std::cout << "Invocation " << inv << ".\n";
    std::memset(buf_out, 0, IO_SIZE);
    bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    run.start();
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete on invocation " << inv
                << ". Returned status: " << r << "\n";
      return 1;
    }

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int errors = 0;
    for (int j = 0; j < IO_LEN; j++) {
      if (buf_out[j] != ref[j]) {
        if (errors < 16) {
          std::cout << "Invocation " << inv << " mismatch at output[" << j
                    << "]: expected 0x" << std::hex << std::setw(4)
                    << std::setfill('0') << ref[j] << " actual 0x"
                    << std::setw(4) << std::setfill('0') << buf_out[j]
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
