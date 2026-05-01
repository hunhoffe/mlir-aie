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
// XRT host harness for the HYBRID Conduit fusion pipeline HW smoke
// (--conduit-fuse-spatial + --conduit-fuse-core-bodies-flag +
// --conduit-fuse-channels-flag together).  E2E companion sibling smokes:
//   test/npu-xrt/fuse_operators_convergent_npu/test.cpp (K=2 component)
//   test/npu-xrt/fuse_core_bodies_npu/test.cpp          (core-bodies component)
//   test/npu-xrt/fuse_channels_npu/test.cpp             (channels component)
//
// Inputs:
//   gate_in[j] = (j % 8)   — bf16 ramp
//   up_in[j]   = (j % 8)   — bf16 ramp (symmetric to gate; max 7)
// Outputs:
//   ext_out_a[j] = (gate_in[j] * up_in[j]) + 1.0   — max 7*7 + 1 = 50
//   ext_out_b[j] = (gate_in[j] * up_in[j]) + 2.0   — max 7*7 + 2 = 51
//
// Symmetric inputs: post-fusion arg reprojection on the producer side
// (gate vs up arg slot) does not affect the byte-equivalence check.
// Distinct +1.0 / +2.0 constants on the SINK side: channel-fusion mis-
// routing between the grouped ext_out_a / ext_out_b would surface as a
// byte mismatch (vs silent passthrough if both outputs computed the same
// value).
//
// Multi-invocation: the harness dispatches the merged kernel 4 times
// (mirroring sibling fuse_core_bodies_npu and fuse_channels_npu) so the
// smoke catches:
//   * Pattern E forward-chain mis-fusion (merged mul+sink core wedges on
//     the SECOND invocation — single-dispatch lit smokes hide this).
//   * Per-launch re-acquire stability for the channel-fused ext_out_a /
//     ext_out_b group (annotation-driven HW-channel folding consumed by
//     Pass C is exercised across multiple dispatch boundaries).
// Per-invocation verification (vs. final-only verification) localizes
// when a failure first surfaces.
//
// Max output values (50, 51) are well within bf16-exact integer range
// (≤ 256), so byte-equivalence to the host-computed reference is
// meaningful even with the truncate-toward-zero bf16 conversion below.

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
  cxxopts::Options options("fuse_hybrid_swiglu_npu");
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
  auto bo_out_a =
      xrt::bo(device, IO_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_out_b =
      xrt::bo(device, IO_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  // Symmetric bf16 ramp: gate[j] = up[j] = (j % 8), all bf16-exact.
  uint16_t *buf_gate = bo_gate.map<uint16_t *>();
  uint16_t *buf_up = bo_up.map<uint16_t *>();
  std::vector<uint16_t> input_vec(IO_LEN);
  for (int j = 0; j < IO_LEN; j++) {
    input_vec[j] = float_to_bf16(static_cast<float>(j % 8));
  }
  std::memcpy(buf_gate, input_vec.data(), IO_SIZE);
  std::memcpy(buf_up, input_vec.data(), IO_SIZE);

  uint16_t *buf_out_a = bo_out_a.map<uint16_t *>();
  uint16_t *buf_out_b = bo_out_b.map<uint16_t *>();

  void *buf_instr = bo_instr.map<void *>();
  std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_gate.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_up.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = xrt::run(kernel);
  run.set_arg(0, opcode);
  run.set_arg(1, bo_instr);
  run.set_arg(2, instr_v.size());
  // arg 3 / 4 / 5 / 6 = surviving runtime-sequence args of the merged
  // device, in source-order: ext_in_gate, ext_in_up, ext_out_a, ext_out_b.
  // Intermediates (inter_gate, inter_up, consume_gate, consume_up,
  // mul_inter, consume_mul) collapse during fusion and have no host-
  // visible runtime-sequence args.  Symmetric gate/up inputs make
  // verification independent of which arg-slot order post-fusion arg
  // reprojection emits for the producer side.
  run.set_arg(3, bo_gate);
  run.set_arg(4, bo_up);
  run.set_arg(5, bo_out_a);
  run.set_arg(6, bo_out_b);

  // Reference: out_a[j] = mul[j] + 1.0, out_b[j] = mul[j] + 2.0,
  // mul[j] = gate[j] * up[j].  All ref values are bf16-exact ints.
  std::vector<uint16_t> ref_a(IO_LEN, 0);
  std::vector<uint16_t> ref_b(IO_LEN, 0);
  for (int j = 0; j < IO_LEN; j++) {
    float a = bf16_to_float(input_vec[j]);
    float b = bf16_to_float(input_vec[j]);
    float m = a * b;
    ref_a[j] = float_to_bf16(m + 1.0f);
    ref_b[j] = float_to_bf16(m + 2.0f);
  }

  int total_errors = 0;
  for (int inv = 0; inv < NUM_INVOCATIONS; inv++) {
    if (verbosity >= 1)
      std::cout << "Invocation " << inv << ".\n";
    std::memset(buf_out_a, 0, IO_SIZE);
    std::memset(buf_out_b, 0, IO_SIZE);
    bo_out_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    run.start();
    ert_cmd_state r = run.wait();
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete on invocation " << inv
                << ". Returned status: " << r << "\n";
      return 1;
    }

    bo_out_a.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out_b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    int errors = 0;
    for (int j = 0; j < IO_LEN; j++) {
      if (buf_out_a[j] != ref_a[j]) {
        if (errors < 16) {
          std::cout << "Invocation " << inv << " mismatch at out_a[" << j
                    << "]: expected 0x" << std::hex << std::setw(4)
                    << std::setfill('0') << ref_a[j] << " actual 0x"
                    << std::setw(4) << std::setfill('0') << buf_out_a[j]
                    << std::dec << std::endl;
        }
        errors++;
      }
      if (buf_out_b[j] != ref_b[j]) {
        if (errors < 16) {
          std::cout << "Invocation " << inv << " mismatch at out_b[" << j
                    << "]: expected 0x" << std::hex << std::setw(4)
                    << std::setfill('0') << ref_b[j] << " actual 0x"
                    << std::setw(4) << std::setfill('0') << buf_out_b[j]
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
