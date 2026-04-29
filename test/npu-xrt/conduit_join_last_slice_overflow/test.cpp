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
// XRT host harness for the JOIN last-slice BD-length overflow pattern
// (Task #48; e2e companion to the lit-only pin in
// test/Dialect/Conduit/conduit_to_dma_join_last_slice_bd_overflow.mlir).
//
// Topology: 2 producers → memtile JOIN → 1 shim consumer.
//   worker_a (compute(0,2)) writes float(j)        for j in [0, 64) into
//     each successive 64-bf16 buffer.
//   worker_b (compute(0,3)) writes float(64 + j)   for j in [0, 64) into
//     each successive 64-bf16 buffer.
//   memtile JOIN at offsets [0, 64] interleaves a worker_a buffer + a
//     worker_b buffer into one 128-bf16 memtile JOIN buffer.
//   Shim consumer reads 512 bf16 in one dispatch (= 4 successive 128-bf16
//     memtile JOIN slices).
//
// Expected output (each of 4 slices, 128 bf16):
//   output[s*128 + j]      = (bf16) j         for j in [0, 64)
//   output[s*128 + 64 + j] = (bf16) (64 + j)  for j in [0, 64)
// All values 0..127 are exactly representable in bf16 (sign + 8 exp + 7
// mantissa is enough for any integer in [0, 256)).
//
// Pre-fix conduit dispatch hangs (BD len 448 over memtile JOIN buffers
// of element count 128 → 4× overflow; firmware never produces a
// completion token).  Post-fix dispatch completes byte-equivalent to the
// reference computed below.

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

constexpr int N_SLICES = 4;
constexpr int PER_SOURCE = 64;
constexpr int PER_SLICE = 2 * PER_SOURCE;        // 128
constexpr int OUTPUT_LEN = N_SLICES * PER_SLICE; // 512

// bf16 = 2 bytes; we store/compare via raw uint16_t bit patterns since
// every value we ever write (integers 0..127) is bf16-exact.
constexpr int OUTPUT_SIZE = OUTPUT_LEN * static_cast<int>(sizeof(uint16_t));

static uint16_t float_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Truncate-toward-zero of the low 16 mantissa bits is exact for values
  // whose float32 representation has zero in those bits, which is true
  // for every integer in [0, 256).
  return static_cast<uint16_t>(bits >> 16);
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("conduit_join_last_slice_overflow");
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
  auto bo_output =
      xrt::bo(device, OUTPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  uint16_t *buf_output = bo_output.map<uint16_t *>();
  std::memset(buf_output, 0, OUTPUT_SIZE);

  void *buf_instr = bo_instr.map<void *>();
  std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = xrt::run(kernel);
  run.set_arg(0, opcode);
  run.set_arg(1, bo_instr);
  run.set_arg(2, instr_v.size());
  run.set_arg(3, bo_output);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  run.start();
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Reference: 4 slices, each = [0..63, 64..127] in bf16 bit-pattern.
  std::vector<uint16_t> ref(OUTPUT_LEN, 0);
  for (int s = 0; s < N_SLICES; s++) {
    for (int j = 0; j < PER_SOURCE; j++) {
      ref[s * PER_SLICE + j] = float_to_bf16(static_cast<float>(j));
      ref[s * PER_SLICE + PER_SOURCE + j] =
          float_to_bf16(static_cast<float>(PER_SOURCE + j));
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
