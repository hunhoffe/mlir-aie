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
// XRT host harness for the Pass C effectiveRepeat double-count NPU smoke.
// Pattern-matched on test/npu-xrt/conduit_per_launch_release_bd_pool/test.cpp.
//
// Source design lives in companion aie.mlir.  Post-fix expected output:
//   output[0..256)    == 0   (configure 1, BD walk 1, producer iter 0)
//   output[256..512)  == 1   (configure 1, BD walk 2, producer iter 1)
//   output[512..768)  == 2   (configure 2, BD walk 1, producer iter 2)
//   output[768..1024) == 3   (configure 2, BD walk 2, producer iter 3)
//
// Pre-fix (bug, push-queue=2 × BD-outer=2 = 4 fires per configure):
//   output[0..256)    == 2,  output[256..512)  == 3   (configure 1
//                                                      overwrites)
//   output[512..768)  == 6,  output[768..1024) == 7   (configure 2)
// → host catches via mismatch against the post-fix reference and prints
//   a hint identifying the expected pre-fix shape so the failure mode is
//   obvious in CI logs.

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

constexpr int SLICE = 256;        // ints per BD walk
constexpr int N_WALKS = 2;        // BD outer dim size per configure
constexpr int N_CONFIGURES = 2;   // configures per runtime sequence
constexpr int OUTPUT_LEN = N_CONFIGURES * N_WALKS * SLICE; // 1024

using DTYPE = int32_t;

constexpr int OUTPUT_SIZE = OUTPUT_LEN * sizeof(DTYPE);

constexpr DTYPE SENTINEL = static_cast<DTYPE>(0xDEADBEEF);

int main(int argc, const char *argv[]) {
  cxxopts::Options options("passc_effective_repeat_no_double_count_npu");
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

  // Initialize host buffer to a sentinel; any untouched bytes after the run
  // remain SENTINEL and we report them as a separate failure mode (would
  // suggest fewer fires than expected, not the double-fire bug).
  DTYPE *buf_output = bo_output.map<DTYPE *>();
  for (int i = 0; i < OUTPUT_LEN; i++)
    buf_output[i] = SENTINEL;

  void *buf_instr = bo_instr.map<void *>();
  memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

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

  // Post-fix reference (derived from first principles in companion
  // aie.mlir's header comment): each of the 4 quarter-buffers holds the
  // i32 broadcast of the producer's iter index that wrote it.  Producer
  // iter starts at 0 fresh per PDI load; FIFO order is FIFO-deterministic
  // regardless of producer pre-cycling.
  std::vector<DTYPE> ref(OUTPUT_LEN, 0);
  for (int q = 0; q < N_CONFIGURES * N_WALKS; q++) {
    DTYPE expected = static_cast<DTYPE>(q);
    for (int j = 0; j < SLICE; j++)
      ref[q * SLICE + j] = expected;
  }

  // Pre-fix (bug) reference: configure 1 sees iters 0..3, configure 2
  // sees iters 4..7; second BD walk per configure overwrites the first.
  std::vector<DTYPE> bug_ref(OUTPUT_LEN, 0);
  // Configure 1: walks 1+2 written, then walks 3+4 overwrite.
  for (int j = 0; j < SLICE; j++) bug_ref[0 * SLICE + j] = 2;
  for (int j = 0; j < SLICE; j++) bug_ref[1 * SLICE + j] = 3;
  // Configure 2: walks 1+2 written, then walks 3+4 overwrite.
  for (int j = 0; j < SLICE; j++) bug_ref[2 * SLICE + j] = 6;
  for (int j = 0; j < SLICE; j++) bug_ref[3 * SLICE + j] = 7;

  int errors = 0;
  int sentinel_holes = 0;
  int matches_bug_shape = 0;
  for (int i = 0; i < OUTPUT_LEN; i++) {
    if (buf_output[i] == SENTINEL)
      sentinel_holes++;
    if (buf_output[i] == bug_ref[i])
      matches_bug_shape++;
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

  std::cout << "\nError count: " << errors << " of " << OUTPUT_LEN << "\n";
  if (sentinel_holes > 0) {
    std::cout << "Sentinel-untouched entries: " << sentinel_holes
              << " — fewer BD fires than expected (different bug class).\n";
  }
  if (matches_bug_shape == OUTPUT_LEN) {
    std::cout << "Output matches the EXACT pre-fix double-count shape "
                 "(push-queue=2 × BD-outer=2 = 4 fires per configure). "
                 "Pass C `effectiveRepeat = std::max(...)` regression has "
                 "returned — see ConduitToDMALower.cpp around line 1300.\n";
  }
  std::cout << "\nFailed.\n\n";
  return 1;
}
