//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>

#include "test_utils.h"
#include "xrt/xrt_bo.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
// ------------------------------------------------------
// Configure this to match your buffer data type
// ------------------------------------------------------
// using DATATYPE = std::uint8_t;
// using DATATYPE = std::uint32_t;
using DATATYPE = std::uint16_t;
#endif

const int scaleFactor = 3;

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  po::variables_map vm;
  test_utils::add_default_options(desc);

  test_utils::parse_options(argc, argv, desc, vm);
  int verbosity = vm["verbosity"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  constexpr bool VERIFY = true;
  constexpr int IN_VOLUME = VECTORSCALARMUL_SIZE;
  constexpr int OUT_VOLUME = IN_VOLUME;

  int IN_SIZE = IN_VOLUME * sizeof(DATATYPE);
  int OUT_SIZE = OUT_VOLUME * sizeof(DATATYPE) + trace_size;

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_sequence(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

  // set up the buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA =
      xrt::bo(device, IN_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inFactor = xrt::bo(device, 1 * sizeof(int32_t),
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_outC =
      xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer bo_inA
  DATATYPE *bufInA = bo_inA.map<DATATYPE *>();
  for (int i = 0; i < IN_VOLUME; i++)
    bufInA[i] = i + 1;

  // Initialize buffer bo_inFactor
  int32_t *bufInFactor = bo_inFactor.map<int32_t *>();
  *bufInFactor = (DATATYPE)scaleFactor;

  // Zero out buffer bo_outC
  DATATYPE *bufOut = bo_outC.map<DATATYPE *>();
  memset(bufOut, 0, OUT_SIZE);

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inFactor.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  /* RESUABLE TEST CODE START HERE */
  float npu_time_min = 9999999;
  float npu_time_max = 0;
  float npu_time_total = 0;
  int n_warmup_iterations = 100;
  int n_iterations = 1000;
  int num_iter = n_warmup_iterations + n_iterations;

  unsigned int opcode = 3;
  for (unsigned iter = 0; iter < num_iter; iter++) {

    // run the kernel
    auto start = std::chrono::high_resolution_clock::now();
    auto run =
        kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inFactor, bo_outC);
    ert_cmd_state r = run.wait();
    auto stop = std::chrono::high_resolution_clock::now();

    // check output and fetch results
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r << "\n";
      return 1;
    }
    bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }
  std::cout << std::endl
            << "ParseHere Avg NPU time: |" << npu_time_total / n_iterations
            << "|us. ParseHere" << std::endl;

  /* RESUABLE TEST CODE ENDS HERE */

  // Compare out to golden
  int errors = 0;
  if (verbosity >= 1) {
    std::cout << "Verifying results ..." << std::endl;
  }
  for (uint32_t i = 0; i < IN_VOLUME; i++) {
    int32_t ref = bufInA[i] * scaleFactor;
    int32_t test = bufOut[i];
    if (test != ref) {
      if (verbosity >= 1)
        std::cout << "Error in output " << test << " != " << ref << std::endl;
      errors++;
    } else {
      if (verbosity >= 1)
        std::cout << "Correct output " << test << " == " << ref << std::endl;
    }
  }

  if (trace_size > 0) {
    test_utils::write_out_trace(((char *)bufOut) + IN_SIZE, trace_size,
                                vm["trace_file"].as<std::string>());
  }

  // Print Pass/Fail result of our test
  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " mismatches." << std::endl
              << std::endl;
    std::cout << std::endl << "fail." << std::endl << std::endl;
    return 1;
  }
}
