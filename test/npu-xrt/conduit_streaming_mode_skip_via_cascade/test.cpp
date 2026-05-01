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
// XRT host harness for the streaming-mode-skip via_cascade smoke.
//
// Single dispatch.  Input: 16 i32 (= 64 bytes = 1 AIE2 cascade vector =
// vector<16xi32>).  Expected output: byte-identical to input — the
// design wires
//
//   shim → memtile → tile(0,3) ─cascade─▶ tile(1,3) → memtile → shim
//
// with the cores doing nothing more than read+forward.  Any mismatch
// indicates the via_cascade route did not transport bytes correctly
// (e.g., Pass A regressed and stamped a `dma_repeat` that confused
// the downstream cascade lowering, or peano failed to emit the
// llvm.aie2.mcd.write.vec / llvm.aie2.scd.read.vec intrinsics, or
// firmware refused the cascade route).
//
// Mirrors the host driver convention from
// test/npu-xrt/conduit_canonicalize_channel_puts/homogeneous_repeat/test.cpp
// (cxxopts + xrt::bo + xrt::run boilerplate).

#include <bits/stdc++.h>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int N = 16;
constexpr int BUF_BYTES = N * static_cast<int>(sizeof(int32_t));

int main(int argc, const char *argv[]) {
  cxxopts::Options options("conduit_streaming_mode_skip_via_cascade");
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
      xrt::bo(device, BUF_BYTES, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out =
      xrt::bo(device, BUF_BYTES, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  // Deterministic input pattern: high bits identify the test (0xCA5CADE0)
  // so a stale or zeroed output is obvious; low nibble = element index.
  int32_t *buf_in = bo_in.map<int32_t *>();
  std::vector<int32_t> input_vec(N);
  for (int i = 0; i < N; i++) {
    input_vec[i] = static_cast<int32_t>(0xCA5CADE0u | static_cast<uint32_t>(i));
  }
  std::memcpy(buf_in, input_vec.data(), BUF_BYTES);

  int32_t *buf_out = bo_out.map<int32_t *>();
  std::memset(buf_out, 0, BUF_BYTES);

  void *buf_instr = bo_instr.map<void *>();
  std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = xrt::run(kernel);
  run.set_arg(0, opcode);
  run.set_arg(1, bo_instr);
  run.set_arg(2, instr_v.size());
  run.set_arg(3, bo_in);
  run.set_arg(4, bo_out);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  run.start();
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Reference: bytes round-trip unchanged through the cascade hop.
  int errors = 0;
  for (int i = 0; i < N; i++) {
    if (buf_out[i] != input_vec[i]) {
      if (errors < 16) {
        std::cout << "Mismatch at output[" << i << "]: expected 0x" << std::hex
                  << std::setw(8) << std::setfill('0')
                  << static_cast<uint32_t>(input_vec[i]) << " actual 0x"
                  << std::setw(8) << std::setfill('0')
                  << static_cast<uint32_t>(buf_out[i]) << std::dec << std::endl;
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
