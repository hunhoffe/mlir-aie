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
// XRT host harness for the IRON `dma_repeat`-as-stride-0-outer-dim
// encoding HW smoke (Pass C v4 case (d) ground-truth anchor).  E2E
// companion to the lit-only pin
//   test/Dialect/Conduit/passc_effective_repeat_iron_stride_zero_encoding.mlir.
//
// Inputs:  one 64-element bf16 ramp, in[j] = (j % 16).
// Outputs: one 256-element bf16 buffer, four concatenated copies of `in`
//          (out[i] = in[i % 64]).
//
// Per-dispatch shim MM2S transfer count:
//   The `aie.mlir` runtime sequence configures the shim MM2S BD with
//   outer dim <size = 4, stride = 0> and {repeat_count = 4 : i32}.  The
//   IRON convention (per AIEDmaToNpu.cpp:385-388) is that a stride-0
//   outermost BD dim encodes `repeat_count` via BD shape — the BD does
//   NOT advance per iteration, so the same 64-byte block of `bo_in` is
//   sent into the L1 fifo four times per dispatch.  The compute tile's
//   identity-copy core therefore loops 4 times per dispatch, producing
//   4 × 64 = 256 bf16 to the shim S2MM @out_chan.
//
// Multi-invocation: NUM_INVOCATIONS=4 host dispatches (mirroring the
// sibling fuse_core_bodies_npu/test.cpp pattern).  Catches any latent
// state-leak between dispatches that single-shot would silently hide
// (push-queue counter wraps, BD chain residual fire, etc.).
//
// All output values fit inside [0, 16), well within bf16-exact integer
// range, so byte-equivalence to the host-computed reference is
// meaningful even with truncate-toward-zero bf16 conversion below.

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

constexpr int IO_LEN_IN = 64;
constexpr int IO_LEN_OUT = 256;
constexpr int IO_SIZE_IN = IO_LEN_IN * static_cast<int>(sizeof(uint16_t));
constexpr int IO_SIZE_OUT = IO_LEN_OUT * static_cast<int>(sizeof(uint16_t));
constexpr int NUM_INVOCATIONS = 4;

static uint16_t float_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Truncate-toward-zero is exact for integers in [0, 256).
  return static_cast<uint16_t>(bits >> 16);
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("passc_iron_stride_zero_encoding_npu");
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
      xrt::bo(device, IO_SIZE_IN, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out =
      xrt::bo(device, IO_SIZE_OUT, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  // Bf16 ramp: in[j] = (j % 16), bf16-exact.
  uint16_t *buf_in = bo_in.map<uint16_t *>();
  std::vector<uint16_t> input_vec(IO_LEN_IN);
  for (int j = 0; j < IO_LEN_IN; j++) {
    input_vec[j] = float_to_bf16(static_cast<float>(j % 16));
  }
  std::memcpy(buf_in, input_vec.data(), IO_SIZE_IN);

  uint16_t *buf_out = bo_out.map<uint16_t *>();

  void *buf_instr = bo_instr.map<void *>();
  std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;
  auto run = xrt::run(kernel);
  run.set_arg(0, opcode);
  run.set_arg(1, bo_instr);
  run.set_arg(2, instr_v.size());
  // arg 3 / 4 = runtime-sequence args in source-order: %in (64xbf16) and
  // %out (256xbf16).
  run.set_arg(3, bo_in);
  run.set_arg(4, bo_out);

  // Reference: four concatenated copies of `in`.  out[i] = in[i % 64].
  std::vector<uint16_t> ref(IO_LEN_OUT, 0);
  for (int i = 0; i < IO_LEN_OUT; i++) {
    ref[i] = input_vec[i % IO_LEN_IN];
  }

  int total_errors = 0;
  for (int inv = 0; inv < NUM_INVOCATIONS; inv++) {
    if (verbosity >= 1)
      std::cout << "Invocation " << inv << ".\n";
    std::memset(buf_out, 0, IO_SIZE_OUT);
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
    for (int i = 0; i < IO_LEN_OUT; i++) {
      if (buf_out[i] != ref[i]) {
        if (errors < 16) {
          std::cout << "Invocation " << inv << " mismatch at output[" << i
                    << "]: expected 0x" << std::hex << std::setw(4)
                    << std::setfill('0') << ref[i] << " actual 0x"
                    << std::setw(4) << std::setfill('0') << buf_out[i]
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
