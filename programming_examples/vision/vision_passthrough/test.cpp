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

#include "xrt/xrt_bo.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "OpenCVUtils.h"
#include "test_utils.h"

constexpr int channels = 4;
constexpr uint64_t testImageWidth = PASSTHROUGH_WIDTH;
constexpr uint64_t testImageHeight = PASSTHROUGH_HEIGHT;
constexpr uint64_t testImageSize = testImageWidth * testImageHeight;

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "xclbin,x", po::value<std::string>()->required(),
      "the input xclbin path")("image,p", po::value<std::string>(),
                               "the input image")(
      "outfile,o",
      po::value<std::string>()->default_value("passThroughOut_test.jpg"),
      "the output image")(
      "kernel,k", po::value<std::string>()->required(),
      "the kernel name in the XCLBIN (for instance PP_PRE_FD)")(
      "verbosity,v", po::value<int>()->default_value(0),
      "the verbosity of the output")(
      "instr,i", po::value<std::string>()->required(),
      "path of file containing userspace instructions to be sent to the LX6");
  po::variables_map vm;

  test_utils::parse_options(argc, argv, desc, vm);

  // Read the input image or generate random one if no input file argument
  // provided
  cv::Mat inImageGray;
  cv::String fileIn;
  if (vm.count("image")) {
    fileIn =
        vm["image"]
            .as<std::
                    string>(); //"/group/xrlabs/imagesAndVideos/images/minion128x128.jpg";
    initializeSingleGrayImageTest(fileIn, inImageGray);
  } else {
    fileIn = "RANDOM";
    inImageGray = cv::Mat(testImageHeight, testImageWidth, CV_8UC1);
    cv::randu(inImageGray, cv::Scalar(0), cv::Scalar(255));
  }

  cv::String fileOut =
      vm["outfile"].as<std::string>(); //"passThroughOut_test.jpg";
  printf("Load input image %s and run passThrough\n", fileIn.c_str());

  cv::resize(inImageGray, inImageGray,
             cv::Size(testImageWidth, testImageHeight));

  // Calculate OpenCV refence for passThrough
  cv::Mat outImageReference = inImageGray.clone();
  cv::Mat outImageTest(testImageHeight, testImageWidth, CV_8UC1);

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_sequence(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
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
  auto bo_inA = xrt::bo(device, inImageGray.total() * inImageGray.elemSize(),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_inB = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out =
      xrt::bo(device, (outImageTest.total() * outImageTest.elemSize()),
              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  uint8_t *bufInA = bo_inA.map<uint8_t *>();

  // Copyt cv::Mat input image to xrt buffer object
  memcpy(bufInA, inImageGray.data,
         (inImageGray.total() * inImageGray.elemSize()));

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

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
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
    ert_cmd_state r = run.wait();
    auto stop = std::chrono::high_resolution_clock::now();

    // check output and fetch results
    if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "Kernel did not complete. Returned status: " << r << "\n";
      return 1;
    }
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

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

  // Store result in cv::Mat
  uint8_t *bufOut = bo_out.map<uint8_t *>();
  memcpy(outImageTest.data, bufOut,
         (outImageTest.total() * outImageTest.elemSize()));

  // Compare to OpenCV reference
  int numberOfDifferences = 0;
  double errorPerPixel = 0;
  imageCompare(outImageTest, outImageReference, numberOfDifferences,
               errorPerPixel, true, false);
  printf("Number of differences: %d, average L1 error: %f\n",
         numberOfDifferences, errorPerPixel);

  cv::imwrite(fileOut, outImageTest);

  // Print Pass/Fail result of our test
  int res = 0;
  if (numberOfDifferences == 0) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }

  printf("Testing passThrough done!\n");
  return res;
}
