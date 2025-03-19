# Artifact evaluation instructions

This folder includes software artifacts for the paper "Efficiency, Expressivity, and Extensibility in a Close-To-Metal NPU Programming Interface" accepted to [FCCM 2205](https://www.fccm.org/).

Artifact evaluation is meant only to be done on the ```fccm-2025-artifact``` branch.

## Overview

### Source Code

Source code is primarily found [here](../../../python/iron). This includes the ```Placer``` interface, which is integrated throughout the source code. Core placement classes are found [here](../../../python/iron/placers.py) and [here](../../../python/iron/placeable.py). ```taplib``` (and useful extensions) are  [here](../../../python/helpers/taplib), with an instructional notebook [here](../../../programming_examples/basic/tiling_exploration/introduction/).

### Environment Setup

Make sure you have checked out the ```fccm-2025-artifact``` branch.

Install IRON according to the quickstart instructions described in the primary repo [README](../../../README.md). Since the extensions to IRON have already been accepted into the code base, this will install Python wheels that contain the source code linked to above, a user is ready to use the extensions.

### Using the extensions

In the general sense (not for replicating paper results) there are examples using the new IRON API found throughout the [```programming examples```](../../../programming_examples/), generally denoted with the name ending in ```_iron``` or without a suffix (as opposed to the ```_placed``` designs, which generally refer to designs written with IRON without extensions). Instructions for running these examples are found in their local directories.

## Replicating results from the paper

Below is a summary of claims from the paper and how to verify them.

The primary results to verify from the paper are:
1. The source code
2. Functionality of examples designs from Figure 2
3. Functionality of code snippets for ```taplib``` from Section V.5, and generation of graphics in Figure 3

The primary results to verify from the evaluation are:
4. Table II - Features found in example designs
5. Figure 4 - Average percent decrease to Single Lines of Code (SLOC)
6. Figure 5 - Differences in Halstead vocabulary and effort
7. Performance results from the last paragraph in Section VI.B
8. Functionality of the ```Placer``` described in Section VI.C
9. Functionality of the ```TensorTiler2D``` described in Section VI.D

Many of these results rely on a "corpus of example designs" - these designs were based on the existing set of [```programming_examples```](../../../programming_examples/). However, to remove unnecessary configuration details, remove designs that were too similar, and remove tracing artifacts and debugging setup from designs that would distract from the analysis, we use a separately currated set of designs found [here](example_designs).

## Artifact evaluation instructions

1. **Source Code**
  Source code is primarily found [here](../../../python/iron). This includes the ```Placer``` interface, which is integrated throughout the source code. Core placement classes are found [here](../../../python/iron/placers.py) and [here](../../../python/iron/placeable.py). ```taplib``` (and useful extensions) are  [here](../../../python/helpers/taplib), with an instructional notebook [here](../../../programming_examples/basic/tiling_exploration/introduction/).

2. **Figure 2 Designs**
  The designs in Figure 2 are pared down examples from the [```example_designs```](example designs). We include nearly identical versions (but with some additional imports/constant defintions omitted from the paper due to space) in [```fig2```](fig2) directory. These are functional examples - you can run them and verify the results.

  For IRON without contributions:
  ```bash
  cd fig2
  make
  make run
  ```

  For IRON with contributions:
  ```bash
  cd fig2
  make
  make run
  ```

3. **