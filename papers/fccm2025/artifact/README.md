# Artifact evaluation instructions

This folder includes software artifacts for the paper "Efficiency, Expressivity, and Extensibility in a Close-To-Metal NPU Programming Interface" accepted to [FCCM 2205](https://www.fccm.org/).

## Overview

### Source code of contributions to IRON

Source code is primarily found [here](../../../python/iron). This includes the ```Placer``` interface, which is integrated throughout the source code. Core placement classes are found [here](../../../python/iron/placers.py) and [here](../../../python/iron/placeable.py). ```taplib``` (and useful extensions) are [here](../../../python/helpers/taplib), with an instructional notebook [here](../../../programming_examples/basic/tiling_exploration/introduction/).

### Environment Setup

Install IRON according to the quickstart instructions described in the primary repo [README](../../../README.md). Since the extensions to IRON have already been accepted into the code base, this will install Python wheels that contain the source code linked to above.

**We assume all commands below are run with the environment variables sourced according to the quickstart script and within the Python env, ```ironenv```, activated.**

### Using the extensions

In the general sense (not for replicating paper results) there are examples using the new IRON API found throughout the [```programming examples```](../../../programming_examples/), generally denoted with the name ending in ```_iron``` or without a suffix (as opposed to the ```_placed``` designs, which generally refer to designs written with IRON without extensions). Instructions for running these examples are found in their local directories.

## Replicating results from the paper

Below is a summary of claims from the paper and how to verify them.

The primary results to verify from the paper are:
1. The source code (the primary artifact)
2. Functionality of examples designs from Figure 2
3. Functionality of code snippets for ```taplib``` from Section V.5, and generation of graphics in Figure 3

The primary results to verify from the evaluation are:
4. Table I and Table II - Features found in example designs
5. Figure 4 - Average percent decrease to Single Lines of Code (SLOC)
6. Figure 5 - Differences in Halstead vocabulary and effort
7. MLIR analysis from Section VI.B
8. Performance results from the last paragraph in Section VI.B
9. Functionality of the ```SequentialPlacer``` described in Section VI.C
10. Functionality of the ```TensorTiler2D``` described in Section VI.D

Many of these results rely on a "corpus of example designs" - these designs are similar to those in [```programming_examples```](../../../programming_examples/) but organized and simplified specifically for this publication (e.g., we remove code targeting other devices, for debugging, etc. so analysis reflects 'core' design characteristics). The set of designs discussed in the paper are found in the [```example_designs```](./example_designs) directory.

## Artifact evaluation instructions

### 1. Source Code

Source code is primarily found [here](../../../python/iron). This includes the ```Placer``` interface, which is integrated throughout the source code. Core placement classes are found [here](../../../python/iron/placers.py) and [here](../../../python/iron/placeable.py). ```taplib``` (and useful extensions) are [here](../../../python/helpers/taplib).

### 2. Figure 2 Designs

The designs in Figure 2 are pared down examples from [```example_designs/iron/MSAdd.py```](,/example_designs/iron/MSAdd.py) and [```example_designs/iron_ext/MSAdd.py```](./example_designs/iron_ext/MSAdd.py). We include nearly identical versions (but with some additional imports/constant defintions omitted from the paper due to space) in [```fig2```](fig2) directory. These are functional examples - you can use them to generate MLIR (while we leave full functionality tests to the examples in ```example_designs```).

We can verify the Python yields valid MLIR of the ```mlir-aie``` dialect by running:
```bash
python fig2/fig2_iron.py
python fig2/fig2_iron_ext.py
```

If these were not valid designs (e.g., not supported by the API) they would throw errors and would not successfully generate MLIR.
Example MLIR generation on our system (for comparison) is found at [```fig2/ext.mlir```](./fig2/ext.mlir) and [```fig2/orig.mlir```](./fig2/orig.mlir).

### 3. ```taplib`` Code Snippets and Figure 3

The code snippets for ```taplib``` found in Section V are found in [```taplib/snippets.py```](taplib/snippets.py) and can be run (to check validity) with:
```bash
python taplib/snippets.py
```

To generate the graphics from Figure 3, run:
```bash
python taplib/fig3.py
```

This will create ```tap00.py``` and ```taps0.py```, which have been combined to form Figure 3.

### 4. Table I and Table II - Features found in example designs

The only way to verify examples are actually as described is to manually review the desings in [```example_designs```](example designs). However, there is a script to generate the SLOC counts in Table 1.

To generate the SLOC counts for all example designs:
```bash
pip install -r requirements.txt
python sloc_all.py -d example_designs
```

This will generate a file, ```loc.csv``` using ```pygount``` and ```radon``` to calculate the SLOC. The SLOC reported in Table I is from ```pygount```, corresponding to the columns ```pygount_loc_baseline``` and ```pygount_loc_ext``` in the csv file.

### 5. Figure 4 - Average percent decrease to Single Lines of Code (SLOC)

A graph can be generated from the previous step using ```loc.csv``` with the command:
```bash
python plot_loc.py -i loc.csv
```

This will print out statistics (average percent decrease being around 25%, average difference in SLOC is ~30 lines) and produce a ```percentage_loc.png```, which is what is included in the paper.

### 6. Figure 5 - Differences in Halstead vocabulary and effort

Reproducing Figure 5 happens in two stages: calculate Halstead metrics, and plot them.
  
To calculate the Halstead metrics using ```radon```, run:
```bash
pip install -r requirements.txt
python halstead_all.py -d example_designs/
```

This will produce a ```halstead.csv``` file. To produce the graphs in Figure 5, run:
```bash
python plot_halstead.py -i halstead.csv
```

This will produce a graph for effort and vocabulary (in separate files but otherwise idential to the joint graph in the paper) in a ```halstead_graphs``` directory.

### 7. MLIR analysis from Section VI.B

MLIR analysis between corresponding examples is done through a mix of scripts and manual inspections. 

The steps are as follows:
* Generate this MLIR: The script to generate the MLIR also runs all of the examples, so **it can take some time** and the results are also used in the next step.
  ```bash
  python collect_mlir.py -d example_designs/
  ```
  MLIR for all the example designs is copied into ```example_designs/iron_ext_mlir/``` and ```example_designs/iron_mlir``` by the [```collect_mlir.py```](./collect_mlir.py) script.

  Note: This is the first step in the verification process so far that requires running examples on the NPU. This script works by copying the IRON and IRON(ext) files into the local [```programming_examples```](./programming_examples/) tree, which contains the Makefiles and other associated files needed to build, run, and collect performance data on all example designs.
  * TODO

### 8. Performance results from the last paragraph in Section VI.B
  TODO

### 9. Functionality of the ```Placer``` described in Section VI.C

  Verification that the ```SequentialPlacer``` is successfully used in examples can be found from:
  * Reviewing the ```SequentialPlacer``` source code, found at [```SequentialPlacer```](../../../python/iron/placers.py).
  * Verifying that examples use the ```SequentialPlacer```:
    This can be done through analyzing the source code in the [```example_designs/iron_ext```](./example_designs/iron_ext/) directory. As a quick hack, to count the examples that use the ```SequentialPlacer```, run: TODO
    ```bash
    find . -iname example_designs/iron_ext/*.py -exec bash -c 'grep -r SequentialPlacer {}' \;
    ```
  * Running the examples that use the ```SequentialPlacer``` successfully to verify placements are valid: (this is already done in step 8 above)

### 10. Functionality of the ```TensorTiler2D``` described in Section VI.D

  Verification of the use of the ```TensorTiler2D``` is successfully used in examples can be found from:
  * Reviewing the ```TensorTiler2D``` source code, found at [```TensorTiler2D```](../../../python/helpers/taplib/tensortiler2d.py)
  * Verifying that examples use the ```TensorTiler2D```:
    This can be done through analyzing the source code in the [```example_designs/iron_ext```](./example_designs/iron_ext/) directory. As a quick hack, to count the examples that use the ```TensorTiler2D```, run: TODO
    ```bash
    find . -iname example_designs/iron_ext/*.py -exec bash -c 'grep -r TensorTiler2D {}' \;
    ```
  * Running the examples that use the ```TensorTiler2D``` successfully to verify the data movements are valid: (this is already done in step 8 above)