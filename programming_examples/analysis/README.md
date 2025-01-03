# Programming Examples Analysis

## Setup

Install required packages:
```shell
pip install -r requirements.txt
```

## Generate Source Code (Python and MLIR) For Analysis

This will save the files in a directory named with the timestamp.
```bash
python3 collect_all.py
```
# Single Lines of Code

## Analyze for Single Lines of Code

This will analyze the files in the given collection directory and output a csv file with lines of code metrics
using both `pygount` and `radon`.

```bash
python3 loc_all.py -d <collection_dir> -o <loc_csv>
```

## Graph Single Lines of Code

This will produce two `png` files, one for absolute lines of code and the other for a percentage diff between the two design types.
```bash
python3 plot_loc.py -i <loc_csv>
```

# Halstead Metrics

## Analyze for Halstead Metrics

This will analyze the files in the given collection directory and output a csv file with Halstead metrics
using `radon`.

```bash
python3 halstead_all.py -d <collection_dir> -o <halstead_csv>
```

## Graph Halstead Metrics

This will produce a directory (`halstead_graphs`) of `png` files, one for each Halstead metric.
```bash
python3 plot_halstead.py -i <halstead_csv>
```

# MLIR Analysis

TODO