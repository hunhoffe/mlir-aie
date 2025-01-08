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

## Wily Halstead Metrics

Install with:
```bash
pip install wily
```

Now create a directory with the IRON source files, e.g. `wily_iron`.
Make a commit with these files.
Now overwrite those files with the corresponded IRON with extensions designs.
Make a second commit with these files.

Now you are ready to use wily.
Change directories to root of the git repository (this is important).

Build the wily index with `wily_iron` being the path to the directory:
```bash
wily build -a git -n 2 wily_iron
```

Now you can get metrics between the two commits with:
```bash
wily report -n 2 wily_iron
```

We want to report halstead metrics per file, so let's give it a shot.
