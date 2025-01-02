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

## Analyze for Lines of Code

This will analyze the files in the given collection directory and output a csv file with lines of code metrics
using both `pygount` and `radon`.

```bash
python3 loc_all.py -d <collection_dir> -o <out_csv>
```

## Graph Lines of Code

TODO