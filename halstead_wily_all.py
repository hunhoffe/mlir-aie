import argparse
import json
import os

import subprocess


def calc_metrics(collection_dir: str, output_file: str):
    if not os.path.isdir(collection_dir):
        print(f"Error: Collection directory {collection_dir} does not exist.")
        exit(-1)

    design_files = [
        f for f in os.listdir(collection_dir) if os.path.isfile(os.path.join(collection_dir, f))
    ]

    halstead_keys = [
        "h1",
        "h2",
        "N1",
        "N2",
        "vocabulary",
        "length",
        "calculated_length",
        "volume",
        "difficulty",
        "effort",
        "time",
        "bugs",
    ]

    # Generate HTML
    for design in design_files:
        run_halstead = subprocess.run(
            f"wily report \'iron/{design}\' -n 2 halstead.h1 halstead.h2 halstead.N1 halstead.N2 halstead.volume halstead.complexity halstead.length halstead.effort halstead.difficulty --format HTML -o \'iron_out/{design}\'",
            shell=True,
            env=os.environ,
            capture_output=True,
        )
        print(run_halstead.stdout)
        print(run_halstead.stderr)

    with open(output_file, "w") as of:
        # Write headers
        of.write(f"name,design,tool,{','.join(halstead_keys)}\n")



def main():
    parser = argparse.ArgumentParser(description="Halstead Metric Calculation")

    # Add arguments
    parser.add_argument(
        "-d", "--dir", help="Path to the input collection directory", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Path to the output file", default="halstead.csv"
    )

    # Parse arguments
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: Collection directory {args.dir} is not a directory.")
        exit(-1)
    if os.path.exists(args.output):
        print(f"Error: Output file {args.output} already exists.")
        exit(-1)

    calc_metrics(args.dir, args.output)


if __name__ == "__main__":
    main()
