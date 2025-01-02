import argparse
import json
import os

import subprocess


def calc_metrics(collection_dir: str, output_file: str):
    iron_dir = os.path.join(collection_dir, "iron")
    iron_ext_dir = os.path.join(collection_dir, "iron_ext")

    if not os.path.isdir(iron_dir):
        print(f"Error: Collection directory {iron_dir} does not exist.")
        exit(-1)
    if not os.path.isdir(iron_ext_dir):
        print(f"Error: Collection directory {iron_ext_dir} does not exist.")

    iron_design_files = [
        f for f in os.listdir(iron_dir) if os.path.isfile(os.path.join(iron_dir, f))
    ]
    iron_ext_design_files = [
        f
        for f in os.listdir(iron_ext_dir)
        if os.path.isfile(os.path.join(iron_ext_dir, f))
    ]
    if iron_design_files != iron_ext_design_files:
        print(f"IRON and IRON ext design names don't match perfectly.")

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

    with open(output_file, "w") as of:
        of.write(f"name,design,{','.join(halstead_keys)}\n")
        # Calculate halstead for radon
        metrics = get_radon_halstead_metrics(iron_dir)
        for src in metrics:
            of.write(
                f"{os.path.basename(src).removesuffix(".py")},iron,{','.join([str(metrics[src]["total"][key]) for key in halstead_keys])}\n"
            )
        metrics = get_radon_halstead_metrics(iron_ext_dir)
        for src in metrics:
            of.write(
                f"{os.path.basename(src).removesuffix(".py")},iron_ext,{','.join([str(metrics[src]["total"][key]) for key in halstead_keys])}\n"
            )


def get_radon_halstead_metrics(src_dir: str):
    # Calculate halstead for radon
    iron_result = subprocess.run(
        f"radon hal {src_dir} -j",
        shell=True,
        env=os.environ,
        capture_output=True,
    )
    iron_radon_metrics = json.loads(iron_result.stdout)
    print(iron_radon_metrics)

    if iron_result.returncode != 0:
        print(
            f"============= radon halstead for IRON failed with exit code: {iron_result.returncode}"
        )
        print(iron_result.stdout)
        print(iron_result.stderr)
        exit(-1)

    return iron_radon_metrics


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
