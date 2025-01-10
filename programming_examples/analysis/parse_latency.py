import argparse
import os


def parse_latency(collection_dir: str, output_file: str):
    iron_dir = os.path.join(collection_dir, "iron_result")
    iron_ext_dir = os.path.join(collection_dir, "iron_ext_result")

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

    with open(output_file, "w") as of:
        of.write(f"name,design,latency\n")
        for f in iron_design_files:
            if "stdout" in f:
                with open(os.path.join(iron_dir, f), "r") as fi:
                    result = fi.read()
                    timing_result = result.split("ParseHere")[1]
                    latency = timing_result.split("|")[1]
                    of.write(f"iron,{f.split('.')[0]},{latency}\n")
        for f in iron_ext_design_files:
            if "stdout" in f:
                with open(os.path.join(iron_ext_dir, f), "r") as fi:
                    result = fi.read()
                    timing_result = result.split("ParseHere")[1]
                    latency = timing_result.split("|")[1]
                    of.write(f"iron_ext,{f.split('.')[0]},{latency}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate CSV for latency")

    # Add arguments
    parser.add_argument(
        "-d", "--dir", help="Path to the input collection directory", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Path to the output file", default="latency.csv"
    )

    # Parse arguments
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: Collection directory {args.dir} is not a directory.")
        exit(-1)
    if os.path.exists(args.output):
        print(f"Error: Output file {args.output} already exists.")
        exit(-1)

    parse_latency(args.dir, args.output)


if __name__ == "__main__":
    main()
