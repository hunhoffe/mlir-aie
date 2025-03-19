import argparse
import os

from pygount import SourceAnalysis
from radon.raw import analyze


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

    with open(output_file, "w") as of:
        of.write(
            "name,pygount_loc_baseline,pygount_loc_ext,radon_loc_baseline,radon_loc_ext\n"
        )
        for d in iron_design_files:
            iron_design = os.path.join(iron_dir, d)
            iron_ext_design = os.path.join(iron_ext_dir, d)

            # Calculate loc for radon
            with open(iron_design, "r") as f:
                code = f.read()
                radon_loc_iron = analyze(code).sloc
            with open(iron_ext_design, "r") as f:
                code = f.read()
                radon_loc_iron_ext = analyze(code).sloc

            # Calculate loc for pygount
            pygount_loc_iron = SourceAnalysis.from_file(iron_design, "pygount")._code
            pygount_loc_iron_ext = SourceAnalysis.from_file(
                iron_ext_design, "pygount"
            )._code

            design_name = d.removesuffix(".py")
            of.write(
                f"{design_name},{pygount_loc_iron},{pygount_loc_iron_ext},{radon_loc_iron},{radon_loc_iron_ext}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="LoC Metric Calculation")

    # Add arguments
    parser.add_argument(
        "-d", "--dir", help="Path to the input collection directory", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Path to the output file", default="loc.csv"
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
