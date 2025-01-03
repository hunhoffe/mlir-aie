import argparse
import os

from difflib import ndiff, restore


def mlir_diffs(collection_dir: str, output_file: str):
    iron_dir = os.path.join(collection_dir, "iron_mlir")
    iron_ext_dir = os.path.join(collection_dir, "iron_ext_mlir")

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

    same_count = 0
    diff_count = 0
    for design in iron_design_files:
        print(f"==================== Design: {design}")
        mlir_file = os.path.join(iron_dir, design)
        mlir_ext_file = os.path.join(iron_ext_dir, design)

        with open(mlir_file, "r") as f:
            mlir_code = f.read()
        with open(mlir_ext_file, "r") as f:
            mlir_ext_code = f.read()

        # Uniform constants for while(true)
        mlir_code = mlir_code.replace("4294967295", "9223372036854775807")
        mlir_code_lines = mlir_code.splitlines(keepends=True)
        mlir_ext_code = mlir_ext_code.replace("4294967295", "9223372036854775807")
        mlir_ext_code_lines = mlir_ext_code.splitlines(keepends=True)

        for i, line in enumerate(mlir_ext_code_lines):
            # Normalize offsets for object fifo link
            if "aie.objectfifo.link [" in line:
                mlir_ext_code_lines[i] = line.replace("[0]", "[]")

        comparison = ndiff(mlir_code_lines, mlir_ext_code_lines)

        one = []
        two = []
        for d in comparison:
            if d.startswith("-"):
                one.append(d[1:])
            elif d.startswith("+"):
                two.append(d[1:])

        updated_one = []
        for line in one:
            if line in two:
                # Declarations. Doesn't matter unless incorrect, which would be caught be functionality tests.
                if (
                    "func.func private @" in line
                    or "aiex.dma_start_task" in line
                    or " = aie.tile("
                    or "aie.objectfifo.link [" in line
                    or "func.func @" in line
                ):
                    two.remove(line)
                else:
                    updated_one.append(line)
            else:
                updated_one.append(line)

        if len(updated_one) > 0 or len(two) > 0:
            print("FOUND MLIR MISMATCH")
            print(f"From one: {updated_one}")
            print(f"From two: {two}")
            diff_count += 1
        else:
            same_count += 1

    total = len(iron_design_files)
    print("++++++++++++++ Summary ++++++++++++++++++++")
    print(f"Same: {same_count}/{total}")
    print(f"Different: {diff_count}/{total}")
    assert same_count + diff_count == total


def main():
    parser = argparse.ArgumentParser(description="MLIR Diff Calculation")

    # Add arguments
    parser.add_argument(
        "-d", "--dir", help="Path to the input collection directory", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Path to the output file", default="mlir.csv"
    )

    # Parse arguments
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: Collection directory {args.dir} is not a directory.")
        exit(-1)
    if os.path.exists(args.output):
        print(f"Error: Output file {args.output} already exists.")
        exit(-1)

    mlir_diffs(args.dir, args.output)


if __name__ == "__main__":
    main()
