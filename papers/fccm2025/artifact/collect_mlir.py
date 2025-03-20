import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from artifact.iron_examples import generate_iron_example_collection

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run examples to collect MLIR and results."
    )

    # Add arguments
    parser.add_argument(
        "-d", "--dir", help="Path to the example directory", required=True
    )

    # Parse arguments
    args = parser.parse_args()

    ie = generate_iron_example_collection()
    ie.collect_mlir_and_results(example_dir=args.dir, verbose=True)
