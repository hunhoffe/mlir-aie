import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from analysis.iron_examples import generate_iron_example_collection

if __name__ == "__main__":
    ie = generate_iron_example_collection()
    print("=========== Running IRON ===============")
    ie.run_all()
    print("========================================")
    print("=========== Running IRON ext ===========")
    ie.run_all(True)
