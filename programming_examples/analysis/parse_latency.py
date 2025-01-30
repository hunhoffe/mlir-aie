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

    results_dict = {}
    with open(output_file, "w") as of:
        of.write(f"name,design,latency\n")
        for f in iron_design_files:
            if "stdout" in f:
                with open(os.path.join(iron_dir, f), "r") as fi:
                    result = fi.read()
                    timing_result = result.split("ParseHere")[1]
                    latency = timing_result.split("|")[1]
                    of.write(f"iron,{f.split('.')[0]},{latency}\n")
                    results_dict[f.split(".")[0]] = {"iron": latency}
        for f in iron_ext_design_files:
            if "stdout" in f:
                with open(os.path.join(iron_ext_dir, f), "r") as fi:
                    result = fi.read()
                    timing_result = result.split("ParseHere")[1]
                    latency = timing_result.split("|")[1]
                    of.write(f"iron_ext,{f.split('.')[0]},{latency}\n")
                    results_dict[f.split(".")[0]]["iron_ext"] = latency

    differences = []
    iron_latencies = []
    iron_ext_latencies = []
    higher_ext = 0
    lower_ext = 0
    for design in results_dict.keys():
        iron_lat = float(results_dict[design]["iron"])
        iron_ext_lat = float(results_dict[design]["iron_ext"])
        iron_latencies.append(iron_lat)
        iron_ext_latencies.append(iron_ext_lat)
        if iron_lat > iron_ext_lat:
            lower_ext += 1
        else:
            higher_ext += 1
        if iron_lat == 0.0:
            iron_lat = 1.0
            print(f"Lat is zero: {design}")
        if iron_ext_lat == 0.0:
            iron_ext_lat == 1.0
            print(f"Ext lat is zero: {design}")
        percentage_difference = (
            abs(iron_lat - iron_ext_lat) / ((iron_lat + iron_ext_lat) / 2)
        ) * 100
        if percentage_difference > 5.0:
            print(f"HIGH DIFF: {design} {percentage_difference}")
        differences.append(percentage_difference)
    print(f"Percentage difference per design: {differences}")
    import statistics

    print(f"Average percentage difference: {statistics.mean(differences)}")
    print(iron_latencies)
    print(iron_ext_latencies)
    print(
        f"Average across designs: {statistics.mean(iron_latencies)} {statistics.mean(iron_ext_latencies)}"
    )
    print(f"Higher ext: {higher_ext}/{higher_ext + lower_ext}")


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
