import argparse
import os
import subprocess
import pandas as pd
from bs4 import BeautifulSoup
import re


def calc_metrics(collection_dir: str, output_file: str):
    if not os.path.isdir(collection_dir):
        print(f"Error: Collection directory {collection_dir} does not exist.")
        exit(-1)

    design_files = [
        f
        for f in os.listdir(collection_dir)
        if os.path.isfile(os.path.join(collection_dir, f))
    ]

    # Generate HTML
    for design in design_files:
        run_halstead = subprocess.run(
            f"wily report 'iron/{design}' -n 2 halstead.h1 halstead.h2 halstead.N1 halstead.N2 halstead.volume halsted.vocabulary halstead.complexity halstead.length halstead.effort halstead.difficulty maintainability.mi --format HTML -o 'iron_out/{design}'",
            shell=True,
            env=os.environ,
            capture_output=True,
        )
        print(run_halstead.stdout)
        print(run_halstead.stderr)

    dfs = []
    for design in design_files:

        path = f"iron_out/{design}/index.html"

        # empty list
        data = []

        # for getting the header from
        # the HTML file
        list_header = []
        soup = BeautifulSoup(open(path), "html.parser")
        header = soup.find_all("table")[0].find("tr")

        for items in header:
            try:
                list_header.append(items.get_text())
            except:
                continue

        # for getting the data
        HTML_data = soup.find_all("table")[0].find_all("tr")[1:]

        for element in HTML_data:
            sub_data = []
            for sub_element in element:
                try:
                    sub_data.append(sub_element.get_text())
                except:
                    continue
            data.append(sub_data)

        # Storing the data into Pandas
        # DataFrame
        dataFrame = pd.DataFrame(data=data, columns=list_header)
        print(dataFrame)
        dataFrame["name"] = design.split(".")[0]
        dfs.append(dataFrame)

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

    print(dfs[0].keys())

    with open(output_file, "w") as of:
        # Write headers
        of.write(f"name,design,tool,{','.join(halstead_keys)},mi\n")

        for df in dfs:
            vocabulary = df.iloc[0]["Unique vocabulary (h1 + h2)"]
            iron_ext_vocab = float(vocabulary.split(" ")[0])
            iron_vocab = float(vocabulary.split(" ")[1].rstrip(")").lstrip("("))
            iron_vocab = iron_ext_vocab + iron_vocab
            print(f"{vocabulary} {iron_ext_vocab} {iron_vocab}")

            difficulty = df.iloc[0]["Difficulty"]
            iron_ext_diff = float(difficulty.split(" ")[0])
            iron_diff = float(difficulty.split(" ")[1].rstrip(")").lstrip("("))
            iron_diff = iron_ext_diff + iron_diff
            print(f"{difficulty} {iron_ext_diff} {iron_diff}")

            mi = df.iloc[0]["Maintainability Index"]
            print(f"MI: {mi}")
            iron_ext_mi = float(mi.split(" ")[0])
            iron_mi = float(mi.split(" ")[1].rstrip(")").lstrip("("))
            iron_mi = iron_ext_mi + iron_mi
            print(f"{mi} {iron_ext_mi} {iron_mi}")

            of.write(
                f"iron_ext,{df.iloc[0]['name']},wily,h1,h2,N1,N2,{iron_ext_vocab},length,calculated_length,volume,{iron_ext_diff},effort,time,bugs,{iron_ext_mi}\n"
            )
            of.write(
                f"iron,{df.iloc[0]['name']},wily,h1,h2,N1,N2,{iron_vocab},length,calculated_length,volume,{iron_diff},effort,time,bugs,{iron_mi}\n"
            )


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
