import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pandas as pd


def plot_halstead(input_file: str, output_file):
    # Read in data
    df = pd.read_csv(input_file)

    # CSV format:
    # name,design,h1,h2,N1,N2,vocabulary,length,calculated_length,volume,difficulty,effort,time,bugs

    df_iron = df[df["design"] == "iron"]
    df_iron.drop("design", axis=1, inplace=True)

    df_iron_ext = df[df["design"] == "iron_ext"]
    df_iron_ext.drop("design", axis=1, inplace=True)

    block = [
        # Block
        "Passthrough(DMA)",
        "Passthrough(Kernel)",
        "Passthrough(PyKernel)",
        "MTranspose",
        "VReduce(Add)",
        "VReduce(Max)",
        "VReduce(Min)",
        "VSOp(Add)",
        "VSOp(Mul)",
        "VVOp(Add)",
        "VVOp(Mul)",
        "VVOp(Mod)",
        "VVOp(AddKern)",
        "VVOp(MulKern)",
        "MVAdd",
        "MVMul",
        # "GEMMSingle",
        "VSoftMax",
        "VReLU",
        "Conv2D()",
        "Conv2D(FusedReLU)",
    ]
    advanced = [
        # Advanced
        "GEMM",
        "BottleneckBlock",
        "ResNetConv2x",
        "ColorDetect",
        "EdgeDetect",
        "ColorThreshold",
    ]
    designlist = block + advanced

    names = df["name"].unique()
    for d in designlist:
        if d not in names:
            print(f"MISSING d: {d}")
            print(names)

    def sorter(column):
        # This also works:
        # mapper = {name: order for order, name in enumerate(reorder)}
        # return column.map(mapper)
        cat = pd.Categorical(column, categories=designlist, ordered=True)
        return pd.Series(cat)

    df_iron = df_iron.sort_values(by="name", key=sorter)
    df_iron.reset_index(drop=True, inplace=True)

    df_iron_ext = df_iron_ext.sort_values(by="name", key=sorter)
    df_iron_ext.reset_index(drop=True, inplace=True)

    df_iron = df_iron.drop(columns=["name"])
    df_iron_ext = df_iron_ext.drop(columns=["name"])

    df_diff = df_iron_ext - df_iron
    df_diff["name"] = designlist

    # This is for one plot per metrics
    # Create the bar chart
    halstead_metrics = [
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
    os.mkdir("halstead_graphs")
    for metric in halstead_metrics:
        ax = df_diff.plot.bar(x="name", y=[metric])

        # Color labels for advanced designs
        plt.xticks(fontfamily="monospace")
        for ticklabel in ax.get_xticklabels():
            if ticklabel.get_text() in advanced:
                ticklabel.set_color("darkgreen")
                ticklabel.set_fontweight("bold")

        plt.legend(
            labels=[metric],
            bbox_to_anchor=(0.5, 1.05),
            loc="lower center",
            ncol=4,
        )
        plt.xlabel("Designs")
        plt.ylabel("Absolute Differences in Haldstead Metrics")
        plt.tight_layout()
        plt.savefig(os.path.join("halstead_graphs", f"{metric}{output_file}"))

        average = np.mean(df_diff[metric])
        print(f"The average change in mean for {metric} is: {average}")


def main():
    parser = argparse.ArgumentParser(description="Halstead Plotter")

    # Add arguments
    parser.add_argument(
        "-i", "--input", help="Path to the input CSV file", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output file",
        default="halstead.png",
    )

    # Parse arguments
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input CSV file {args.input} is not a file.")
        exit(-1)
    if os.path.exists(args.output):
        print(f"Error: Output file {args.output} already exists.")
        exit(-1)

    plot_halstead(args.input, args.output)


if __name__ == "__main__":
    main()
