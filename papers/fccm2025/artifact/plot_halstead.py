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
    # name,design,tool,h1,h2,N1,N2,vocabulary,length,calculated_length,volume,difficulty,effort,time,bugs,mi

    df_iron = df[df["name"] == "iron"]
    df_iron.drop("name", axis=1, inplace=True)
    df_iron.drop("tool", axis=1, inplace=True)
    df_iron.drop("mi", axis=1, inplace=True)

    df_iron_ext = df[df["name"] == "iron_ext"]
    df_iron_ext.drop("name", axis=1, inplace=True)
    df_iron_ext.drop("tool", axis=1, inplace=True)
    df_iron_ext.drop("mi", axis=1, inplace=True)

    block = [
        # Block
        "Copy(DMA)",
        "Copy(Kern)",
        "Copy(ExtKern)",
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
        "MSAdd",
        "MVAdd",
        "MVMul",
        "VSoftMax",
        "VReLU",
        "Conv2D()",
        "Conv2D(FusedReLU)",
    ]
    advanced = [
        # Advanced
        "GEMM",
        "BBlock",
        "ResNetConv2x",
        "ColorDetect",
        "EdgeDetect",
        "ColorThreshold",
    ]
    designlist = block + advanced

    names = df["design"].unique()
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

    df_iron = df_iron.sort_values(by="design", key=sorter)
    df_iron.reset_index(drop=True, inplace=True)

    df_iron_ext = df_iron_ext.sort_values(by="design", key=sorter)
    df_iron_ext.reset_index(drop=True, inplace=True)

    df_iron = df_iron.drop(columns=["design"])
    df_iron_ext = df_iron_ext.drop(columns=["design"])

    print(df_iron_ext)
    print(df_iron)
    df_diff = df_iron_ext - df_iron
    df_diff["design"] = designlist

    # This is for one plot per metrics
    # Create the bar chart
    halstead_metrics = [
        #"h1",
        #"h2",
        #"N1",
        #"N2",
        "vocabulary",
        #"length",
        #"calculated_length",
        #"volume",
        #"difficulty",
        "effort",
        #"time",
        #"bugs",
    ]
    os.mkdir("halstead_graphs")
    for metric in halstead_metrics:
        ax = df_diff.plot.bar(x="design", y=[metric])

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
        plt.ylabel(f"Differences in {metric}")
        plt.tight_layout()
        plt.savefig(os.path.join("halstead_graphs", f"{metric}{output_file}"))

        df_diff["percent_decrease"] = df_diff[metric] / df_iron[metric]
        average = np.mean(df_diff[metric])
        print(f"The average change in mean for {metric} is: {average}")
        average_percent_reduction = np.mean(df_diff["percent_decrease"])
        print(
            f"The average percent reduction for {metric} is: {average_percent_reduction}"
        )


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
