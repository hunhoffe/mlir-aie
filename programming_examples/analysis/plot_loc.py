import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pandas as pd


def plot_loc(input_file: str, output_file: str):
    # Read in data
    df = pd.read_csv(input_file)

    # CSV format:
    # name,pygount_loc_baseline, pygount_loc_ext, radon_loc_baseline, radon_loc_ext

    # Create percentage difference for loc of each type
    df["radon_loc_ratio"] = (df["radon_loc_ext"] / df["radon_loc_baseline"]) - 1
    df["pygount_loc_ratio"] = (df["pygount_loc_ext"] / df["pygount_loc_baseline"]) - 1
    df["avg_loc_ratio"] = (df["radon_loc_ratio"] + df["pygount_loc_ratio"]) / 2

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
        "VVop(Add)",
        "VVop(Mul)",
        "VVop(Mod)",
        "VVop(AddKern)",
        "VVop(MulKern)",
        "MVAdd",
        "MVMul",
        "GEMMSingle",
        "VSoftMax",
        "VReLU",
        "Conv2D()",
        "Conv2D(FusedRelu)",
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

    df = df.sort_values(by="name", key=sorter)

    # Create the bar chart
    ax = df.plot.bar(x="name", y=["radon_loc_ratio", "pygount_loc_ratio"])

    # Calculate and plot the average line
    average = np.mean(df["avg_loc_ratio"])
    ax.axhline(average, color="gray", linestyle="--")

    # Add the rectangle for advanced designs
    rect = patches.Rectangle(
        (len(block) - 0.5, -0.53),
        len(advanced) + 0.25,
        0.55,
        linewidth=1,
        edgecolor="darkgreen",
        facecolor="honeydew",
        zorder=0,
    )
    ax.add_patch(rect)

    # Color labels for advanced designs
    for ticklabel in ax.get_xticklabels():
        if ticklabel.get_text() in advanced:
            ticklabel.set_color("darkgreen")
            ticklabel.set_fontweight("bold")

    # plt.title("Percentage Difference in LoC")
    plt.legend(
        labels=["average", "advanced", "radon", "pygount"],
        bbox_to_anchor=(0.5, 1.05),
        loc="lower center",
        ncol=4,
    )
    plt.xlabel("Design")
    plt.ylabel("Percentage Difference in LoC")
    plt.tight_layout()
    plt.savefig(output_file)


def main():
    parser = argparse.ArgumentParser(description="LoC Plotter")

    # Add arguments
    parser.add_argument(
        "-i", "--input", help="Path to the input CSV file", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Path to the output file", default="loc.png"
    )

    # Parse arguments
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input CSV file {args.input} is not a file.")
        exit(-1)
    if os.path.exists(args.output):
        print(f"Error: Output file {args.output} already exists.")
        exit(-1)

    plot_loc(args.input, args.output)


if __name__ == "__main__":
    main()
