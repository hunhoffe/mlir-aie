import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pandas as pd


def plot_loc(input_file: str, output_file_percentage: str, output_file_absolute: str):
    # Read in data
    df = pd.read_csv(input_file)

    # CSV format:
    # name,pygount_loc_baseline, pygount_loc_ext, radon_loc_baseline, radon_loc_ext

    # Create percentage difference for loc of each type
    df["radon_loc_ratio"] = (df["radon_loc_ext"] / df["radon_loc_baseline"]) - 1
    df["pygount_loc_ratio"] = (df["pygount_loc_ext"] / df["pygount_loc_baseline"]) - 1
    df["avg_loc_ratio"] = (df["radon_loc_ratio"] + df["pygount_loc_ratio"]) / 2

    # Create absolute differences in LoC
    df["radon_abs"] = df["radon_loc_ext"] - df["radon_loc_baseline"]
    df["pygount_abs"] = df["pygount_loc_ext"] - df["pygount_loc_baseline"]
    df["avg_loc_abs"] = (df["radon_abs"] + df["pygount_abs"]) / 2

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
        # "GEMMSingle",
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

    ############################### For Ratios/Percentage

    # Create the bar chart
    ax = df.plot.bar(x="name", y=["radon_loc_ratio", "pygount_loc_ratio"])

    # Calculate and plot the average line
    average = np.mean(df["avg_loc_ratio"])
    ax.axhline(average, color="gray", linestyle="--")
    print(f"Average percent decrease: {average}")

    # Add the rectangle for advanced designs
    rect = patches.Rectangle(
        (len(block) - 0.5, -0.6),
        len(advanced) + 0.25,
        0.6,
        linewidth=1,
        edgecolor="darkgreen",
        facecolor="honeydew",
        zorder=0,
    )
    ax.add_patch(rect)

    # Color labels for advanced designs
    plt.xticks(fontfamily="monospace")
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
    plt.xlabel("Designs")
    plt.ylabel("Percent Decrease (SLoC)")
    plt.tight_layout()
    plt.savefig(output_file_percentage)

    ############################### For Absolute

    # Create the bar chart
    ax = df.plot.bar(x="name", y=["radon_abs", "pygount_abs"])

    # Calculate and plot the average line
    average = np.mean(df["avg_loc_abs"])
    ax.axhline(average, color="gray", linestyle="--")
    print(f"Average absolute difference: {average}")

    # Add the rectangle for advanced designs
    rect = patches.Rectangle(
        (len(block) - 0.5, -350),
        len(advanced) + 0.25,
        350,
        linewidth=1,
        edgecolor="darkgreen",
        facecolor="honeydew",
        zorder=0,
    )
    ax.add_patch(rect)

    # Color labels for advanced designs
    plt.xticks(fontfamily="monospace")
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
    plt.xlabel("Designs")
    plt.ylabel("Absolute Difference in SLoC")
    plt.tight_layout()
    plt.savefig(output_file_absolute)


def main():
    parser = argparse.ArgumentParser(description="LoC Plotter")

    # Add arguments
    parser.add_argument(
        "-i", "--input", help="Path to the input CSV file", required=True
    )
    parser.add_argument(
        "-p",
        "--percentage-output",
        help="Path to the output file (percentage)",
        default="percentage_loc.png",
    )
    parser.add_argument(
        "-a",
        "--absolute-output",
        help="Path to the output file (absolute)",
        default="absolute_loc.png",
    )

    # Parse arguments
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: Input CSV file {args.input} is not a file.")
        exit(-1)
    if os.path.exists(args.percentage_output):
        print(f"Error: Output file {args.percentage_output} already exists.")
        exit(-1)
    if os.path.exists(args.absolute_output):
        print(f"Error: Output file {args.absolute_output} already exists.")
        exit(-1)

    plot_loc(args.input, args.percentage_output, args.absolute_output)


if __name__ == "__main__":
    main()
