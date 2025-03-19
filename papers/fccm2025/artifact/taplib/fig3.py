from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence

tap00 = TensorAccessPattern((6, 4), offset=0, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1])
tap11 = TensorAccessPattern((6, 4), offset=14, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1])

taps0 = TensorAccessSequence.from_taps([tap00, tap11])

tap00.visualize(
    plot_access_count=True, show_arrows=True, title="", file_path="tap00.png"
)
taps0.visualize(plot_access_count=True, file_path="taps0.png", title="")