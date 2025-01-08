from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence, TensorTiler2D

# taps = TensorTiler2D.simple_tiler((6, 4), (3, 2))
# for t in taps:
#    print(t)

tap00 = TensorAccessPattern((6, 4), offset=0, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1])
# tap01 = TensorAccessPattern((6, 4), offset=2, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1])
# tap10 = TensorAccessPattern((6, 4), offset=12, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1])
tap11 = TensorAccessPattern((6, 4), offset=14, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1])

taps = TensorAccessSequence.from_taps([tap00, tap11])

tap00.visualize(
    plot_access_count=True, show_arrows=True, title="", file_path="tap00.png"
)
tap11.visualize(
    plot_access_count=True, show_arrows=True, title="", file_path="tap11.png"
)
taps.visualize(plot_access_count=True, file_path="taps.png", title="")

# Count index starts at 0, so highest count is 5
assert tap00.access_order().max() == 3 * 2 - 1
# 3x2 elements are accessed by tap00
assert tap00.access_count().sum() == 3 * 2
# The tas accesses 2 tiles worth of elements
assert taps.access_order().max() == 2 * (3 * 2) - 1
# The tas does not access an element more than once
assert taps.access_count().max() == 1
