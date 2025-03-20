from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence

print("Comparing taps from MVAdd diff results")

#### Tap from design 1
# aie.dma_bd(%arg1 : memref<32xf32>, 0, 2304, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 72, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
tap1 = TensorAccessPattern(
    [32, 32], offset=0, sizes=[1, 1, 72, 32], strides=[0, 0, 32, 1]
)

#### Tap from design 2
# aie.dma_bd(%arg1 : memref<32xf32>, 0, 2304, [<size = 1, stride = 0>, <size = 72, stride = 32>, <size = 1, stride = 2304>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
tap2 = TensorAccessPattern(
    [32, 32], offset=0, sizes=[1, 72, 1, 32], strides=[0, 32, 2304, 1]
)

if not tap1.compare_access_orders(tap2):
    print("ERROR: Taps for MVAdd are NOT access equivalent.")
    exit(-1)
print("MVAdd taps are access equivalent!")

print("Comparing taps from MTranspose diff results")

#### Tap from design 1
# aie.dma_bd(%arg2 : memref<64x32xi32>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
tap1 = TensorAccessPattern(
    [64, 32, 32], offset=2048, sizes=[1, 1, 1, 2048], strides=[0, 0, 0, 1]
)

#### Tap from design 2
# aie.dma_bd(%arg2 : memref<64x32xi32>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 32>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
tap2 = TensorAccessPattern(
    [64, 32, 32], offset=2048, sizes=[1, 1, 64, 32], strides=[0, 0, 32, 1]
)

if not tap1.compare_access_orders(tap2):
    print("ERROR: Taps for MTranspose are NOT access equivalent.")
    exit(-1)
print("MTranspose taps are access equivalent!")

print("Comparing taps from GEMM diff results")

#### From diff of design one
taps1 = []
# aie.dma_bd(%arg1 : memref<262144xi16>, 0, 32768, [<size = 2, stride = 256>, <size = 8, stride = 32768>, <size = 64, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}} {repeat_count = 1 : i32}
tap1_0 = TensorAccessPattern(
    [262144], offset=0, sizes=[2, 8, 64, 64], strides=[256, 32768, 512, 1]
)
taps1.append(tap1_0)

# aie.dma_bd(%arg1 : memref<262144xi16>, 0, 32768, [<size = 2, stride = 256>, <size = 8, stride = 32768>, <size = 64, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}} {repeat_count = 1 : i32}
tap1_1 = TensorAccessPattern(
    [262144], offset=0, sizes=[2, 8, 64, 64], strides=[256, 32768, 512, 1]
)
taps1.append(tap1_1)

# aie.dma_bd(%arg1 : memref<262144xi16>, 64, 32768, [<size = 2, stride = 256>, <size = 8, stride = 32768>, <size = 64, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}} {repeat_count = 1 : i32}
tap1_2 = TensorAccessPattern(
    [262144], offset=64, sizes=[2, 8, 64, 64], strides=[256, 32768, 512, 1]
)
taps1.append(tap1_2)

# aie.dma_bd(%arg1 : memref<262144xi16>, 64, 32768, [<size = 2, stride = 256>, <size = 8, stride = 32768>, <size = 64, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}} {repeat_count = 1 : i32}
tap1_3 = TensorAccessPattern(
    [262144], offset=64, sizes=[2, 8, 64, 64], strides=[256, 32768, 512, 1]
)
taps1.append(tap1_3)

# aie.dma_bd(%arg1 : memref<262144xi16>, 128, 32768, [<size = 2, stride = 256>, <size = 8, stride = 32768>, <size = 64, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}} {repeat_count = 1 : i32}
tap1_4 = TensorAccessPattern(
    [262144], offset=128, sizes=[2, 8, 64, 64], strides=[256, 32768, 512, 1]
)
taps1.append(tap1_4)

# aie.dma_bd(%arg1 : memref<262144xi16>, 128, 32768, [<size = 2, stride = 256>, <size = 8, stride = 32768>, <size = 64, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}} {repeat_count = 1 : i32}
tap1_5 = TensorAccessPattern(
    [262144], offset=128, sizes=[2, 8, 64, 64], strides=[256, 32768, 512, 1]
)
taps1.append(tap1_5)

# aie.dma_bd(%arg1 : memref<262144xi16>, 192, 32768, [<size = 2, stride = 256>, <size = 8, stride = 32768>, <size = 64, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}} {repeat_count = 1 : i32}
tap1_6 = TensorAccessPattern(
    [262144], offset=192, sizes=[2, 8, 64, 64], strides=[256, 32768, 512, 1]
)
taps1.append(tap1_6)

# aie.dma_bd(%arg1 : memref<262144xi16>, 192, 32768, [<size = 2, stride = 256>, <size = 8, stride = 32768>, <size = 64, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}} {repeat_count = 1 : i32}
tap1_7 = TensorAccessPattern(
    [262144], offset=192, sizes=[2, 8, 64, 64], strides=[256, 32768, 512, 1]
)
taps1.append(tap1_7)

##### From diff of design 2
taps2 = []
# aie.dma_bd(%arg1 : memref<262144xi16>, 0, 65536, [<size = 1, stride = 0>, <size = 2, stride = 256>, <size = 512, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}}
tap2_0 = TensorAccessPattern(
    [262144], offset=0, sizes=[1, 2, 512, 64], strides=[0, 256, 512, 1]
)
taps2.append(tap2_0)

# aie.dma_bd(%arg1 : memref<262144xi16>, 0, 65536, [<size = 1, stride = 0>, <size = 2, stride = 256>, <size = 512, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}}
tap2_1 = TensorAccessPattern(
    [262144], offset=0, sizes=[1, 2, 512, 64], strides=[0, 256, 512, 1]
)
taps2.append(tap2_1)

# aie.dma_bd(%arg1 : memref<262144xi16>, 64, 65536, [<size = 1, stride = 0>, <size = 2, stride = 256>, <size = 512, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}}
tap2_2 = TensorAccessPattern(
    [262144], offset=64, sizes=[1, 2, 512, 64], strides=[0, 256, 512, 1]
)
taps2.append(tap2_2)

# aie.dma_bd(%arg1 : memref<262144xi16>, 64, 65536, [<size = 1, stride = 0>, <size = 2, stride = 256>, <size = 512, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}}
tap2_3 = TensorAccessPattern(
    [262144], offset=64, sizes=[1, 2, 512, 64], strides=[0, 256, 512, 1]
)
taps2.append(tap2_3)

# aie.dma_bd(%arg1 : memref<262144xi16>, 128, 65536, [<size = 1, stride = 0>, <size = 2, stride = 256>, <size = 512, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}}
tap2_4 = TensorAccessPattern(
    [262144], offset=128, sizes=[1, 2, 512, 64], strides=[0, 256, 512, 1]
)
taps2.append(tap2_4)

# aie.dma_bd(%arg1 : memref<262144xi16>, 128, 65536, [<size = 1, stride = 0>, <size = 2, stride = 256>, <size = 512, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}}
tap2_5 = TensorAccessPattern(
    [262144], offset=128, sizes=[1, 2, 512, 64], strides=[0, 256, 512, 1]
)
taps2.append(tap2_5)

# aie.dma_bd(%arg1 : memref<262144xi16>, 192, 65536, [<size = 1, stride = 0>, <size = 2, stride = 256>, <size = 512, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}}
tap2_6 = TensorAccessPattern(
    [262144], offset=192, sizes=[1, 2, 512, 64], strides=[0, 256, 512, 1]
)
taps2.append(tap2_6)

# aie.dma_bd(%arg1 : memref<262144xi16>, 192, 65536, [<size = 1, stride = 0>, <size = 2, stride = 256>, <size = 512, stride = 512>, <size = 64, stride = 1>]) {burst_length = 0 : i32}}
tap2_7 = TensorAccessPattern(
    [262144], offset=192, sizes=[1, 2, 512, 64], strides=[0, 256, 512, 1]
)
taps2.append(tap2_7)

#### Compare two designs
assert len(taps1) == len(taps2)
for t1, t2 in zip(taps1, taps2):
    # Check if each pair is equal
    if not t1.compare_access_orders(t2):
        print("ERROR: taps for GEMM are NOT access equivalent!")
        exit(-1)
print("GEMM taps are access equivalent!")
