import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from analysis.example import Example, ExampleCollection


def generate_iron_example_collection():

    ################# Blocks

    # Passthrough x4
    iron_examples = ExampleCollection()
    e = Example(
        name="DMA",
        category="Passthrough",
        dir="../basic/passthrough_dmas",
    )
    iron_examples.append(e)
    e = Example(
        name="Kernel",
        category="Passthrough",
        dir="../basic/passthrough_kernel",
    )
    iron_examples.append(e)
    e = Example(
        name="PyKernel",
        category="Passthrough",
        dir="../basic/passthrough_pykernel",
    )
    iron_examples.append(e)
    e = Example(
        name="SubVectors",
        category="Passthrough",
        dir="../vision/vision_passthrough",
    )
    iron_examples.append(e)

    # MTranspose

    # VReduce x3

    # VSOp x2

    # VVop x5

    # MVAdd

    # MVMul

    # GEMMSingle

    # VSoftMax

    # VRelu

    # Conv2d x2

    ############### Advanced Designs

    # GEMM

    # BottleneckBlock

    # ResNet Conv2x Layer

    # ColorDetect

    # EdgeDetect

    # ColorThreshold

    return iron_examples
