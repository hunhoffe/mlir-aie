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
    e = Example(
        name="MTranspose",
        dir="../basic/dma_transpose",
        iron_src="dma_transpose_alt.py",
        iron_ext_src="dma_transpose_iron.py",
        iron_build_env="use_alt=1",
        iron_ext_build_env="use_iron=1",
    )
    iron_examples.append(e)

    # VReduce x3
    e = Example(
        name="Add",
        category="VReduce",
        dir="../basic/vector_reduce_add",
    )
    iron_examples.append(e)
    e = Example(
        name="Max",
        category="VReduce",
        dir="../basic/vector_reduce_max",
    )
    iron_examples.append(e)
    e = Example(
        name="Min",
        category="VReduce",
        dir="../basic/vector_reduce_min",
    )
    iron_examples.append(e)

    # VSOp x2
    e = Example(
        name="Add",
        category="VSOp",
        dir="../basic/vector_scalar_add",
    )
    iron_examples.append(e)
    e = Example(
        name="Mul",
        category="VSOp",
        dir="../basic/vector_scalar_mul",
    )
    iron_examples.append(e)

    # VVop x5
    e = Example(
        name="Add",
        category="VVop",
        dir="../basic/vector_vector_add",
    )
    iron_examples.append(e)
    e = Example(
        name="Mod",
        category="VVop",
        dir="../basic/vector_vector_modulo",
    )
    iron_examples.append(e)
    e = Example(
        name="Mul",
        category="VVop",
        dir="../basic/vector_vector_mul",
    )
    iron_examples.append(e)
    e = Example(
        name="AddKern",
        category="VVop",
        dir="../ml/eltwise_add",
    )
    iron_examples.append(e)
    e = Example(
        name="MulKern",
        category="VVop",
        dir="../ml/eltwise_mul",
    )
    iron_examples.append(e)

    # MVAdd
    e = Example(
        name="MVAdd",
        dir="../basic/row_wise_bias_add",
    )
    iron_examples.append(e)

    # MVMul
    e = Example(
        name="MVMul",
        dir="../basic/matrix_multiplication/matrix_vector",
        iron_src="matrix_vector_alt.py",
        iron_ext_src="matrix_vector_iron.py",
        iron_build_env="use_alt=1",
        iron_ext_build_env="use_iron=1",
    )
    iron_examples.append(e)

    # GEMMSingle
    e = Example(
        name="GEMMSingle",
        dir="../basic/matrix_multiplication/single_core",
        iron_src="single_core_alt.py",
        iron_ext_src="single_core_iron.py",
        iron_build_env="use_alt=1",
        iron_ext_build_env="use_iron=1",
    )
    iron_examples.append(e)

    # VSoftMax
    e = Example(
        name="VSoftMax",
        dir="../ml/softmax",
    )
    iron_examples.append(e)

    # VReLU
    e = Example(
        name="VReLU",
        dir="../ml/relu",
    )
    iron_examples.append(e)

    # Conv2D x2
    e = Example(
        name="",
        category="Conv2D",
        dir="../ml/conv2d",
        run_cmd="run_py",
    )
    iron_examples.append(e)
    e = Example(
        name="FusedRelu",
        category="Conv2D",
        dir="../ml/conv2d_fused_relu",
        run_cmd="run_py",
    )
    iron_examples.append(e)

    ############### Advanced Designs

    # GEMM

    # BottleneckBlock

    # ResNet Conv2x Layer

    # ColorDetect

    # EdgeDetect

    # ColorThreshold

    return iron_examples
