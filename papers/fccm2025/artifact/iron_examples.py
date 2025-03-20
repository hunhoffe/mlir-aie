import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from artifact.example import Example, ExampleCollection

PROGRAMMING_EXAMPLES_DIR = "./programming_examples"


def generate_iron_example_collection():
    iron_examples = ExampleCollection()

    ################# Blocks

    # Passthrough x3
    e = Example(
        name="DMA",
        category="Copy",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/passthrough_dmas",
        iron_src="dma_transpose_alt.py",
        iron_ext_src="dma_transpose_iron.py",
        iron_build_env="use_alt=1",
        iron_ext_build_env="use_iron=1",
    )
    iron_examples.append(e)
    e = Example(
        name="ExtKern",
        category="Copy",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/passthrough_kernel",
        mlir_src="build/aie2_lineBased_8b_4096.mlir",
    )
    iron_examples.append(e)
    e = Example(
        name="Kern",
        category="Copy",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/passthrough_pykernel",
    )
    iron_examples.append(e)

    # MTranspose
    e = Example(
        name="MTranspose",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/dma_transpose",
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
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/vector_reduce_add",
    )
    iron_examples.append(e)
    e = Example(
        name="Max",
        category="VReduce",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/vector_reduce_max",
    )
    iron_examples.append(e)
    e = Example(
        name="Min",
        category="VReduce",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/vector_reduce_min",
    )
    iron_examples.append(e)

    # VSOp x2
    e = Example(
        name="Add",
        category="VSOp",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/vector_scalar_add",
    )
    iron_examples.append(e)
    e = Example(
        name="Mul",
        category="VSOp",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/vector_scalar_mul",
        mlir_src="build/aie_4096.mlir",
    )
    iron_examples.append(e)

    # VVop x5
    e = Example(
        name="Add",
        category="VVOp",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/vector_vector_add",
    )
    iron_examples.append(e)
    e = Example(
        name="Mod",
        category="VVOp",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/vector_vector_modulo",
    )
    iron_examples.append(e)
    e = Example(
        name="Mul",
        category="VVOp",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/vector_vector_mul",
    )
    iron_examples.append(e)
    e = Example(
        name="AddKern",
        category="VVOp",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/ml/eltwise_add",
    )
    iron_examples.append(e)
    e = Example(
        name="MulKern",
        category="VVOp",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/ml/eltwise_mul",
    )
    iron_examples.append(e)

    # MSAdd
    e = Example(
        name="MSAdd",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/matrix_scalar_add",
    )
    iron_examples.append(e)

    # MVAdd
    e = Example(
        name="MVAdd",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/row_wise_bias_add",
    )
    iron_examples.append(e)

    # MVMul
    e = Example(
        name="MVMul",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/matrix_multiplication/matrix_vector",
        iron_src="matrix_vector_alt.py",
        iron_ext_src="matrix_vector_iron.py",
        iron_build_env="use_alt=1",
        iron_ext_build_env="use_iron=1",
        mlir_src="build/aie_288x288x1.mlir",
    )
    iron_examples.append(e)

    # VSoftMax
    e = Example(
        name="VSoftMax",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/ml/softmax",
    )
    iron_examples.append(e)

    # VReLU
    e = Example(
        name="VReLU",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/ml/relu",
    )
    iron_examples.append(e)

    # Conv2D x2
    e = Example(
        name="",
        category="Conv2D",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/ml/conv2d",
        run_cmd="run_py",
    )
    iron_examples.append(e)
    e = Example(
        name="FusedReLU",
        category="Conv2D",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/ml/conv2d_fused_relu",
        run_cmd="run_py",
        mlir_src="build/aieWithTrace_1core.mlir",
    )
    iron_examples.append(e)

    ############### Advanced Designs

    # GEMM
    e = Example(
        name="GEMM",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/basic/matrix_multiplication/whole_array",
        iron_src="whole_array_alt.py",
        iron_ext_src="whole_array_iron.py",
        iron_build_env="use_alt=1",
        iron_ext_build_env="use_iron=1",
        mlir_src="build/aie_512x512x512_64x64x64_4c.mlir",
    )
    iron_examples.append(e)

    # BottleneckBlock
    e = Example(
        name="BBlock",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/ml/bottleneck",
        run_cmd="run_py",
    )
    iron_examples.append(e)

    # ResNet Conv2x Layer
    e = Example(
        name="ResNetConv2x",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/ml/resnet/layers_conv2_x",
        iron_src="resnet_alt.py",
        iron_ext_src="resnet.py",
        run_cmd="run_py",
    )
    iron_examples.append(e)

    # ColorDetect
    e = Example(
        name="ColorDetect",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/vision/color_detect",
        mlir_src="build/aie2_lineBased_8b_1920.mlir",
    )
    iron_examples.append(e)

    # EdgeDetect
    e = Example(
        name="EdgeDetect",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/vision/edge_detect",
        mlir_src="build/aie2_lineBased_8b_1920.mlir",
    )
    iron_examples.append(e)

    # ColorThreshold
    e = Example(
        name="ColorThreshold",
        dir=f"{PROGRAMMING_EXAMPLES_DIR}/vision/color_threshold",
        mlir_src="build/aie2_1920.mlir",
    )
    iron_examples.append(e)

    return iron_examples
