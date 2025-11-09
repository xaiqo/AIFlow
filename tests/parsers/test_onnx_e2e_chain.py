from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from aiflow.ir import GraphValidator
from aiflow.parsers.onnx import OnnxParser


def test_e2e_conv_bn_relu_pool_matmul_reshape_transpose() -> None:
    # Shapes
    n, c_in, h, w = 1, 3, 8, 8
    c_out = 4
    k = 3
    # Inputs
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [n, c_in, h, w])  # NCHW
    # Conv weights/bias
    w_conv = helper.make_tensor(
        "w_conv",
        TensorProto.FLOAT,
        [c_out, c_in, k, k],
        np.ones((c_out, c_in, k, k), dtype=np.float32).flatten().tolist(),
    )
    b_conv = helper.make_tensor(
        "b_conv",
        TensorProto.FLOAT,
        [c_out],
        np.zeros((c_out,), dtype=np.float32).tolist(),
    )
    # BatchNorm parameters
    scale = helper.make_tensor(
        "bn_scale",
        TensorProto.FLOAT,
        [c_out],
        np.ones((c_out,), dtype=np.float32).tolist(),
    )
    bias = helper.make_tensor(
        "bn_bias",
        TensorProto.FLOAT,
        [c_out],
        np.zeros((c_out,), dtype=np.float32).tolist(),
    )
    mean = helper.make_tensor(
        "bn_mean",
        TensorProto.FLOAT,
        [c_out],
        np.zeros((c_out,), dtype=np.float32).tolist(),
    )
    var = helper.make_tensor(
        "bn_var",
        TensorProto.FLOAT,
        [c_out],
        np.ones((c_out,), dtype=np.float32).tolist(),
    )

    # Chain: Conv -> BN -> Relu -> MaxPool -> MatMul -> Reshape -> Transpose
    conv = helper.make_node(
        "Conv",
        ["x", "w_conv", "b_conv"],
        ["y1"],
        kernel_shape=[k, k],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    bn = helper.make_node(
        "BatchNormalization",
        ["y1", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["y2"],
        epsilon=1e-5,
        momentum=0.9,
    )
    relu = helper.make_node("Relu", ["y2"], ["y3"])
    pool = helper.make_node(
        "MaxPool", ["y3"], ["y4"], kernel_shape=[2, 2], strides=[2, 2]
    )
    # Prepare MatMul dimensions: flatten spatial and multiply by a weight matrix
    # Let y4: [N, C_out, H/2, W/2] => flatten to [N, C_out * H/2 * W/2]
    # Make weight: [C_out * H/2 * W/2, 6]
    h2, w2 = h // 2, w // 2
    flat = c_out * h2 * w2
    w_mm = helper.make_tensor(
        "w_mm",
        TensorProto.FLOAT,
        [flat, 6],
        np.ones((flat, 6), dtype=np.float32).flatten().tolist(),
    )
    reshape1_shape = helper.make_tensor("shape1", TensorProto.INT64, [2], [n, flat])
    reshape1 = helper.make_node("Reshape", ["y4", "shape1"], ["y5"])
    mm = helper.make_node("MatMul", ["y5", "w_mm"], ["y6"])
    reshape2_shape = helper.make_tensor(
        "shape2", TensorProto.INT64, [3], [2, 3, 1]
    )  # 6 -> [2,3,1]
    reshape2 = helper.make_node("Reshape", ["y6", "shape2"], ["y7"])
    transpose = helper.make_node("Transpose", ["y7"], ["y_out"], perm=[1, 2, 0])
    y_out = helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [3, 1, 2])

    graph = helper.make_graph(
        [conv, bn, relu, pool, reshape1, mm, reshape2, transpose],
        "e2e_chain",
        [x],
        [y_out],
        initializer=[
            w_conv,
            b_conv,
            scale,
            bias,
            mean,
            var,
            w_mm,
            reshape1_shape,
            reshape2_shape,
        ],
        value_info=[
            helper.make_tensor_value_info("y1", TensorProto.FLOAT, [n, c_out, h, w]),
            helper.make_tensor_value_info("y2", TensorProto.FLOAT, [n, c_out, h, w]),
            helper.make_tensor_value_info("y3", TensorProto.FLOAT, [n, c_out, h, w]),
            helper.make_tensor_value_info("y4", TensorProto.FLOAT, [n, c_out, h2, w2]),
            helper.make_tensor_value_info("y5", TensorProto.FLOAT, [n, flat]),
            helper.make_tensor_value_info("y6", TensorProto.FLOAT, [n, 6]),
            helper.make_tensor_value_info("y7", TensorProto.FLOAT, [2, 3, 1]),
        ],
    )
    model = helper.make_model(graph, producer_name="e2e-test")

    ir = OnnxParser().parse(model)  # validates and infers
    # Ensure parser produced nodes and final shape matches expectation
    assert [n.op_type for n in ir.nodes] == [
        "Conv",
        "BatchNormalization",
        "Relu",
        "MaxPool",
        "Reshape",
        "MatMul",
        "Reshape",
        "Transpose",
    ]
    assert ir.tensors["y_out"].shape == [3, 1, 2]
    GraphValidator(ir).validate()
