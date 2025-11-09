from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from aiflow.ir import GraphValidator
from aiflow.parsers.onnx import OnnxParser


def _make_tensor_value_info(
    name: str, dtype: int, shape: list[int]
) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def _make_initializer(name: str, array: np.ndarray) -> onnx.TensorProto:
    return helper.make_tensor(
        name=name,
        data_type=(
            TensorProto.DataType.Name(array.dtype.type).index(array.dtype.type.__name__)
            if hasattr(TensorProto, "DataType")
            else TensorProto.FLOAT
        ),
        dims=list(array.shape),
        vals=array.flatten().tolist(),
    )


def test_parse_add_graph_with_initializer_consts() -> None:
    # a and b as initializers, c as graph output
    a = helper.make_tensor("a", TensorProto.FLOAT, [2, 2], [1.0, -2.0, 3.0, -4.0])
    b = helper.make_tensor("b", TensorProto.FLOAT, [2, 2], [5.0, 6.0, 7.0, 8.0])
    c_info = _make_tensor_value_info("c", TensorProto.FLOAT, [2, 2])

    node = helper.make_node("Add", inputs=["a", "b"], outputs=["c"])
    graph = helper.make_graph(
        nodes=[node],
        name="add_graph",
        inputs=[],  # no graph inputs; initializers only
        outputs=[c_info],
        initializer=[a, b],
    )
    model = helper.make_model(graph, producer_name="test")

    ir = OnnxParser().parse(model)
    # Check tensors presence and constants
    assert "a" in ir.tensors and "b" in ir.tensors and "c" in ir.tensors
    assert ir.tensors["a"].metadata.get("const") == [[1.0, -2.0], [3.0, -4.0]]
    assert ir.tensors["b"].metadata.get("const") == [[5.0, 6.0], [7.0, 8.0]]
    # Node mapping
    assert len(ir.nodes) == 1 and ir.nodes[0].op_type == "Add"
    # Graph outputs set
    assert ir.outputs == ["c"]
    # Should validate basic invariants (tensors exist, shapes/dtypes present)
    GraphValidator(ir).validate()


def test_parse_with_graph_input_and_output() -> None:
    x_info = _make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])
    y_info = _make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 32, 32])
    node = helper.make_node("Relu", inputs=["x"], outputs=["y"])
    graph = helper.make_graph([node], "relu_graph", [x_info], [y_info])
    model = helper.make_model(graph)

    ir = OnnxParser().parse(model)
    assert "x" in ir.tensors and "y" in ir.tensors
    assert ir.inputs == ["x"]
    assert ir.outputs == ["y"]
    assert ir.nodes[0].op_type == "Relu"
    GraphValidator(ir).validate()


def test_parse_reshape_with_shape_initializer() -> None:
    # x -> Reshape -> y, with shape provided as initializer s
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 6, 1])
    s = helper.make_tensor("s", TensorProto.INT64, [3], [-1, 6, 1])
    node = helper.make_node("Reshape", inputs=["x", "s"], outputs=["y"])
    graph = helper.make_graph(
        [node], "reshape_graph", [x_info], [y_info], initializer=[s]
    )
    model = helper.make_model(graph)

    ir = OnnxParser().parse(model)
    assert ir.nodes[0].op_type == "Reshape"
    # Ensure 'shape' attribute was captured from initializer
    assert ir.nodes[0].attributes.get("shape") == [-1, 6, 1]
    GraphValidator(ir).validate()


def test_parse_maxpool_attrs() -> None:
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 16, 16])
    node = helper.make_node(
        "MaxPool",
        inputs=["x"],
        outputs=["y"],
        kernel_shape=[2, 2],
        strides=[2, 2],
    )
    graph = helper.make_graph([node], "pool_graph", [x_info], [y_info])
    model = helper.make_model(graph)

    ir = OnnxParser().parse(model)
    n = ir.nodes[0]
    assert n.op_type == "MaxPool"
    assert n.attributes.get("kernel_shape") == [2, 2]
    assert n.attributes.get("strides") == [2, 2]
    GraphValidator(ir).validate()


def test_parse_infers_internal_tensor_shapes() -> None:
    # x (2,3,4) + b (1,3,1) -> c; Relu(c) -> y. Only y is graph output.
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    b_info = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 3, 1])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    add = helper.make_node("Add", ["x", "b"], ["c"])
    relu = helper.make_node("Relu", ["c"], ["y"])
    graph = helper.make_graph([add, relu], "chain", [x_info, b_info], [y_info])
    model = helper.make_model(graph)

    ir = OnnxParser().parse(model)  # validate_and_infer=True by default
    # 'c' is internal; parser created placeholder but inference should populate shape
    assert "c" in ir.tensors
    assert ir.tensors["c"].shape == [2, 3, 4]
    GraphValidator(ir).validate()
