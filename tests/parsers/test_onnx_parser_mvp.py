from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from aiflow.ir import GraphValidator
from aiflow.parsers.onnx import OnnxParser


def _make_tensor_value_info(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def _make_initializer(name: str, array: np.ndarray) -> onnx.TensorProto:
    return helper.make_tensor(
        name=name,
        data_type=TensorProto.DataType.Name(array.dtype.type).index(array.dtype.type.__name__)
        if hasattr(TensorProto, "DataType")
        else TensorProto.FLOAT,
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


