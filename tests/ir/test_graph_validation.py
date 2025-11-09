from __future__ import annotations

import pytest

from aiflow.ir import Graph, GraphValidator, Node, Tensor, ValidationError


def make_tensor(
    name: str, shape: list[int] | None = None, dtype: str = "float32"
) -> Tensor:
    actual_shape = shape if shape is not None else [1]
    return Tensor(name=name, dtype=dtype, shape=actual_shape)


def test_valid_graph_passes_validation() -> None:
    g = Graph()
    g.add_tensor(make_tensor("x", [1, 3, 224, 224]))
    g.add_tensor(make_tensor("w", [64, 3, 7, 7]))
    g.add_tensor(make_tensor("y", [1, 64, 112, 112]))
    g.inputs = ["x", "w"]
    g.outputs = ["y"]
    g.add_node(Node(op_type="Conv", inputs=["x", "w"], outputs=["y"]))

    GraphValidator(g).validate()  # should not raise


def test_missing_tensor_in_inputs_raises() -> None:
    g = Graph()
    g.add_tensor(make_tensor("x", [1]))
    g.inputs = ["x", "z"]  # z missing
    with pytest.raises(ValidationError) as exc:
        GraphValidator(g).validate()
    assert exc.value.code == "EGRAPH_INPUT"


def test_missing_node_input_tensor_raises() -> None:
    g = Graph()
    g.add_tensor(make_tensor("x", [1]))
    g.add_tensor(make_tensor("y", [1]))
    g.add_node(Node(op_type="Add", inputs=["x", "z"], outputs=["y"]))  # z missing
    with pytest.raises(ValidationError) as exc:
        GraphValidator(g).validate()
    assert exc.value.code == "EINPUT_MISSING"


def test_duplicate_producer_raises() -> None:
    g = Graph()
    g.add_tensor(make_tensor("x", [1]))
    g.add_tensor(make_tensor("y", [1]))
    g.add_tensor(make_tensor("y2", [1]))
    g.add_node(Node(op_type="Id", inputs=["x"], outputs=["y"]))
    g.add_node(Node(op_type="Id", inputs=["x"], outputs=["y"]))  # duplicate
    with pytest.raises(ValidationError) as exc:
        GraphValidator(g).validate()
    assert exc.value.code == "EDUP_PRODUCER"


def test_cycle_detection_raises() -> None:
    # x -> n1 -> y; y -> n2 -> x (cycle)
    g = Graph()
    g.add_tensor(make_tensor("x", [1]))
    g.add_tensor(make_tensor("y", [1]))
    g.add_node(Node(op_type="Id", inputs=["x"], outputs=["y"]))
    g.add_node(Node(op_type="Id", inputs=["y"], outputs=["x"]))
    with pytest.raises(ValidationError) as exc:
        GraphValidator(g).validate()
    assert exc.value.code == "ECYCLE"


def test_toposort_ordering() -> None:
    # x -> a -> y -> b -> z
    g = Graph()
    for t in ["x", "y", "z"]:
        g.add_tensor(make_tensor(t, [1]))
    g.add_node(Node(op_type="A", inputs=["x"], outputs=["y"]))
    g.add_node(Node(op_type="B", inputs=["y"], outputs=["z"]))
    order = GraphValidator(g).toposort()
    # order of nodes should respect dependencies
    assert order[0].op_type == "A"
    assert order[1].op_type == "B"
