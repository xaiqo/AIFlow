from __future__ import annotations

from aiflow.ir import Graph, Node, Tensor
from aiflow.optimizer import FusionCBRPass


def t(name: str, shape: list[int]) -> Tensor:
    return Tensor(name=name, dtype="float32", shape=shape)


def test_match_simple_conv_bn_relu_chain() -> None:
    g = Graph()
    # tensors
    for name in ["x", "w", "b", "y1", "y2", "y3", "s", "bb", "m", "v"]:
        g.add_tensor(t(name, [1]))
    # nodes: Conv -> BN -> Relu
    g.add_node(Node("Conv", ["x", "w", "b"], ["y1"], attributes={"strides": [1, 1]}))
    g.add_node(
        Node(
            "BatchNormalization",
            ["y1", "s", "bb", "m", "v"],
            ["y2"],
            attributes={"epsilon": 1e-5},
        )
    )
    g.add_node(Node("Relu", ["y2"], ["y3"]))
    g.outputs = ["y3"]  # final output only

    matches = list(FusionCBRPass().match(g))
    assert matches == [[0, 1, 2]]


def test_no_match_when_missing_bn_or_relu() -> None:
    g = Graph()
    for name in ["x", "w", "b", "y1", "y2"]:
        g.add_tensor(t(name, [1]))
    g.add_node(Node("Conv", ["x", "w", "b"], ["y1"]))
    g.add_node(Node("Relu", ["y1"], ["y2"]))  # missing BN in the middle
    g.outputs = ["y2"]

    matches = list(FusionCBRPass().match(g))
    assert matches == []


def test_no_match_on_branching_intermediate() -> None:
    g = Graph()
    for name in ["x", "w", "b", "y1", "y2", "y3", "z"]:
        g.add_tensor(t(name, [1]))
    g.add_node(Node("Conv", ["x", "w", "b"], ["y1"]))
    g.add_node(Node("BatchNormalization", ["y1", "w", "b", "y2", "y3"], ["y2"]))
    g.add_node(Node("Relu", ["y2"], ["y3"]))
    # Branch: y1 consumed by an extra node, breaking linear chain property
    g.add_node(Node("Add", ["y1", "y1"], ["z"]))
    g.outputs = ["y3"]

    matches = list(FusionCBRPass().match(g))
    assert matches == []


