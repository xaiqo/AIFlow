from __future__ import annotations

import pytest

from aiflow.ir import Graph, InferenceError, Node, Tensor, infer_graph


def t(name: str, shape: list[int], dtype: str = "float32") -> Tensor:
    return Tensor(name=name, dtype=dtype, shape=shape)


def test_conv2d_basic() -> None:
    g = Graph()
    g.add_tensor(t("x", [1, 4, 8, 8]))
    g.add_tensor(t("w", [6, 4, 3, 3]))
    g.add_tensor(t("y", [1]))
    g.add_node(
        Node(
            "Conv",
            ["x", "w"],
            ["y"],
            attributes={"pads": [1, 1, 1, 1], "strides": [1, 1]},
        )
    )
    infer_graph(g)
    assert g.tensors["y"].shape == [1, 6, 8, 8]


def test_conv2d_stride_and_dilation() -> None:
    g = Graph()
    g.add_tensor(t("x", [1, 3, 32, 32]))
    g.add_tensor(t("w", [8, 3, 5, 5]))
    g.add_tensor(t("y", [1]))
    g.add_node(
        Node(
            "Conv",
            ["x", "w"],
            ["y"],
            attributes={"pads": [2, 2, 2, 2], "strides": [2, 2], "dilations": [2, 2]},
        )
    )
    infer_graph(g)
    # h_out = floor((32 + 2 + 2 - 2*(5-1) - 1)/2 + 1)
    #        = floor((34 - 8 - 1)/2 + 1) = floor(25/2 + 1) = 12
    # w_out same, C_out=8
    assert g.tensors["y"].shape == [1, 8, 12, 12]


def test_conv2d_groups() -> None:
    g = Graph()
    g.add_tensor(t("x", [1, 6, 16, 16]))
    g.add_tensor(t("w", [10, 3, 3, 3]))  # Cin/groups = 3
    g.add_tensor(t("y", [1]))
    g.add_node(
        Node(
            "Conv",
            ["x", "w"],
            ["y"],
            attributes={"groups": 2, "strides": [1, 1], "pads": [1, 1, 1, 1]},
        )
    )
    infer_graph(g)
    assert g.tensors["y"].shape == [1, 10, 16, 16]


def test_conv2d_invalid_groups_raises() -> None:
    g = Graph()
    g.add_tensor(t("x", [1, 5, 8, 8]))
    g.add_tensor(t("w", [6, 3, 3, 3]))
    g.add_tensor(t("y", [1]))
    g.add_node(Node("Conv", ["x", "w"], ["y"], attributes={"groups": 2}))
    with pytest.raises(InferenceError):
        infer_graph(g)


