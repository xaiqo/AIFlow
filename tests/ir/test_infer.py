from __future__ import annotations

import pytest

from aiflow.ir import (
    Graph,
    InferenceError,
    Node,
    Tensor,
    infer_graph,
)


def t(name: str, shape: list[int], dtype: str = "float32") -> Tensor:
    return Tensor(name=name, dtype=dtype, shape=shape)


def test_infer_add_broadcast() -> None:
    g = Graph()
    g.add_tensor(t("a", [2, 3, 4]))
    g.add_tensor(t("b", [1, 3, 1]))
    g.add_tensor(t("c", [1]))  # placeholder, will be updated
    g.add_node(Node("Add", ["a", "b"], ["c"]))
    infer_graph(g)
    assert g.tensors["c"].shape == [2, 3, 4]
    assert g.tensors["c"].dtype == "float32"


def test_infer_matmul_batch() -> None:
    g = Graph()
    g.add_tensor(t("a", [5, 2, 3]))
    g.add_tensor(t("b", [5, 3, 4]))
    g.add_tensor(t("c", [1]))
    g.add_node(Node("MatMul", ["a", "b"], ["c"]))
    infer_graph(g)
    assert g.tensors["c"].shape == [5, 2, 4]


def test_infer_relu_passthrough() -> None:
    g = Graph()
    g.add_tensor(t("x", [7, 8]))
    g.add_tensor(t("y", [1]))
    g.add_node(Node("Relu", ["x"], ["y"]))
    infer_graph(g)
    assert g.tensors["y"].shape == [7, 8]
    assert g.tensors["y"].dtype == "float32"


def test_infer_add_mismatch_raises() -> None:
    g = Graph()
    g.add_tensor(t("a", [2, 3]))
    g.add_tensor(t("b", [4, 5]))
    g.add_tensor(t("c", [1]))
    g.add_node(Node("Add", ["a", "b"], ["c"]))
    with pytest.raises(InferenceError):
        infer_graph(g)


def test_infer_reshape_with_infer_dim() -> None:
    g = Graph()
    g.add_tensor(t("x", [2, 3, 4]))
    g.add_tensor(t("y", [1]))
    g.add_node(Node("Reshape", ["x"], ["y"], attributes={"shape": [4, -1, 1]}))
    infer_graph(g)
    assert g.tensors["y"].shape == [4, 6, 1]
    assert g.tensors["y"].dtype == "float32"


def test_infer_transpose_perm() -> None:
    g = Graph()
    g.add_tensor(t("x", [2, 3, 4]))
    g.add_tensor(t("y", [1]))
    g.add_node(Node("Transpose", ["x"], ["y"], attributes={"perm": [1, 2, 0]}))
    infer_graph(g)
    assert g.tensors["y"].shape == [3, 4, 2]


def test_infer_maxpool_2d() -> None:
    g = Graph()
    g.add_tensor(t("x", [1, 3, 32, 32]))
    g.add_tensor(t("y", [1]))
    g.add_node(
        Node(
            "MaxPool",
            ["x"],
            ["y"],
            attributes={"kernel_shape": [2, 2], "strides": [2, 2]},
        )
    )
    infer_graph(g)
    assert g.tensors["y"].shape == [1, 3, 16, 16]


def test_infer_concat_axis() -> None:
    g = Graph()
    g.add_tensor(t("a", [1, 2, 3]))
    g.add_tensor(t("b", [1, 5, 3]))
    g.add_tensor(t("c", [1]))
    g.add_node(Node("Concat", ["a", "b"], ["c"], attributes={"axis": 1}))
    infer_graph(g)
    assert g.tensors["c"].shape == [1, 7, 3]
