from __future__ import annotations

import numpy as np

from aiflow.ir import Graph, Node, Tensor
from aiflow.optimizer import ConstantFoldingPass, DeadCodeEliminationPass, Pipeline


def t(name: str, shape: list[int], dtype: str = "float32") -> Tensor:
    return Tensor(name=name, dtype=dtype, shape=shape)


def test_constant_folding_add_and_relu_and_transpose() -> None:
    g = Graph()
    g.add_tensor(t("a", [2, 2]))
    g.add_tensor(t("b", [2, 2]))
    g.add_tensor(t("c", [2, 2]))
    g.add_tensor(t("d", [2, 2]))
    g.add_tensor(t("e", [2, 2]))
    g.tensors["a"].metadata["const"] = [[1, -2], [3, -4]]
    g.tensors["b"].metadata["const"] = [[5, 6], [7, 8]]
    g.add_node(Node("Add", ["a", "b"], ["c"]))
    g.add_node(Node("Relu", ["c"], ["d"]))
    g.add_node(Node("Transpose", ["d"], ["e"], attributes={"perm": [1, 0]}))

    Pipeline([ConstantFoldingPass()]).run(g)

    add_out = np.array(g.tensors["c"].metadata["const"])
    relu_out = np.array(g.tensors["d"].metadata["const"])
    trans_out = np.array(g.tensors["e"].metadata["const"])
    np.testing.assert_array_equal(add_out, np.array([[6, 4], [10, 4]]))
    np.testing.assert_array_equal(relu_out, np.maximum(add_out, 0))
    np.testing.assert_array_equal(trans_out, relu_out.T)


def test_dce_removes_unused_nodes() -> None:
    g = Graph()
    g.add_tensor(t("x", [1]))
    g.add_tensor(t("y", [1]))
    g.add_tensor(t("z", [1]))
    g.add_node(Node("A", ["x"], ["y"]))
    g.add_node(Node("B", ["y"], ["z"]))  # used -> should stay
    # Create an extra node whose output is unused
    g.add_tensor(t("w", [1]))
    g.add_node(Node("C", ["x"], ["w"]))  # unused -> should be removed
    Pipeline([DeadCodeEliminationPass()]).run(g)
    # Expect only two nodes left (A and B)
    assert len(g.nodes) == 2
    assert [n.op_type for n in g.nodes] == ["A", "B"]
