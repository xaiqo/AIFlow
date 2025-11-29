from __future__ import annotations

from aiflow.ir import (
    Graph,
    GraphValidator,
    Node,
    Tensor,
    build_consumer_map,
    build_producer_map,
    extract_subgraph,
    find_linear_chains,
)


def t(name: str, shape: list[int]) -> Tensor:
    return Tensor(name=name, dtype="float32", shape=shape)


def test_producer_consumer_maps() -> None:
    # x -> A -> y -> B -> z
    g = Graph()
    g.add_tensor(t("x", [1]))
    g.add_tensor(t("y", [1]))
    g.add_tensor(t("z", [1]))
    g.add_node(Node("A", ["x"], ["y"]))
    g.add_node(Node("B", ["y"], ["z"]))

    p = build_producer_map(g)
    c = build_consumer_map(g)

    assert p["y"] == 0
    assert p["z"] == 1
    assert c["x"] == [0]
    assert c["y"] == [1]


def test_find_linear_chains() -> None:
    # x -> A -> y -> B -> z -> C -> w
    g = Graph()
    for name in ["x", "y", "z", "w"]:
        g.add_tensor(t(name, [1]))
    g.add_node(Node("A", ["x"], ["y"]))
    g.add_node(Node("B", ["y"], ["z"]))
    g.add_node(Node("C", ["z"], ["w"]))

    chains = find_linear_chains(g, ["A", "B", "C"])
    assert len(chains) == 1
    assert chains[0] == [0, 1, 2]


def test_extract_subgraph_middle_chain() -> None:
    # a -> A -> b -> B -> c -> C -> d
    g = Graph()
    for name in ["a", "b", "c", "d"]:
        g.add_tensor(t(name, [1]))
    g.add_node(Node("A", ["a"], ["b"]))
    g.add_node(Node("B", ["b"], ["c"]))
    g.add_node(Node("C", ["c"], ["d"]))
    g.inputs = ["a"]
    g.outputs = ["d"]
    GraphValidator(g).validate()

    sub = extract_subgraph(g, {1})  # only node B
    # Should include tensors b and c and set inputs/outputs properly
    assert "b" in sub.inputs
    assert "c" in sub.outputs
    assert len(sub.nodes) == 1 and sub.nodes[0].op_type == "B"
