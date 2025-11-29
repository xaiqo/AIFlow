from __future__ import annotations

from aiflow.ir import Graph, Node, Tensor
from aiflow.optimizer import FusionCBRPass, build_default_pipeline


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


def test_rewrite_conv_bn_relu_to_fused_conv() -> None:
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
    g.outputs = ["y3"]

    # Run pipeline with just fusion pass (apply should perform rewrite)
    matches = list(FusionCBRPass().match(g))
    assert matches == [[0, 1, 2]]
    FusionCBRPass().apply(g, matches[0])

    # After rewrite, only Conv should remain and output y3, with fused metadata
    assert len(g.nodes) == 1
    n0 = g.nodes[0]
    assert n0.op_type == "Conv"
    assert n0.outputs == ["y3"]
    assert n0.metadata.get("fused_relu") is True
    fused_bn = n0.metadata.get("fused_bn")
    assert isinstance(fused_bn, dict)
    assert fused_bn.get("scale") == "s"
    assert fused_bn.get("bias") == "bb"
    assert fused_bn.get("mean") == "m"
    assert fused_bn.get("var") == "v"
    assert fused_bn.get("epsilon") == 1e-5


def test_pipeline_toggle_enables_fusion() -> None:
    g = Graph()
    for name in ["x", "w", "b", "y1", "y2", "y3", "s", "bb", "m", "v"]:
        g.add_tensor(t(name, [1]))
    g.add_node(Node("Conv", ["x", "w", "b"], ["y1"]))
    g.add_node(Node("BatchNormalization", ["y1", "s", "bb", "m", "v"], ["y2"]))
    g.add_node(Node("Relu", ["y2"], ["y3"]))
    g.outputs = ["y3"]

    # Fusion disabled
    g_disabled = build_default_pipeline(enable_fusion=False).run(
        Graph(
            nodes=list(g.nodes),
            tensors=dict(g.tensors),
            metadata=dict(g.metadata),
            inputs=list(g.inputs),
            outputs=list(g.outputs),
        )
    )
    assert [n.op_type for n in g_disabled.nodes] == ["Conv", "BatchNormalization", "Relu"]

    # Fusion enabled
    g_enabled = build_default_pipeline(enable_fusion=True).run(g)
    assert [n.op_type for n in g_enabled.nodes] == ["Conv"]

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


def test_no_rewrite_when_bn_params_missing() -> None:
    g = Graph()
    for name in ["x", "w", "b", "y1", "y2", "y3"]:
        g.add_tensor(t(name, [1]))
    g.add_node(Node("Conv", ["x", "w", "b"], ["y1"]))
    # BN missing some params (only 3 given instead of 5)
    g.add_node(Node("BatchNormalization", ["y1", "w", "b"], ["y2"]))
    g.add_node(Node("Relu", ["y2"], ["y3"]))
    g.outputs = ["y3"]

    matches = list(FusionCBRPass().match(g))
    # Matcher may still detect chain; apply should no-op
    if matches:
        FusionCBRPass().apply(g, matches[0])
    assert [n.op_type for n in g.nodes] == ["Conv", "BatchNormalization", "Relu"]


