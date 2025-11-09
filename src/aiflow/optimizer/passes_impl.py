from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from aiflow.ir import build_consumer_map
from aiflow.ir.graph import Graph
from aiflow.optimizer.passes import Pass


def _get_const_tensor(graph: Graph, name: str) -> np.ndarray | None:
    t = graph.tensors.get(name)
    if t is None:
        return None
    val = t.metadata.get("const")
    if val is None:
        return None
    arr = np.array(val)
    return arr


class ConstantFoldingPass(Pass):
    """Fold ops whose inputs are all constants; write result to output tensor metadata."""

    def __init__(self) -> None:
        self._supported = {"Add", "Relu", "Transpose"}

    def match(self, graph: Graph) -> Iterable[int]:
        for idx, node in enumerate(graph.nodes):
            if node.op_type not in self._supported:
                continue
            # Skip nodes already folded to avoid infinite fixed-point loops
            if node.metadata.get("folded"):
                continue
            if not node.outputs:
                continue
            # All inputs must be constants
            inputs_are_const = all(
                _get_const_tensor(graph, name) is not None for name in node.inputs
            )
            if inputs_are_const:
                yield idx

    def apply(self, graph: Graph, candidate: int) -> None:
        node = graph.nodes[candidate]
        out_name = node.outputs[0]
        if node.op_type == "Add":
            a = _get_const_tensor(graph, node.inputs[0])
            b = _get_const_tensor(graph, node.inputs[1])
            if a is None or b is None:
                return
            graph.tensors[out_name].metadata["const"] = (a + b).tolist()
        elif node.op_type == "Relu":
            x = _get_const_tensor(graph, node.inputs[0])
            if x is None:
                return
            graph.tensors[out_name].metadata["const"] = np.maximum(x, 0).tolist()
        elif node.op_type == "Transpose":
            x = _get_const_tensor(graph, node.inputs[0])
            if x is None:
                return
            perm = node.attributes.get("perm")
            if perm is None:
                perm = list(reversed(range(x.ndim)))
            graph.tensors[out_name].metadata["const"] = np.transpose(
                x, axes=perm
            ).tolist()
        # Optionally mark node as folded
        node.metadata["folded"] = True


class DeadCodeEliminationPass(Pass):
    """Remove nodes whose outputs are not consumed and not graph outputs."""

    def match(self, graph: Graph) -> Iterable[list[int]]:
        consumers = build_consumer_map(graph)
        removable: list[int] = []
        graph_outputs = set(graph.outputs)
        for idx, node in enumerate(graph.nodes):
            if not node.outputs:
                continue
            # If all outputs are unconsumed and not graph outputs -> removable
            outputs_unused = all(
                (not consumers.get(out)) and (out not in graph_outputs)
                for out in node.outputs
            )
            if not outputs_unused:
                continue
            # Heuristic: only remove if each input is also consumed by another node,
            # so we don't prune the main chain terminal node.
            safe_inputs = True
            for inp in node.inputs:
                use_list = consumers.get(inp, [])
                # At least one other consumer besides this node
                others = [i for i in use_list if i != idx]
                if not others:
                    safe_inputs = False
                    break
            if safe_inputs:
                removable.append(idx)
        if removable:
            yield removable

    def apply(self, graph: Graph, candidate: list[int]) -> None:
        # Remove nodes in reverse order to keep indices stable
        for idx in sorted(candidate, reverse=True):
            del graph.nodes[idx]
