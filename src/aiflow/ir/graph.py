from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: list[int]
    layout: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    tensors: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_tensor(self, tensor: Tensor) -> None:
        self.tensors[tensor.name] = tensor

    def get_tensor(self, name: str) -> Tensor | None:
        return self.tensors.get(name)


class ValidationError(Exception):
    """Graph validation error with optional code and context."""

    def __init__(
        self, message: str, code: str = "EVALID", node_index: int | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.node_index = node_index


class GraphValidator:
    """Validates basic HIR invariants and provides graph utilities like toposort."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def validate(self) -> None:
        self._validate_tensors_typed()
        producer_map = self._build_producer_map()
        self._validate_unique_producers(producer_map)
        self._validate_node_io_exist()
        self._validate_inputs_outputs_exist()
        self._topological_order(producer_map)  # raises on cycles

    def _validate_tensors_typed(self) -> None:
        for name, t in self.graph.tensors.items():
            if not t.dtype or not isinstance(t.dtype, str):
                raise ValidationError(
                    f"Tensor '{name}' missing dtype", code="ETENSOR_DTYPE"
                )
            if t.shape is None or not isinstance(t.shape, list):
                raise ValidationError(
                    f"Tensor '{name}' missing shape", code="ETENSOR_SHAPE"
                )
            for dim in t.shape:
                if not isinstance(dim, int) or dim <= 0:
                    raise ValidationError(
                        f"Tensor '{name}' has invalid shape {t.shape}",
                        code="ETENSOR_SHAPE",
                    )

    def _build_producer_map(self) -> dict[str, int]:
        """Map tensor name -> producing node index. Graph inputs have no producer."""
        producer: dict[str, int] = {}
        for idx, node in enumerate(self.graph.nodes):
            for out in node.outputs:
                if out in producer:
                    raise ValidationError(
                        f"Multiple producers for tensor '{out}' at node {idx} and {producer[out]}",
                        code="EDUP_PRODUCER",
                        node_index=idx,
                    )
                producer[out] = idx
        return producer

    def _validate_unique_producers(self, producer_map: dict[str, int]) -> None:
        # Producer map creation already enforces uniqueness; nothing else here.
        return

    def _validate_node_io_exist(self) -> None:
        for idx, node in enumerate(self.graph.nodes):
            for name in node.inputs:
                if name not in self.graph.tensors:
                    raise ValidationError(
                        f"Node {idx} input '{name}' not found in tensors",
                        code="EINPUT_MISSING",
                        node_index=idx,
                    )
            for name in node.outputs:
                if name not in self.graph.tensors:
                    raise ValidationError(
                        f"Node {idx} output '{name}' not found in tensors",
                        code="EOUTPUT_MISSING",
                        node_index=idx,
                    )

    def _validate_inputs_outputs_exist(self) -> None:
        for name in self.graph.inputs:
            if name not in self.graph.tensors:
                raise ValidationError(
                    f"Graph input '{name}' missing tensor", code="EGRAPH_INPUT"
                )
        for name in self.graph.outputs:
            if name not in self.graph.tensors:
                raise ValidationError(
                    f"Graph output '{name}' missing tensor", code="EGRAPH_OUTPUT"
                )

    def _topological_order(
        self, producer_map: dict[str, int] | None = None
    ) -> list[int]:
        """
        Return topological order of node indices. Raise ValidationError on cycles.
        """
        if producer_map is None:
            producer_map = self._build_producer_map()

        indegree: list[int] = [0] * len(self.graph.nodes)
        adj: dict[int, set[int]] = {i: set() for i in range(len(self.graph.nodes))}

        # Build edges: u -> v if v consumes a tensor produced by u
        for v_idx, node in enumerate(self.graph.nodes):
            for inp in node.inputs:
                u_idx = producer_map.get(inp)
                if u_idx is not None:
                    if v_idx not in adj[u_idx]:
                        adj[u_idx].add(v_idx)
                        indegree[v_idx] += 1

        # Kahn's algorithm
        queue: list[int] = [i for i, d in enumerate(indegree) if d == 0]
        order: list[int] = []
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in list(adj[u]):
                indegree[v] -= 1
                adj[u].remove(v)
                if indegree[v] == 0:
                    queue.append(v)

        if len(order) != len(self.graph.nodes):
            raise ValidationError("Cycle detected in graph", code="ECYCLE")
        return order

    def toposort(self) -> list[Node]:
        order = self._topological_order()
        return [self.graph.nodes[i] for i in order]
