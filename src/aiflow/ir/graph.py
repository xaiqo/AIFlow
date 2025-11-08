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

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_tensor(self, tensor: Tensor) -> None:
        self.tensors[tensor.name] = tensor




