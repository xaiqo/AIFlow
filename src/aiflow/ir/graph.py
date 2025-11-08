from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Tensor:
    name: str
    dtype: str
    shape: List[int]
    layout: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    nodes: List[Node] = field(default_factory=list)
    tensors: Dict[str, Tensor] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_tensor(self, tensor: Tensor) -> None:
        self.tensors[tensor.name] = tensor


