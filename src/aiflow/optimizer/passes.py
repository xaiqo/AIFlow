from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Protocol

from aiflow.ir.graph import Graph, Node


class Candidate(Protocol):
    """A pass-specific candidate match object."""
    ...


class Pass(ABC):
    """Base class for graph passes."""

    @abstractmethod
    def match(self, graph: Graph) -> Iterable[Candidate]:
        raise NotImplementedError

    @abstractmethod
    def apply(self, graph: Graph, candidate: Candidate) -> None:
        raise NotImplementedError


class Pipeline:
    """An ordered sequence of passes."""

    def __init__(self, passes: List[Pass]) -> None:
        self._passes = passes

    def run(self, graph: Graph) -> Graph:
        for p in self._passes:
            for c in list(p.match(graph)):
                p.apply(graph, c)
        return graph


