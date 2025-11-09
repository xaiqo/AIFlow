from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Protocol

from aiflow.ir.graph import Graph


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

    def __init__(self, passes: list[Pass]) -> None:
        self._passes = passes

    def run(self, graph: Graph) -> Graph:
        for p in self._passes:
            # Run each pass to a fixed point
            while True:
                candidates = list(p.match(graph))
                if not candidates:
                    break
                for c in candidates:
                    p.apply(graph, c)
        return graph




