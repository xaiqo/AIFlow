from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Parser(ABC):
    """Parser interface for importing models into Graph IR."""

    @abstractmethod
    def parse(self, model: Any) -> GraphIR:
        """Convert the given model into a GraphIR."""
        raise NotImplementedError


# Avoid circular imports: typed as string above
class GraphIR:  # minimal placeholder for type reference
    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = metadata or {}




