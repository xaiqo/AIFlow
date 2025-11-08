from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class BackendCapabilities:
    name: str
    vector_width: int | None = None
    has_tensor_cores: bool = False
    max_threads: int | None = None
    metadata: dict[str, str] | None = None


class Backend(Protocol):
    """Interface for kernel backends."""

    def capabilities(self) -> BackendCapabilities: ...

    def generate(self, schedule: Schedule) -> Artifact: ...


@dataclass
class Schedule:
    """Skeleton schedule representation."""
    description: str


@dataclass
class Artifact:
    """Kernel artifact container."""
    source: str
    metadata: dict[str, str]




