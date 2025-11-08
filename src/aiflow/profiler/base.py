from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping


class ProfileResult:
    def __init__(self, metrics: Mapping[str, float]) -> None:
        self.metrics = dict(metrics)


class Profiler(ABC):
    """Base class for profiling."""

    @abstractmethod
    def run(self, artifact: object) -> ProfileResult:
        raise NotImplementedError




