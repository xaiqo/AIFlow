from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass
class TunableConfig:
    """Represents a set of tunable parameters."""
    values: Dict[str, Any]


class CostModel(ABC):
    @abstractmethod
    def predict(self, config: TunableConfig, features: Mapping[str, float]) -> float:
        raise NotImplementedError


class SearchStrategy(ABC):
    @abstractmethod
    def suggest(self, previous: list[TunableConfig]) -> TunableConfig:
        raise NotImplementedError


