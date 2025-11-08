from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class RegisteredComponent:
    kind: str
    name: str
    factory: Callable[[], Any]


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, RegisteredComponent] = {}

    def register(self, kind: str, name: str, factory: Callable[[], Any]) -> None:
        key = f"{kind}:{name}"
        self._items[key] = RegisteredComponent(kind=kind, name=name, factory=factory)

    def get(self, kind: str, name: str) -> RegisteredComponent | None:
        return self._items.get(f"{kind}:{name}")

    def create(self, kind: str, name: str) -> Any:
        item = self.get(kind, name)
        if not item:
            raise KeyError(f"Component not found: {kind}:{name}")
        return item.factory()

global_registry = Registry()




