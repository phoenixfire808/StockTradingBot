"""Plugin registry — register/get/list discovered plugins."""

import logging
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Registry(Generic[T]):
    """Maps plugin names to instances; single source of truth for a plugin kind."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._store: dict[str, T] = {}

    def register(self, name: str, obj: T) -> None:
        if name in self._store:
            logger.warning("Duplicate %s '%s' — overwriting old registration", self._kind, name)
        self._store[name] = obj
        logger.debug("%s plugin registered: %s", self._kind, name)

    def get(self, name: str) -> T:
        if name not in self._store:
            raise KeyError(
                f"Unknown {self._kind}: '{name}'. "
                f"Available: {', '.join(sorted(self._store.keys())) or '(none)'}"
            )
        return self._store[name]

    def names(self) -> list[str]:
        return sorted(self._store.keys())

    def items(self) -> list[tuple[str, T]]:
        return [(n, self._store[n]) for n in self.names()]

    def all(self) -> list[T]:
        return [self._store[n] for n in self.names()]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._kind!r}, {len(self._store)} items)"
