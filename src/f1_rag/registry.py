"""A tiny name -> factory registry: the project's dependency-injection mechanism.

Why this exists
---------------
The CLI must be able to pick implementations by name (``--chunker regulation``,
``--index numpy``) without importing concrete classes directly. A registry keeps
that coupling loose: each stage owns a ``Registry`` instance, implementations
register themselves with ``@registry.register("name")``, and the CLI resolves a
name to a factory at runtime.

This is deliberately ~40 lines instead of a heavy DI framework. It is easy to
read, easy to test, and easy to replace.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

from .errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a short string name to a zero-or-more-arg factory callable."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that registers a factory (usually a class) under ``name``."""

        def _decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise RegistryError(f"{self.kind} '{name}' is already registered")
            self._factories[name] = factory
            return factory

        return _decorator

    def create(self, name: str, /, *args, **kwargs) -> T:
        """Instantiate the implementation registered under ``name``."""

        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise RegistryError(
                f"unknown {self.kind} '{name}'. available: {', '.join(self.available()) or '(none)'}"
            ) from exc
        return factory(*args, **kwargs)

    def available(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories
