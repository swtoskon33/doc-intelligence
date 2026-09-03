"""Serving aliases: point champion / challenger at extraction backends.

Promotion is an alias flip, not a redeploy: the champion serves production traffic while
a challenger can be exercised on the side, and rollback is flipping the alias back. The
aliases are read from the environment so a deployment changes them without code changes.
"""
from __future__ import annotations

import os

from doc_intelligence.extraction.base import Extractor
from doc_intelligence.extraction.extractor import get_extractor

DEFAULT_ALIASES = {"champion": "rule", "challenger": "llm"}


class AliasRegistry:
    """Maps serving aliases to extraction backends, with lazy instantiation."""

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self.aliases = aliases or {
            "champion": os.getenv("CHAMPION_BACKEND", DEFAULT_ALIASES["champion"]),
            "challenger": os.getenv("CHALLENGER_BACKEND", DEFAULT_ALIASES["challenger"]),
        }
        self._instances: dict[str, Extractor] = {}

    def backend_for(self, alias: str) -> str:
        if alias not in self.aliases:
            raise KeyError(f"unknown alias: {alias}")
        return self.aliases[alias]

    def get(self, alias: str) -> Extractor:
        """Return the extractor an alias points at, instantiated once."""
        backend = self.backend_for(alias)
        if backend not in self._instances:
            self._instances[backend] = get_extractor(backend)
        return self._instances[backend]

    def promote(self, alias: str, backend: str) -> None:
        """Repoint an alias at a different backend (the promotion / rollback primitive)."""
        self.aliases[alias] = backend
        self._instances.pop(backend, None)

    def as_dict(self) -> dict[str, str]:
        return dict(self.aliases)
