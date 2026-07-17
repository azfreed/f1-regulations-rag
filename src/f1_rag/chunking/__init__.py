"""Chunking stage: split articles into retrievable chunks.

Implementations register themselves in :data:`f1_rag.chunking.base.chunker_registry`
so the CLI can select one by name (``--chunker regulation`` / ``fixed_window``).
Importing this package imports the implementations so they self-register.
"""

from __future__ import annotations

from . import fixed_window, regulation  # noqa: F401  (import for registration side-effect)
from .base import Chunker, chunker_registry

__all__ = ["Chunker", "chunker_registry"]
