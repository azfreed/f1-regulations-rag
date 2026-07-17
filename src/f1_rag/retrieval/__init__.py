"""Retrieval stage: find candidate chunks for a query.

Strategies register in :data:`f1_rag.retrieval.base.retriever_registry`
(``--retriever vector`` / ``bm25`` / ``hybrid``). Each factory takes a
:class:`RetrievalDeps` bundle so the CLI can construct any strategy uniformly.
"""

from __future__ import annotations

from . import hybrid, keyword, vector  # noqa: F401  (self-register)
from .base import RetrievalDeps, Retriever, normalize_query, retriever_registry

__all__ = ["RetrievalDeps", "Retriever", "normalize_query", "retriever_registry"]
