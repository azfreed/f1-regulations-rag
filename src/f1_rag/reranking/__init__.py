"""Reranking stage: optionally reorder candidates with a stronger model.

Implementations register in :data:`f1_rag.reranking.base.reranker_registry`
(``--reranker none``). Only an identity (no-op) reranker ships in this milestone; a
cross-encoder reranker can be added later without touching retrieval or context.
"""

from __future__ import annotations

from . import identity  # noqa: F401  (self-register)
from .base import Reranker, reranker_registry

__all__ = ["Reranker", "reranker_registry"]
