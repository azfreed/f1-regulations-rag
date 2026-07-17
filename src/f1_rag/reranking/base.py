"""Reranker interface.

A reranker takes the query and the retrieved candidates and returns a (possibly)
reordered list, setting ``rerank_score`` on each candidate for the trace. Keeping
reranking as its own stage lets us measure retrieval with and without it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import RetrievedCandidate
from ..registry import Registry

reranker_registry: Registry["Reranker"] = Registry("reranker")


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, candidates: list[RetrievedCandidate]
    ) -> list[RetrievedCandidate]:
        ...
