"""Retriever interface, shared dependencies, and score utilities.

Every retriever takes a query and returns ranked :class:`RetrievedCandidate`
objects with all available scores populated, so the tracing layer can show raw
distances, similarities, and keyword scores without the retriever knowing about
tracing.

Query normalization
-------------------
We lightly normalize the query (collapse whitespace, strip, keep case for the
embedder which is case-robust). Both the raw and normalized query are recorded in
the trace so you can see exactly what was embedded/searched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..models import Chunk, RetrievedCandidate
from ..registry import Registry

retriever_registry: Registry["Retriever"] = Registry("retriever")


@dataclass
class RetrievalDeps:
    """Everything a retriever might need, supplied by the CLI/orchestrator.

    Not every retriever uses every field: the vector retriever needs ``store`` and
    ``embedder``; BM25 needs ``chunks``; hybrid needs both.
    """

    chunks: list[Chunk] = field(default_factory=list)
    store: object | None = None  # VectorStore
    embedder: object | None = None  # Embedder
    top_k: int = 8


def normalize_query(query: str) -> str:
    """Collapse whitespace and strip. Kept deliberately minimal and visible."""

    return re.sub(r"\s+", " ", query).strip()


def min_max_normalize(scores: list[float]) -> list[float]:
    """Scale scores to [0, 1] so heterogeneous score types can be combined.

    Used by hybrid retrieval to put cosine similarities and BM25 scores on a common
    scale before weighting. If all scores are equal, returns all zeros.
    """

    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return ((arr - lo) / (hi - lo)).tolist()


@runtime_checkable
class Retriever(Protocol):
    name: str

    def retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedCandidate]:
        ...
