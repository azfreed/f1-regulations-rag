"""Vector store interface.

A vector store owns (a) the chunk metadata and (b) the embedding matrix, and knows
how to persist itself, reload, and run a top-k similarity search with optional
metadata filtering. Retrieval strategies sit *on top* of a store; the store itself
only does nearest-neighbour math.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..models import Chunk
from ..registry import Registry

index_registry: Registry["VectorStore"] = Registry("index")


@dataclass(slots=True)
class StoreHit:
    """One nearest-neighbour result from a store."""

    chunk: Chunk
    similarity: float  # cosine similarity in [-1, 1] (higher = closer)
    distance: float  # cosine distance = 1 - similarity (lower = closer)
    index: int  # row index in the embedding matrix (for diagnostics)


# A metadata filter is a mapping of chunk-attribute -> allowed value(s).
MetadataFilter = dict[str, object]


@runtime_checkable
class VectorStore(Protocol):
    name: str

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Populate the store from chunks and their (normalized) embeddings."""
        ...

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[StoreHit]:
        ...

    def save(self, path: str | Path) -> None:
        ...

    def load(self, path: str | Path) -> None:
        ...

    def __len__(self) -> int:
        ...


def matches_filter(chunk: Chunk, metadata_filter: MetadataFilter | None) -> bool:
    """Return True if ``chunk`` satisfies every key in ``metadata_filter``.

    Each value may be a scalar (equality) or a collection (membership). This is the
    "filtering" step of retrieval: we restrict the candidate set by metadata (e.g.
    only Section D) *before* or *after* scoring, keeping the operation explicit.
    """

    if not metadata_filter:
        return True
    for key, allowed in metadata_filter.items():
        value = getattr(chunk, key, None)
        # Enums compare by their .value for convenience.
        if hasattr(value, "value"):
            value = value.value
        if isinstance(allowed, (list, tuple, set)):
            if value not in allowed:
                return False
        elif value != allowed:
            return False
    return True
