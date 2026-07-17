"""Indexing stage: store chunk vectors + metadata and search them.

Implementations register in :data:`f1_rag.indexing.base.index_registry`
(``--index numpy`` / ``chroma``). The Chroma import is guarded so the package
stays importable without the optional dependency.
"""

from __future__ import annotations

from .base import StoreHit, VectorStore, index_registry
from .numpy_store import NumpyVectorStore  # noqa: F401  (self-registers)

try:  # optional dependency
    from . import chroma_store  # noqa: F401
except Exception:  # noqa: BLE001
    pass

__all__ = ["StoreHit", "VectorStore", "index_registry"]
