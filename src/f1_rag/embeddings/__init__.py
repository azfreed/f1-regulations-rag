"""Embedding stage: turn chunk text into dense vectors.

Implementations register in :data:`f1_rag.embeddings.base.embedder_registry`
(``--embedder minilm`` / ``hashing``). Importing the package self-registers them.
The sentence-transformer import is guarded so the package remains importable even
if the optional dependency is not installed.
"""

from __future__ import annotations

from .base import Embedder, embedder_registry
from .hashing import HashingEmbedder  # noqa: F401  (self-registers, no heavy deps)

try:  # optional heavy dependency (torch + sentence-transformers)
    from . import sentence_transformer  # noqa: F401
except Exception:  # noqa: BLE001 - keep package importable without the extra
    pass

__all__ = ["Embedder", "embedder_registry"]
