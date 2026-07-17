"""Chunker interface + shared helpers.

The :class:`Chunker` protocol defines the single method every chunking strategy
must implement: turn a list of :class:`Article` records (plus their document meta
and optional page-image map) into a list of :class:`Chunk`. Keeping the interface
this small is what makes strategies trivially swappable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import Article, Chunk, SourceDocumentMeta
from ..registry import Registry

# Registry is the DI seam: implementations register under a short name.
chunker_registry: Registry["Chunker"] = Registry("chunker")


def estimate_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate.

    We approximate tokens as whitespace words scaled by 1.3 (subword models emit
    slightly more tokens than words). This is only used for budgeting chunk sizes
    and context windows, so an estimate is sufficient and keeps chunking free of a
    tokenizer dependency.
    """

    words = len(text.split())
    return int(words * 1.3) + 1


@runtime_checkable
class Chunker(Protocol):
    """Turns parsed articles into retrievable chunks."""

    name: str

    def chunk(
        self,
        articles: list[Article],
        meta: SourceDocumentMeta,
        page_image_paths: dict[int, str] | None = None,
    ) -> list[Chunk]:
        ...
