"""Regulation-aware chunker.

What it does
------------
Creates one chunk per article/subarticle/provision record, splitting a record into
overlapping sub-chunks *only* when its text exceeds the token budget. Chunk ids are
deterministic (``make_chunk_id``) so re-ingesting is idempotent.

Why it exists
-------------
Regulation text is already authored in a clean hierarchy (articles -> subarticles
-> numbered provisions). Chunking at those boundaries keeps each chunk aligned to a
single citable reference, which improves both retrieval precision and citation
quality versus arbitrary fixed windows.

Assumptions
-----------
- Most provisions fit within ``max_tokens``; overlap-splitting is the exception,
  not the rule.

Limitations
-----------
- Very short subarticle *headings* with no body are skipped (nothing to retrieve).

Replaceable alternatives
-------------------------
- :mod:`f1_rag.chunking.fixed_window` ignores structure entirely; comparing the two
  is a core experiment this test bed supports.
"""

from __future__ import annotations

from ..logging_utils import get_logger
from ..models import Article, Chunk, SourceDocumentMeta, make_chunk_id
from .base import Chunker, chunker_registry, estimate_tokens

logger = get_logger(__name__)


def _split_overlapping(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split ``text`` into overlapping windows measured in whitespace words.

    Overlap preserves context across a boundary so a provision cut mid-clause is
    still retrievable from either side. Uses words as the unit (see
    ``estimate_tokens`` for why an approximation is fine here).
    """

    words = text.split()
    if not words:
        return []
    # Convert token budgets to word budgets using the same 1.3 factor.
    max_words = max(1, int(max_tokens / 1.3))
    overlap_words = max(0, min(int(overlap_tokens / 1.3), max_words - 1))
    if len(words) <= max_words:
        return [text]

    windows: list[str] = []
    start = 0
    step = max_words - overlap_words
    while start < len(words):
        windows.append(" ".join(words[start : start + max_words]))
        start += step
    return windows


@chunker_registry.register("regulation")
class RegulationChunker(Chunker):
    name = "regulation"

    def __init__(self, max_tokens: int = 350, overlap_tokens: int = 60) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(
        self,
        articles: list[Article],
        meta: SourceDocumentMeta,
        page_image_paths: dict[int, str] | None = None,
    ) -> list[Chunk]:
        page_image_paths = page_image_paths or {}
        chunks: list[Chunk] = []

        for article in articles:
            body = article.text.strip()
            if not body:
                continue  # heading-only record: nothing retrievable
            pieces = _split_overlapping(body, self.max_tokens, self.overlap_tokens)

            char_cursor = 0
            for idx, piece in enumerate(pieces):
                char_start = article.text.find(piece[:40], char_cursor)
                if char_start < 0:
                    char_start = char_cursor
                char_end = char_start + len(piece)
                char_cursor = max(char_cursor, char_end - 1)

                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(meta.source_filename, article.article_number, idx),
                        text=piece,
                        section=article.section,
                        article_number=article.article_number,
                        parent_article=article.parent_article,
                        article_heading=article.heading,
                        pdf_page_number=article.pdf_page_number,
                        page_label=article.page_label,
                        source_filename=meta.source_filename,
                        document_title=meta.document_title,
                        issue_number=meta.issue_number,
                        publication_date=meta.publication_date,
                        page_image_path=page_image_paths.get(article.pdf_page_number),
                        chunk_index=idx,
                        char_start=char_start,
                        char_end=char_end,
                        token_estimate=estimate_tokens(piece),
                    )
                )

        logger.info(
            "regulation chunker: %d chunks from %d articles (%s)",
            len(chunks),
            len(articles),
            meta.source_filename,
        )
        return chunks
