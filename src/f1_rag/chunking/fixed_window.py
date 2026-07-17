"""Fixed-window chunker (structure-agnostic baseline).

What it does
------------
Concatenates all article text for a document and slices it into fixed-size
overlapping windows, ignoring article boundaries. Each window still records the
article it *starts* in, so citations remain approximately meaningful.

Why it exists
-------------
This is the classic RAG baseline. Comparing it against the regulation-aware chunker
demonstrates how structure-aware chunking affects retrieval/citation quality - a
primary experiment this test bed is built to run.

Limitations
-----------
- Chunks can straddle two articles, so citations are less precise than the
  regulation chunker's.
"""

from __future__ import annotations

from ..logging_utils import get_logger
from ..models import Article, Chunk, SourceDocumentMeta, make_chunk_id
from .base import Chunker, chunker_registry, estimate_tokens

logger = get_logger(__name__)


@chunker_registry.register("fixed_window")
class FixedWindowChunker(Chunker):
    name = "fixed_window"

    def __init__(self, max_tokens: int = 350, overlap_tokens: int = 60) -> None:
        self.max_words = max(1, int(max_tokens / 1.3))
        self.overlap_words = max(0, min(int(overlap_tokens / 1.3), self.max_words - 1))

    def chunk(
        self,
        articles: list[Article],
        meta: SourceDocumentMeta,
        page_image_paths: dict[int, str] | None = None,
    ) -> list[Chunk]:
        page_image_paths = page_image_paths or {}

        # Build a flat word stream, remembering which article each word came from so
        # a window can be attributed to the article it begins in.
        words: list[str] = []
        origin: list[Article] = []
        for article in articles:
            body = article.text.strip()
            if not body:
                continue
            for w in body.split():
                words.append(w)
                origin.append(article)

        chunks: list[Chunk] = []
        step = self.max_words - self.overlap_words
        start = 0
        idx = 0
        while start < len(words):
            end = min(start + self.max_words, len(words))
            window_words = words[start:end]
            anchor = origin[start]
            text = " ".join(window_words)
            chunks.append(
                Chunk(
                    # Namespace the id under a synthetic per-window article key so ids
                    # stay deterministic and never collide with the regulation chunker.
                    chunk_id=make_chunk_id(meta.source_filename, f"fw:{anchor.article_number}", idx),
                    text=text,
                    section=anchor.section,
                    article_number=anchor.article_number,
                    parent_article=anchor.parent_article,
                    article_heading=anchor.heading,
                    pdf_page_number=anchor.pdf_page_number,
                    page_label=anchor.page_label,
                    source_filename=meta.source_filename,
                    document_title=meta.document_title,
                    issue_number=meta.issue_number,
                    publication_date=meta.publication_date,
                    page_image_path=page_image_paths.get(anchor.pdf_page_number),
                    chunk_index=idx,
                    token_estimate=estimate_tokens(text),
                )
            )
            idx += 1
            start += step

        logger.info(
            "fixed_window chunker: %d chunks (%s)", len(chunks), meta.source_filename
        )
        return chunks
