"""Typed core models shared across every pipeline stage.

These Pydantic models are the *contracts* between stages. Because each stage only
depends on these models (not on other stages' implementations), stages stay
loosely coupled and independently testable.

Design notes
------------
- Every model is JSON-serializable so intermediate artifacts can be written to
  disk (JSONL) and inspected.
- ``Chunk.chunk_id`` is *deterministic*: the same source + article + index always
  produces the same id, which makes ingestion idempotent and lets us diff runs.
- Visual metadata is preserved from the start (page images, bounding boxes) even
  though multimodal retrieval is not yet implemented, so future stages can surface
  the original page, a figure, nearby text, and captions.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class RegulationSection(str, Enum):
    """The six FIA F1 regulation sections, keyed by their letter."""

    A = "A"  # General Regulatory Provisions
    B = "B"  # Sporting
    C = "C"  # Technical
    D = "D"  # Financial (F1 Teams)
    E = "E"  # Financial (Power Unit)
    F = "F"  # Operational


class VisualKind(str, Enum):
    DIAGRAM = "diagram"
    DRAWING = "drawing"
    TABLE = "table"
    EQUATION = "equation"
    IMAGE = "image"


# ---------------------------------------------------------------------------
# Source document + pages (extraction stage output)
# ---------------------------------------------------------------------------
class SourceDocumentMeta(BaseModel):
    """Document-level metadata detected from the PDF's front matter + filename."""

    source_filename: str
    section: RegulationSection
    document_title: str
    issue_number: int | None = None
    publication_date: str | None = None  # ISO-ish string as printed (DD/MM/YYYY normalized)
    wmsc_approval_date: str | None = None
    page_count: int


class TextBlock(BaseModel):
    """A positioned block of text from PyMuPDF, used for header/footer detection."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float


class RawPage(BaseModel):
    """Raw, pre-cleanup output for a single PDF page."""

    pdf_page_number: int  # 1-based index within the PDF
    page_label: str | None = None  # e.g. "A6", "C15" (from the footer)
    raw_text: str
    blocks: list[TextBlock] = Field(default_factory=list)
    width: float
    height: float


class RawDocument(BaseModel):
    """Everything extraction produces for one PDF, before cleanup/parsing."""

    meta: SourceDocumentMeta
    pages: list[RawPage]


# ---------------------------------------------------------------------------
# Visual assets (visual stage output; kept separate from text)
# ---------------------------------------------------------------------------
class ExtractedImage(BaseModel):
    """A raster/vector image embedded in a page, with its bounding box if known."""

    image_path: str
    bbox: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1)
    xref: int | None = None


class PageVisual(BaseModel):
    """Visual metadata for one page: a rendered page image + detection flags."""

    pdf_page_number: int
    page_image_path: str
    has_diagram: bool = False
    has_drawing: bool = False
    has_table: bool = False
    has_equation: bool = False
    has_image: bool = False
    drawing_path_count: int = 0
    image_count: int = 0
    extracted_images: list[ExtractedImage] = Field(default_factory=list)
    kinds: list[VisualKind] = Field(default_factory=list)

    @property
    def is_likely_visual(self) -> bool:
        return bool(self.kinds)


class VisualDocument(BaseModel):
    source_filename: str
    section: RegulationSection
    pages: list[PageVisual]


# ---------------------------------------------------------------------------
# Cleaned pages (cleanup stage output)
# ---------------------------------------------------------------------------
class CleanPage(BaseModel):
    """A page after header/footer removal and TOC flagging."""

    pdf_page_number: int
    page_label: str | None = None
    text: str
    is_toc: bool = False
    removed_lines: list[str] = Field(default_factory=list)  # diagnostics


# ---------------------------------------------------------------------------
# Parsed structure (parsing stage output)
# ---------------------------------------------------------------------------
class Article(BaseModel):
    """A regulation article or subarticle with its text and provenance.

    We keep a flat list of articles (each carrying its ``parent_article``) rather
    than a nested tree, because the flat form is what chunking consumes and is
    trivially serializable.
    """

    section: RegulationSection
    article_number: str  # e.g. "A1", "A1.2", "A1.2.2"
    parent_article: str | None = None  # e.g. "A1" for "A1.2"
    heading: str | None = None
    text: str
    pdf_page_number: int  # page where the article/subarticle begins
    page_label: str | None = None
    source_filename: str


# ---------------------------------------------------------------------------
# Chunks (chunking stage output)
# ---------------------------------------------------------------------------
def make_chunk_id(source_filename: str, article_number: str, chunk_index: int) -> str:
    """Deterministic chunk id.

    Same (file, article, index) -> same id on every run. This is what makes
    ingestion idempotent: re-ingesting an unchanged corpus produces identical ids
    so the index can be rebuilt or diffed reliably.
    """

    raw = f"{source_filename}|{article_number}|{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class Chunk(BaseModel):
    """A retrievable unit of text plus the metadata needed for filtering + citation."""

    chunk_id: str
    text: str
    # --- provenance / metadata (used for filters + citations) ---
    section: RegulationSection
    article_number: str
    parent_article: str | None = None
    article_heading: str | None = None
    pdf_page_number: int
    page_label: str | None = None
    source_filename: str
    document_title: str
    issue_number: int | None = None
    publication_date: str | None = None
    # --- visual linkage (for future multimodal surfacing) ---
    page_image_path: str | None = None
    # --- chunking bookkeeping ---
    chunk_index: int = 0
    char_start: int | None = None
    char_end: int | None = None
    token_estimate: int = 0

    def citation_label(self) -> str:
        """Short human-readable citation, e.g. ``Art. A1.2.2 (Section A, p.A6)``."""

        page = self.page_label or f"pdf p.{self.pdf_page_number}"
        return f"Art. {self.article_number} (Section {self.section.value}, {page})"


class EmbeddedChunk(BaseModel):
    """A chunk paired with its embedding vector (kept out of the searchable JSONL)."""

    chunk: Chunk
    embedding: list[float]


# ---------------------------------------------------------------------------
# Retrieval + context + answers
# ---------------------------------------------------------------------------
class RetrievedCandidate(BaseModel):
    """A chunk returned by retrieval, annotated with every available score.

    All scores are optional because different retrievers populate different fields
    (vector distance vs BM25 vs reranker). The tracing layer records them all.
    """

    chunk: Chunk
    score: float  # the primary ranking score used to order candidates
    vector_distance: float | None = None  # raw distance (lower = closer)
    similarity: float | None = None  # cosine similarity (higher = closer)
    keyword_score: float | None = None  # BM25
    rerank_score: float | None = None
    rank: int | None = None
    retriever: str | None = None


class DiscardedChunk(BaseModel):
    """A candidate that was retrieved but not placed into the context, with a reason."""

    chunk_id: str
    citation: str
    reason: str


class AssembledContext(BaseModel):
    """The context block handed to the generator, plus what was dropped and why."""

    text: str
    selected: list[RetrievedCandidate]
    discarded: list[DiscardedChunk] = Field(default_factory=list)
    token_estimate: int = 0


class Citation(BaseModel):
    chunk_id: str
    label: str
    article_number: str
    section: RegulationSection
    page_label: str | None = None
    pdf_page_number: int
    source_filename: str


class Answer(BaseModel):
    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    is_unanswerable: bool = False
    generation_model: str | None = None
    prompt_version: str | None = None


# ---------------------------------------------------------------------------
# Small JSONL helpers (used across stages for inspectable artifacts)
# ---------------------------------------------------------------------------
def write_jsonl(path: Any, models: list[BaseModel]) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for m in models:
            fh.write(m.model_dump_json())
            fh.write("\n")


def read_jsonl(path: Any, model_cls: type[BaseModel]) -> list[Any]:
    from pathlib import Path

    p = Path(path)
    out: list[Any] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(model_cls.model_validate_json(line))
    return out
