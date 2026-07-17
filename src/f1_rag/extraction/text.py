"""PDF text extraction with PyMuPDF.

What it does
------------
Reads each page of a regulation PDF into a :class:`RawPage` (raw text + positioned
text blocks) and detects document-level metadata (issue number, publication date,
WMSC approval date, document title, section) from the PDF's front matter and
filename. Persists the result to ``data/extracted/<section>.raw.jsonl`` so the raw
extraction output is stored *before* any parsing.

Why it exists
-------------
This is stage 1 of the pipeline. Isolating raw extraction means we can inspect
exactly what came out of the PDF before any cleanup/parsing decisions are applied.

Assumptions
-----------
- The 2026 FIA F1 Regulation PDFs are Word-exported with a highly regular layout
  (verified across all six sections): a page-1 metadata block ``Version: Issue NN``
  / ``Date: DD/MM/YYYY`` / ``WMSC approval date: ...`` and a two-line footer whose
  trailing token is the page label (``A6``, ``C15``).

Limitations
-----------
- Text order follows PyMuPDF's block order. For heavily multi-column or figure-rich
  pages (Section C technical drawings), reading order may be imperfect; the visual
  stage compensates by preserving page images.

Replaceable alternatives
-------------------------
- A different extractor (pdfminer, pdfplumber, or OCR for scanned inputs) could
  produce ``RawDocument`` without changing any downstream stage.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from ..errors import ExtractionError
from ..logging_utils import get_logger
from ..models import (
    RawDocument,
    RawPage,
    RegulationSection,
    SourceDocumentMeta,
    TextBlock,
)

logger = get_logger(__name__)

# Filename convention: section-a-general.pdf -> section A
_FILENAME_SECTION_RE = re.compile(r"section-([a-f])", re.IGNORECASE)

# Front-matter metadata patterns (verified against all six PDFs).
_ISSUE_RE = re.compile(r"Issue\s+(\d+)", re.IGNORECASE)
_DATE_RE = re.compile(r"Date:\s*(\d{2}/\d{2}/\d{4})", re.IGNORECASE)
_WMSC_RE = re.compile(r"WMSC approval date:\s*(\d{2}/\d{2}/\d{4})", re.IGNORECASE)
# Header line: "SECTION A: GENERAL REGULATORY PROVISIONS ... 0A"
_SECTION_HEADER_RE = re.compile(r"SECTION\s+([A-F]):\s+(.+?)(?:\s{2,}\S+)?$", re.MULTILINE)
# Page label: a line that is *only* a token like "A6" / "C15". In PyMuPDF text
# order this appears near the top of the page (just after the "0  A" band).
_PAGE_LABEL_RE = re.compile(r"^([A-F]\d{1,3})$")


def _detect_section(filename: str, first_page_text: str) -> RegulationSection:
    m = _FILENAME_SECTION_RE.search(filename)
    if m:
        return RegulationSection(m.group(1).upper())
    m2 = _SECTION_HEADER_RE.search(first_page_text)
    if m2:
        return RegulationSection(m2.group(1).upper())
    raise ExtractionError(f"could not determine regulation section for '{filename}'")


def _detect_title(section: RegulationSection, pdf_title: str | None, first_page_text: str) -> str:
    """Prefer the printed section header; fall back to embedded PDF title."""

    m = _SECTION_HEADER_RE.search(first_page_text)
    if m:
        return f"SECTION {m.group(1).upper()}: {m.group(2).strip()}"
    if pdf_title:
        return pdf_title.strip()
    return f"SECTION {section.value}"


def _normalize_date(ddmmyyyy: str | None) -> str | None:
    if not ddmmyyyy:
        return None
    d, m, y = ddmmyyyy.split("/")
    # Keep the printed form but normalize to a stable ISO date for downstream use.
    return f"{y}-{m}-{d}"


def _page_label(page_text: str) -> str | None:
    """Extract the printed page label (e.g. ``A6``) from the header band.

    The label is a standalone ``<letter><number>`` token that appears within the
    first several lines of the page (right below the ``0  A`` section band). We scan
    only the top lines to avoid matching tokens like ``F1`` inside body prose.
    """

    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    for line in lines[:12]:
        m = _PAGE_LABEL_RE.match(line)
        if m:
            return m.group(1)
    return None


def extract_document(pdf_path: str | Path) -> RawDocument:
    """Extract a single PDF into a :class:`RawDocument`."""

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise ExtractionError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 - surface as our own error type
        raise ExtractionError(f"failed to open {pdf_path}: {exc}") from exc

    with doc:
        pages: list[RawPage] = []
        first_page_text = doc[0].get_text("text") if doc.page_count else ""
        section = _detect_section(pdf_path.name, first_page_text)
        pdf_title = doc.metadata.get("title") if doc.metadata else None
        title = _detect_title(section, pdf_title, first_page_text)

        issue = _ISSUE_RE.search(first_page_text)
        date = _DATE_RE.search(first_page_text)
        wmsc = _WMSC_RE.search(first_page_text)

        meta = SourceDocumentMeta(
            source_filename=pdf_path.name,
            section=section,
            document_title=title,
            issue_number=int(issue.group(1)) if issue else None,
            publication_date=_normalize_date(date.group(1)) if date else None,
            wmsc_approval_date=_normalize_date(wmsc.group(1)) if wmsc else None,
            page_count=doc.page_count,
        )

        for i, page in enumerate(doc):
            text = page.get_text("text")
            blocks_raw = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
            blocks = [
                TextBlock(
                    text=b[4].strip(),
                    x0=float(b[0]),
                    y0=float(b[1]),
                    x1=float(b[2]),
                    y1=float(b[3]),
                )
                for b in blocks_raw
                if isinstance(b[4], str) and b[4].strip()
            ]
            pages.append(
                RawPage(
                    pdf_page_number=i + 1,
                    page_label=_page_label(text),
                    raw_text=text,
                    blocks=blocks,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                )
            )

    logger.info(
        "extracted %s: section=%s issue=%s pages=%d",
        pdf_path.name,
        meta.section.value,
        meta.issue_number,
        meta.page_count,
    )
    return RawDocument(meta=meta, pages=pages)
