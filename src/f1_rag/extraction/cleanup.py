"""Header/footer removal and table-of-contents exclusion.

What it does
------------
Turns :class:`RawPage` objects into :class:`CleanPage` objects by:
1. Removing recurring headers/footers detected by *frequency + position* (text
   that repeats near the top/bottom of many pages), rather than brittle per-file
   regexes. This generalizes across all six PDFs.
2. Flagging table-of-contents pages so they can be excluded from the searchable
   corpus (TOC entries are just article titles + page numbers, which pollute
   retrieval).

Why it exists
-------------
Recurring boilerplate ("SECTION A: ...", the copyright footer) appears on every
page and would otherwise dominate embeddings and keyword search. TOC pages look
like real content to a naive extractor but carry no substantive regulation text.

How TOC detection works
------------------------
A page is TOC-like if it contains the ``CONTENTS`` marker OR a high density of
"dot-leader" lines (``.... 12``) / lines ending in a bare page number next to an
article-title pattern.

Limitations
-----------
- Frequency detection needs several pages to be reliable; for a 1-2 page document
  it falls back to conservative regex-based line removal.
"""

from __future__ import annotations

import re
from collections import Counter

from ..logging_utils import get_logger
from ..models import CleanPage, RawDocument

logger = get_logger(__name__)

# A line is a header/footer candidate if it sits within these fractions of page height.
_TOP_BAND = 0.10
_BOTTOM_BAND = 0.90

# TOC signals.
_CONTENTS_RE = re.compile(r"^\s*CONTENTS", re.IGNORECASE | re.MULTILINE)
# A bare page-reference integer on its own line (TOC entries look like:
# "A4.1" / "Fit and Proper Persons Test" / "17").
_BARE_INT_RE = re.compile(r"^\s*\d{1,3}\s*$")
# A 3-level numbered id (e.g. A1.1.1) reliably marks the start of body text; the
# tables of contents in these PDFs only list down to 2 levels.
_THREE_LEVEL_RE = re.compile(r"^\s*[A-F]\d+\.\d+\.\d+\b")
_TOC_MIN_BARE_INTS = 5

# Conservative fallback boilerplate patterns (used when frequency data is sparse,
# and always applied to strip the header page-label band that frequency misses).
_FALLBACK_BOILERPLATE = [
    re.compile(r"^SECTION\s+[A-F]:", re.IGNORECASE),
    re.compile(r"^\d{4}\s+Formula\s+1", re.IGNORECASE),
    re.compile(r"Fédération Internationale de l|Federation Internationale de l"),
    re.compile(r"^\s*Issue\s+\d+\s*[A-F]?\d*\s*$", re.IGNORECASE),
    re.compile(r"^\d+\s+[A-F]$"),  # the "0  A" section band
    re.compile(r"^[A-F]\d{1,3}$"),  # standalone printed page label, e.g. "A6"
]


def _normalize(line: str) -> str:
    """Collapse whitespace + strip trailing page tokens so repeated lines match."""

    s = re.sub(r"\s+", " ", line.strip())
    # Drop a trailing page-label/number so "... A6" and "... A7" count as the same header.
    s = re.sub(r"\s+([A-F]?\d{1,3})$", "", s)
    return s.casefold()


def _collect_recurring_lines(doc: RawDocument, min_fraction: float = 0.5) -> set[str]:
    """Find normalized lines that recur on at least ``min_fraction`` of pages.

    We only consider blocks physically located in the top/bottom bands, because
    body text can legitimately repeat a phrase without being a header/footer.
    """

    counter: Counter[str] = Counter()
    n_pages = len(doc.pages)
    for page in doc.pages:
        seen_on_page: set[str] = set()
        for block in page.blocks:
            rel_top = block.y0 / page.height if page.height else 0.5
            rel_bottom = block.y1 / page.height if page.height else 0.5
            in_band = rel_top <= _TOP_BAND or rel_bottom >= _BOTTOM_BAND
            if not in_band:
                continue
            for raw_line in block.text.splitlines():
                norm = _normalize(raw_line)
                if len(norm) >= 6 and norm not in seen_on_page:
                    seen_on_page.add(norm)
                    counter[norm] += 1
    threshold = max(2, int(n_pages * min_fraction))
    return {line for line, c in counter.items() if c >= threshold}


def _page_toc_signal(text: str) -> bool:
    """True if a page *looks* like TOC: has the CONTENTS marker or many bare
    page-reference integers. Used together with the front-matter gate below so
    numeric tables deep in the body are not mistaken for a TOC."""

    if _CONTENTS_RE.search(text):
        return True
    bare_ints = sum(1 for ln in text.splitlines() if _BARE_INT_RE.match(ln))
    return bare_ints >= _TOC_MIN_BARE_INTS


def _first_body_page(doc: RawDocument) -> int:
    """1-based page number where body text begins (first 3-level numbered id).

    The tables of contents only enumerate down to 2 levels, so the first page
    containing something like ``A1.1.1`` marks the boundary between front matter
    (cover, conventions, contents) and the searchable body.
    """

    for page in doc.pages:
        for ln in page.raw_text.splitlines():
            if _THREE_LEVEL_RE.match(ln):
                return page.pdf_page_number
    return 1  # no clear body marker: treat everything as body


def clean_document(doc: RawDocument) -> list[CleanPage]:
    """Produce cleaned pages with headers/footers removed and TOC pages flagged."""

    recurring = _collect_recurring_lines(doc)
    use_frequency = len(recurring) > 0
    logger.info(
        "cleanup %s: %d recurring header/footer lines detected (frequency mode=%s)",
        doc.meta.source_filename,
        len(recurring),
        use_frequency,
    )

    # A page is TOC only if it shows a TOC signal AND lies in the front-matter
    # region (before body text begins). This prevents numeric tables deep in the
    # body from being mistaken for a table of contents.
    body_start = _first_body_page(doc)

    clean_pages: list[CleanPage] = []
    for page in doc.pages:
        kept: list[str] = []
        removed: list[str] = []
        for raw_line in page.raw_text.splitlines():
            norm = _normalize(raw_line)
            stripped = raw_line.strip()
            is_boilerplate = (use_frequency and norm in recurring) or any(
                p.search(stripped) for p in _FALLBACK_BOILERPLATE
            )
            if is_boilerplate:
                removed.append(stripped)
            else:
                kept.append(raw_line)

        cleaned_text = "\n".join(kept).strip()
        page_is_toc = page.pdf_page_number < body_start and _page_toc_signal(page.raw_text)

        clean_pages.append(
            CleanPage(
                pdf_page_number=page.pdf_page_number,
                page_label=page.page_label,
                text=cleaned_text,
                is_toc=page_is_toc,
                removed_lines=removed,
            )
        )

    n_toc = sum(1 for p in clean_pages if p.is_toc)
    logger.info(
        "cleanup %s: body starts at page %d, %d TOC pages flagged",
        doc.meta.source_filename,
        body_start,
        n_toc,
    )
    return clean_pages
