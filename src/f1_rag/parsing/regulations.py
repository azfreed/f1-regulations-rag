"""Structural segmentation: assemble flat :class:`Article` records from clean pages.

What it does
------------
Walks cleaned, non-TOC pages as a single line stream and reconstructs
article/subarticle/provision records. Because PyMuPDF emits the number and its
heading/text on separate lines (see :mod:`headings`), the parser uses look-ahead:
when it sees a bare number line, it peeks at the next line to decide whether the
number introduces a heading (short title) or a provision (prose), then accumulates
the following body lines until the next boundary.

Why it exists
-------------
Splitting the corpus at article/subarticle boundaries is what lets chunks map
cleanly to citable regulation references.

Assumptions / limitations
--------------------------
- Deep list items (``a.``, ``i.``) stay inside their provision's text.
- ``Advisory Committee:`` / ``Governance:`` annotation lines are skipped.
- A provision split across a page boundary is attributed to the page where its id
  line appears.
"""

from __future__ import annotations

from ..logging_utils import get_logger
from ..models import Article, CleanPage, RegulationSection, SourceDocumentMeta
from .headings import (
    ANNOTATION_RE,
    ARTICLE_INLINE_RE,
    BARE_NUMBER_RE,
    looks_like_heading,
    parent_of,
)

logger = get_logger(__name__)


def parse_articles(
    meta: SourceDocumentMeta, clean_pages: list[CleanPage]
) -> list[Article]:
    """Group cleaned body text into flat article/subarticle/provision records."""

    section: RegulationSection = meta.section
    articles: list[Article] = []
    current: Article | None = None
    current_heading: str | None = None  # most recent article/subarticle heading in scope

    def flush() -> None:
        nonlocal current
        if current is not None:
            current.text = current.text.strip()
            if current.text or current.heading:
                articles.append(current)
        current = None

    def start(number: str, heading: str | None, text: str, page: CleanPage) -> Article:
        return Article(
            section=section,
            article_number=number,
            parent_article=parent_of(number),
            heading=heading,
            text=text,
            pdf_page_number=page.pdf_page_number,
            page_label=page.page_label,
            source_filename=meta.source_filename,
        )

    # Flatten all non-TOC pages into (line, page) pairs so look-ahead can cross
    # page boundaries cleanly.
    stream: list[tuple[str, CleanPage]] = []
    for page in clean_pages:
        if page.is_toc:
            continue
        for raw_line in page.text.splitlines():
            s = raw_line.strip()
            if s:
                stream.append((s, page))

    i = 0
    n = len(stream)
    while i < n:
        line, page = stream[i]

        # 1) Inline top-level article heading.
        m = ARTICLE_INLINE_RE.match(line)
        if m:
            flush()
            number = m.group(1)
            inline_heading = m.group(2).strip()
            heading = inline_heading
            consumed = 1
            # Some article headings wrap onto the next line; if inline part is empty
            # or looks unfinished, borrow the next heading-like line.
            if not heading and i + 1 < n and looks_like_heading(stream[i + 1][0]):
                heading = stream[i + 1][0].strip()
                consumed = 2
            current_heading = heading or current_heading
            current = start(number, heading or None, "", page)
            i += consumed
            continue

        # 2) Bare number line: decide heading vs provision by look-ahead.
        m = BARE_NUMBER_RE.match(line)
        if m:
            number = m.group(1)
            nxt = stream[i + 1][0] if i + 1 < n else ""
            if looks_like_heading(nxt):
                # Subarticle heading (e.g. "A1.1" / "Overview").
                flush()
                current_heading = nxt.strip()
                current = start(number, current_heading, "", page)
                i += 2
                continue
            # Numbered provision (e.g. "A1.1.1" / prose...). Heading inherited.
            flush()
            current = start(number, current_heading, nxt.strip(), page)
            i += 2
            continue

        # 3) Annotation lines under a heading are not body text.
        if ANNOTATION_RE.match(line):
            i += 1
            continue

        # 4) Ordinary body line: append to the current record's text.
        if current is not None:
            current.text += ("\n" if current.text else "") + line
        i += 1

    flush()
    logger.info(
        "parsed %s: %d article/subarticle records",
        meta.source_filename,
        len(articles),
    )
    return articles
