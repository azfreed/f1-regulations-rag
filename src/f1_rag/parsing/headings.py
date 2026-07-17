"""Heading detection primitives for FIA regulation text.

Important layout fact
---------------------
PyMuPDF ``get_text("text")`` emits the article *number* and its *heading/body* on
**separate lines**. For example a subarticle renders as::

    A1.1
    Overview

and a numbered provision renders as::

    A1.1.1
    The FIA is responsible for ...

Top-level article headings, however, render inline::

    ARTICLE A1: GENERAL PRINCIPLES

So detection is line-stream based (see :mod:`f1_rag.parsing.regulations`): this
module provides the regexes and the heading-vs-provision heuristic used there.

Assumptions
-----------
- Article numbers are ``<SectionLetter><digits>`` with optional ``.<digits>``
  groups (``A1``, ``A1.2``, ``A1.2.2``, ``C3.2.6``).
- Whether a numbered id introduces a *heading* (short title) or a *provision*
  (prose) is decided by looking at the following line, not by numbering depth
  (Section C uses 3-level numbers for headings, Section A uses them for provisions).
"""

from __future__ import annotations

import re

# Inline top-level article: "ARTICLE A1: GENERAL PRINCIPLES"
ARTICLE_INLINE_RE = re.compile(r"^ARTICLE\s+([A-F]\d+)\s*:\s*(.*)$")
# A line that is *only* an article/subarticle number: "A1.1", "A1.1.1", "C3.2.6"
BARE_NUMBER_RE = re.compile(r"^([A-F]\d+(?:\.\d+)+)\s*$")
# A line that is only a page-reference integer (a strong table-of-contents signal).
BARE_INT_RE = re.compile(r"^\d{1,3}$")
# A page-label token like "A6" / "C15" (used for the printed page label).
PAGE_LABEL_RE = re.compile(r"^([A-F]\d{1,3})$")
# Annotation lines that sit under article headings; treated as non-body metadata.
ANNOTATION_RE = re.compile(r"^(Advisory Committee|Governance)\s*:", re.IGNORECASE)

_HEADING_MAX_WORDS = 10


def parent_of(article_number: str) -> str | None:
    """Return the parent article number, or None for a top-level article.

    ``A1.2.2`` -> ``A1.2``; ``A1.2`` -> ``A1``; ``A1`` -> None.
    """

    if "." not in article_number:
        return None
    return article_number.rsplit(".", 1)[0]


def looks_like_heading(text: str) -> bool:
    """True if ``text`` reads like a short title rather than a sentence/prose.

    Used to decide whether the line following a bare number is that item's heading
    (short, no terminal punctuation) or the start of its provision text (prose).
    """

    t = text.strip()
    if not t:
        return False
    if BARE_INT_RE.match(t):  # a page number is never a heading
        return False
    words = t.split()
    if len(words) > _HEADING_MAX_WORDS:
        return False
    # Sentence/lead-in punctuation disqualifies a heading. A trailing ":" almost
    # always introduces a list, i.e. it starts a provision, not a title.
    if t.rstrip().endswith((".", ";", ",", ":")):
        return False
    return True
