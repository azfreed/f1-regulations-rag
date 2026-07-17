"""Extraction stage: turn PDFs into raw text pages and visual metadata.

Two independent sub-stages, deliberately kept separate:

- :mod:`f1_rag.extraction.text` reads text + positioned blocks with PyMuPDF.
- :mod:`f1_rag.extraction.visuals` renders page images and detects diagrams,
  drawings, tables, equations, and embedded images.

:mod:`f1_rag.extraction.cleanup` then removes recurring headers/footers and flags
tables-of-contents so downstream stages see clean body text.
"""

from __future__ import annotations
