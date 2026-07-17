"""Parsing stage: detect article structure and assemble article records.

- :mod:`f1_rag.parsing.headings` recognizes article and subarticle heading lines.
- :mod:`f1_rag.parsing.regulations` walks cleaned pages and groups text under the
  article/subarticle it belongs to, producing flat :class:`Article` records.
"""

from __future__ import annotations
