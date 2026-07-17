"""f1_rag: an inspectable, stage-by-stage RAG test bed over the FIA F1 Regulations.

The package is intentionally organized so that every RAG pipeline stage lives in
its own module with a typed interface (a ``Protocol`` or ABC), one or more
standalone implementations, and a registry entry so implementations can be
selected by name at runtime. See ``docs/architecture.md`` for the big picture.
"""

from __future__ import annotations

__version__ = "0.1.0"
