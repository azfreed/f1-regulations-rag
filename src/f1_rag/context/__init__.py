"""Context stage: select and pack chunks into the prompt context.

See :mod:`f1_rag.context.assembly`. Implementations register in
:data:`f1_rag.context.assembly.assembler_registry` so context strategies are
swappable (``--context ...`` in a future milestone).
"""

from __future__ import annotations

from .assembly import ContextAssembler, GreedyContextAssembler, assembler_registry

__all__ = ["ContextAssembler", "GreedyContextAssembler", "assembler_registry"]
