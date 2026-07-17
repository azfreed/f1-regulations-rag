"""Generation stage: turn context + question into a cited answer.

Generators register in :data:`f1_rag.generation.base.generator_registry`
(``--generator anthropic``). The Anthropic import is guarded so the package is
importable without the optional dependency.
"""

from __future__ import annotations

from .base import Generator, generator_registry
from .prompts import PROMPTS, get_prompt

try:  # optional dependency
    from . import anthropic_client  # noqa: F401
except Exception:  # noqa: BLE001
    pass

__all__ = ["Generator", "generator_registry", "PROMPTS", "get_prompt"]
