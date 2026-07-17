"""Clear, specific exception types for the pipeline.

Using explicit exceptions (rather than bare ``ValueError`` / ``RuntimeError``
everywhere) makes failures self-describing and lets callers handle categories of
problems distinctly.
"""

from __future__ import annotations


class F1RagError(Exception):
    """Base class for all f1_rag errors."""


class ConfigurationError(F1RagError):
    """Raised when configuration or environment is invalid or incomplete."""


class RegistryError(F1RagError):
    """Raised when a component name is requested but not registered."""


class ExtractionError(F1RagError):
    """Raised when a PDF cannot be read or extracted."""


class ParsingError(F1RagError):
    """Raised when document structure cannot be parsed."""


class IndexError_(F1RagError):
    """Raised for index build/load problems (named to avoid shadowing builtin)."""


class GenerationError(F1RagError):
    """Raised when answer generation fails or is misconfigured.

    Notably raised when a generator is asked to answer with no retrieved context,
    to guarantee the system never silently falls back to general model knowledge.
    """
