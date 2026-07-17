"""Structured, human-friendly logging for the pipeline.

Why this exists
---------------
Every stage emits diagnostics. We want a single, consistent logger configuration
so that stage output is easy to read on the console and easy to grep. This module
keeps logging setup in one place instead of scattering ``basicConfig`` calls.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging once, idempotently.

    Level resolution order: explicit ``level`` arg, then ``F1RAG_LOG_LEVEL`` env
    var, then ``INFO``.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = level or os.environ.get("F1RAG_LOG_LEVEL", "INFO")
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, ensuring logging is configured first."""

    configure_logging()
    return logging.getLogger(name)
