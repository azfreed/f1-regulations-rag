"""Tracing stage: capture every decision a query makes, for inspection.

- :mod:`f1_rag.tracing.models` defines the :class:`QueryTrace` schema.
- :mod:`f1_rag.tracing.recorder` accumulates a trace during a query.
- :mod:`f1_rag.tracing.render` prints a human-readable trace and writes JSON.
"""

from __future__ import annotations

from .models import CandidateTrace, QueryTrace
from .recorder import TraceRecorder
from .render import render_trace_text, write_trace_json

__all__ = [
    "CandidateTrace",
    "QueryTrace",
    "TraceRecorder",
    "render_trace_text",
    "write_trace_json",
]
