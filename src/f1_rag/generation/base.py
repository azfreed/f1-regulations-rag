"""Generator interface + citation extraction shared by generators.

The generator receives the question and the assembled context and returns an
:class:`Answer`. Citation extraction is shared here: we parse the ``[Art. ...]``
labels the model actually used and map them back to the chunks in context, so
citations reflect what was cited (not merely what was retrieved).
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from ..models import AssembledContext, Citation
from ..registry import Registry

generator_registry: Registry["Generator"] = Registry("generator")

# Matches the citation headers we emit in context, e.g.
# [Art. D3.1 (Section D, p.D12)]  -> captures the article number "D3.1".
_CITE_RE = re.compile(r"\[Art\.\s+([A-F]\d+(?:\.\d+)*)")


@runtime_checkable
class Generator(Protocol):
    name: str

    def generate(self, question: str, context: AssembledContext, prompt_version: str | None = None):
        ...


def extract_citations(answer_text: str, context: AssembledContext) -> list[Citation]:
    """Map article labels used in the answer back to the chunks in context.

    Only chunks that are (a) present in the context and (b) referenced by the answer
    become citations. This is what lets us later measure citation accuracy and
    unsupported-claim rate.
    """

    cited_numbers = set(_CITE_RE.findall(answer_text))
    citations: list[Citation] = []
    seen: set[str] = set()
    for cand in context.selected:
        chunk = cand.chunk
        if chunk.article_number in cited_numbers and chunk.chunk_id not in seen:
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    label=chunk.citation_label(),
                    article_number=chunk.article_number,
                    section=chunk.section,
                    page_label=chunk.page_label,
                    pdf_page_number=chunk.pdf_page_number,
                    source_filename=chunk.source_filename,
                )
            )
            seen.add(chunk.chunk_id)
    return citations
