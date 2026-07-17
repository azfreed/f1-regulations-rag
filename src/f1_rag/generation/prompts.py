"""Versioned prompt templates.

Why versioned
-------------
Prompts materially change answer behavior, so we track them by version string and
record the version in every experiment. Comparing ``v1`` vs ``v2`` is a first-class
experiment this test bed supports.

Policy baked into the prompts
-----------------------------
- Answer *only* from the provided context (no silent fallback to general model
  knowledge). If the context does not contain the answer, say so explicitly.
- Cite the article labels shown in the context brackets, e.g. ``[Art. D3.1 ...]``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system: str
    user_template: str  # expects {question} and {context}

    def render_user(self, question: str, context: str) -> str:
        return self.user_template.format(question=question, context=context)


_V1_SYSTEM = (
    "You are an assistant that answers questions strictly about the 2026 FIA "
    "Formula 1 Regulations. You must follow these rules without exception:\n"
    "1. Use ONLY the information in the provided context. Do not rely on prior "
    "knowledge of F1 or its regulations.\n"
    "2. Every factual claim must cite the relevant article using the bracketed "
    "labels exactly as they appear in the context, e.g. [Art. D3.1 (Section D, p.D12)].\n"
    "3. If the context does not contain enough information to answer, reply with "
    "exactly: 'UNANSWERABLE: the provided regulations do not cover this.' and stop.\n"
    "4. Be precise and concise. Do not speculate or generalize beyond the text."
)

_V1_USER = (
    "Question:\n{question}\n\n"
    "Context (each block is a regulation excerpt with its citation label):\n"
    "-----\n{context}\n-----\n\n"
    "Answer the question using only the context above, citing article labels inline."
)


PROMPTS: dict[str, PromptTemplate] = {
    "v1": PromptTemplate(version="v1", system=_V1_SYSTEM, user_template=_V1_USER),
}

DEFAULT_PROMPT_VERSION = "v1"


def get_prompt(version: str | None = None) -> PromptTemplate:
    return PROMPTS[version or DEFAULT_PROMPT_VERSION]
